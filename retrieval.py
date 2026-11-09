"""Original prototype of lexical, dense, and reciprocal-rank-fusion retrieval."""
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import CountVectorizer
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
METHODS = ["bm25", "dense_minilm", "hybrid_rrf"]


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_qrels(path):
    out = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if int(row["score"]) > 0:
                out.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return out


def top_indices(scores, k):
    # Deterministic corpus-order tie handling is useful for reproducibility.
    return np.argsort(-np.asarray(scores), kind="stable")[:k]


def rrf(left, right, k=60):
    scores = {}
    for ranking in [left, right]:
        for rank, idx in enumerate(ranking, 1):
            scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k+rank)
    ordered = sorted(scores, key=lambda idx: (-scores[idx], idx))
    return np.array(ordered, dtype=int), scores


def query_metrics(ranked_ids, relevant, k=10):
    ranking = ranked_ids[:k]
    gains = np.array([2**relevant.get(str(doc), 0)-1 for doc in ranking], dtype=float)
    discounts = np.log2(np.arange(len(ranking)) + 2)
    dcg = float((gains / discounts).sum())
    ideal = sorted([2**grade-1 for grade in relevant.values()], reverse=True)[:k]
    idcg = float(sum(gain/np.log2(rank+2) for rank, gain in enumerate(ideal)))
    hits = [rank for rank, doc in enumerate(ranking, 1) if str(doc) in relevant]
    return {"ndcg_at_10": dcg/idcg if idcg else 0.0,
            "recall_at_10": len(hits)/len(relevant) if relevant else 0.0,
            "mrr_at_10": 1.0/min(hits) if hits else 0.0}


class MiniLM:
    def __init__(self):
        modeldir = ROOT / "models/minilm"
        self.tokenizer = Tokenizer.from_file(str(modeldir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(modeldir / "model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.truncated_count = 0

    def encode(self, texts, batch_size=32, progress=False):
        result = []
        for start in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(texts[start:start+batch_size])
            self.truncated_count += sum(bool(item.overflowing) for item in encodings)
            batch = {
                "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.array([item.attention_mask for item in encodings], dtype=np.int64),
                "token_type_ids": np.array([item.type_ids for item in encodings], dtype=np.int64),
            }
            hidden = self.session.run(None, {key: val for key, val in batch.items() if key in self.inputs})[0]
            mask = batch["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden*mask).sum(axis=1)/np.maximum(mask.sum(axis=1), 1e-9)
            pooled = pooled/np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9)
            result.append(pooled.astype(np.float32))
            if progress and (start == 0 or (start//batch_size+1) % 20 == 0):
                print(f"Embedded {min(start+batch_size,len(texts))}/{len(texts)}", flush=True)
        return np.vstack(result)


class EvidenceIndex:
    def __init__(self, encoder, rebuild=False):
        self.encoder = encoder
        self.documents = load_jsonl(ROOT / "data/scifact/corpus.jsonl")
        self.docids = [str(doc["_id"]) for doc in self.documents]
        if len(set(self.docids)) != len(self.docids):
            raise ValueError("Duplicate document identifiers")
        texts = [doc["title"].strip()+" "+doc["text"].strip() for doc in self.documents]
        if any(not text.strip() for text in texts):
            raise ValueError("Empty documents")
        self.revision = json.loads((ROOT / "model_revision.json").read_text())
        signature = hashlib.sha256((json.dumps(self.revision)+json.dumps(texts)).encode()).hexdigest()
        artifact = ROOT / "artifacts"
        artifact.mkdir(exist_ok=True)
        cache = artifact / "index.joblib"
        if cache.exists() and not rebuild:
            saved = joblib.load(cache)
            if saved["signature"] != signature:
                raise ValueError("Index cache does not match corpus/model; rebuild it")
            self.vectorizer, self.weighted, self.embeddings = saved["vectorizer"], saved["weighted"], saved["embeddings"]
            self.truncated_documents = saved["truncated_documents"]
        else:
            self.vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
            counts = self.vectorizer.fit_transform(texts).tocsr()
            lengths = np.asarray(counts.sum(axis=1)).ravel()
            df = np.asarray((counts > 0).sum(axis=0)).ravel()
            idf = np.log1p((len(texts)-df+.5)/(df+.5))
            coo = counts.tocoo()
            denominator = coo.data + 1.2*(1-.75+.75*lengths[coo.row]/lengths.mean())
            coo.data = coo.data*(1.2+1)/denominator*idf[coo.col]
            self.weighted = coo.tocsc()
            before = encoder.truncated_count
            self.embeddings = encoder.encode(texts, progress=True)
            self.truncated_documents = encoder.truncated_count-before
            joblib.dump({"signature": signature, "vectorizer": self.vectorizer,
                         "weighted": self.weighted, "embeddings": self.embeddings,
                         "truncated_documents": self.truncated_documents}, cache, compress=3)
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)

    def rankings(self, query, query_embedding=None, limit=100):
        if not query.strip():
            raise ValueError("Query must contain text")
        sparse_query = self.vectorizer.transform([query])
        sparse_score = np.zeros(len(self.docids), dtype=np.float32)
        for column, count in zip(sparse_query.indices, sparse_query.data):
            a, b = self.weighted.indptr[column:column+2]
            sparse_score[self.weighted.indices[a:b]] += self.weighted.data[a:b]*count
        lexical = top_indices(sparse_score, limit)
        lexical = lexical[sparse_score[lexical] > 0]
        vector = query_embedding if query_embedding is not None else self.encoder.encode([query])[0]
        dense_score = self.embeddings @ vector
        dense = top_indices(dense_score, limit)
        fused, fused_score = rrf(lexical, dense)
        return {"bm25": lexical, "dense_minilm": dense, "hybrid_rrf": fused[:limit]}, {
            "bm25": sparse_score, "dense_minilm": dense_score, "hybrid_rrf": fused_score}

    def search(self, query, method="hybrid_rrf", top_k=5):
        if method not in METHODS or not 1 <= top_k <= 100:
            raise ValueError("Unknown method or top_k outside 1 to 100")
        rankings, scores = self.rankings(query)
        hits = []
        for idx in rankings[method][:top_k]:
            doc = self.documents[int(idx)]
            hits.append({"document_id": self.docids[int(idx)], "title": doc["title"],
                         "abstract": doc["text"], "score": float(scores[method][int(idx)])})
        return hits
