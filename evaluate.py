"""Select retrieval method on BEIR train queries and report the public test split."""
import csv
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from retrieval import (ROOT, METHODS, MiniLM, EvidenceIndex, load_jsonl, load_qrels,
                       query_metrics)


def main():
    encoder = MiniLM()
    with threadpool_limits(limits=2):
        index = EvidenceIndex(encoder)
        queries = {str(q["_id"]): q["text"] for q in load_jsonl(ROOT / "data/scifact/queries.jsonl")}
        validation = load_qrels(ROOT / "data/scifact/qrels/train.tsv")
        test = load_qrels(ROOT / "data/scifact/qrels/test.tsv")
        assert not set(validation).intersection(test)
        corpus_ids = set(index.docids)
        assert all(set(rels).issubset(corpus_ids) for rels in list(validation.values())+list(test.values()))
        assert all(qid in queries for qid in set(validation)|set(test))
        per_query, result = [], {}
        selected = None
        for split, qrels in [("validation", validation), ("test", test)]:
            qids = sorted(qrels, key=int)
            vectors = encoder.encode([queries[qid] for qid in qids], progress=True)
            rows = {method: [] for method in METHODS}
            for qid, vector in zip(qids, vectors):
                rankings, _ = index.rankings(queries[qid], vector)
                for method in METHODS:
                    ids = [index.docids[int(i)] for i in rankings[method]]
                    values = query_metrics(ids, qrels[qid])
                    row = {"split": split, "query_id": qid, "method": method, **values}
                    rows[method].append(row)
                    per_query.append(row)
            result[split] = {method: {metric: float(np.mean([row[metric] for row in rows[method]]))
                                     for metric in ["ndcg_at_10", "recall_at_10", "mrr_at_10"]}
                             for method in METHODS}
            if split == "validation":
                selected = max(METHODS, key=lambda method: result[split][method]["ndcg_at_10"])
            print(split, json.dumps(result[split]), flush=True)
        # Warm-process sequential in-process timings include query encoding and all three rankings.
        probe = [queries[qid] for qid in sorted(validation, key=int)[:50]]
        for query in probe[:3]:
            index.search(query, selected)
        timings = []
        for query in probe:
            start = perf_counter()
            index.search(query, selected)
            timings.append((perf_counter()-start)*1000)
    artifact = ROOT / "artifacts"
    with (artifact / "per_query_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    report = {
        "corpus_documents": len(index.docids), "validation_queries": len(validation),
        "test_queries": len(test), "selected_method": selected,
        "selection_metric": "validation nDCG@10; BEIR train queries used only for method selection",
        "methods": result, "model": index.revision,
        "encoder_dimension": int(index.embeddings.shape[1]), "max_wordpieces": 256,
        "truncated_documents": index.truncated_documents,
        "dataset_sha256": hashlib.sha256((ROOT / "data/scifact.zip").read_bytes()).hexdigest(),
        "local_sequential_search": {"queries": len(timings), "median_ms": float(np.median(timings)),
                                    "p95_ms": float(np.percentile(timings,95)),
                                    "includes": "encoding, lexical/dense scoring, fusion; excludes loading/network"},
        "versions": {p: metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "onnxruntime", "tokenizers", "joblib", "threadpoolctl"]},
    }
    (artifact / "metrics.json").write_text(json.dumps(report, indent=2))
    example_query = queries[sorted(test,key=int)[0]]
    example = {"query": example_query, "method": selected, "hits": index.search(example_query, selected)}
    (artifact / "example_search.json").write_text(json.dumps(example,indent=2))
    print("SELECTED", selected, "TEST", json.dumps(result["test"][selected]), flush=True)


if __name__ == "__main__":
    main()
