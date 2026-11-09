"""Download the public benchmark and an immutable pretrained model revision."""
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    total = 0
    with urlopen(url, timeout=180) as source, temporary.open("wb") as target:
        while block := source.read(1024*1024):
            target.write(block)
            total += len(block)
            if total % (10*1024*1024) == 0:
                print(path.name, f"{total//(1024*1024)} MiB", flush=True)
    temporary.replace(path)


def main():
    archive = ROOT / "data/scifact.zip"
    download(DATA_URL, archive)
    with ZipFile(archive) as z:
        for name in ["corpus.jsonl", "queries.jsonl", "qrels/train.tsv", "qrels/test.tsv"]:
            dest = ROOT / "data/scifact" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read("scifact/" + name))
    revision = json.loads((ROOT / "model_revision.json").read_text())
    base = f"https://huggingface.co/{revision['model_id']}/resolve/{revision['revision']}"
    for remote, local in [("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")]:
        download(base + "/" + remote, ROOT / "models/minilm" / local)
    print("Prepared SciFact and pinned MiniLM ONNX model", flush=True)


if __name__ == "__main__":
    main()
