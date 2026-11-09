"""Check retrieval metrics, masking, search outputs, and local API behavior."""
import json
import math
import os
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from threadpoolctl import threadpool_limits
from retrieval import ROOT, MiniLM, EvidenceIndex, query_metrics, rrf


def main():
    metric = query_metrics(["a", "x", "b"], {"a": 1, "b": 1})
    expected = 1.5/(1+1/math.log2(3))
    assert abs(metric["ndcg_at_10"]-expected) < 1e-12
    assert metric["recall_at_10"] == 1 and metric["mrr_at_10"] == 1
    assert query_metrics(["x"], {"a": 1}) == {"ndcg_at_10": 0, "recall_at_10": 0, "mrr_at_10": 0}
    order, _ = rrf([0, 1], [1, 2])
    assert int(order[0]) == 1 and len(set(order)) == len(order)
    with threadpool_limits(limits=2):
        encoder = MiniLM()
        short = "Scientific evidence for a biological mechanism."
        solo = encoder.encode([short])[0]
        batch = encoder.encode([short, "A long scientific abstract. "*100])[0]
        np.testing.assert_allclose(solo, batch, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(np.linalg.norm(solo), 1, atol=1e-5)
        index = EvidenceIndex(encoder)
        for method in ["bm25", "dense_minilm", "hybrid_rrf"]:
            hits = index.search(short, method, 5)
            assert len(hits) == 5 and len({h["document_id"] for h in hits}) == 5
            assert all(h["document_id"] in index.docids and np.isfinite(h["score"]) for h in hits)
        try:
            index.search("   ")
        except ValueError:
            pass
        else:
            raise AssertionError("An empty query was accepted")
    # Exercise the actual API, including an error response, on a temporary local port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable, str(ROOT / "serve.py"), "--port", str(port)],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    try:
        ready = False
        for _ in range(80):
            if process.poll() is not None:
                raise RuntimeError(process.stderr.read().decode())
            try:
                with urlopen(base+"/health", timeout=.5) as response:
                    assert json.load(response)["documents"] == len(index.docids)
                ready = True
                break
            except OSError:
                time.sleep(.25)
        assert ready, "API startup timed out"
        with urlopen(base+"/search?"+urlencode({"q": short, "top_k": 3}), timeout=10) as response:
            assert len(json.load(response)["hits"]) == 3
        try:
            urlopen(base+"/search?q=", timeout=5)
        except HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("API did not reject an empty query")
    finally:
        process.terminate()
        process.communicate(timeout=10)
    print("PASS: nDCG/recall/MRR reference example, rank fusion, masked pooling, unit vectors, search IDs, local API and invalid query")


if __name__ == "__main__":
    main()
