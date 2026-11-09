"""Search scientific abstracts from the command line."""
import argparse
import json
from threadpoolctl import threadpool_limits
from retrieval import METHODS, MiniLM, EvidenceIndex

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--method", choices=METHODS, default="hybrid_rrf")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        index = EvidenceIndex(MiniLM())
        print(json.dumps(index.search(args.query, args.method, args.top_k), indent=2))
