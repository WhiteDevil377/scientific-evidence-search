"""Minimal local JSON API; research demonstration only."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from threadpoolctl import threadpool_limits
from retrieval import MiniLM, EvidenceIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    index = EvidenceIndex(MiniLM())

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.respond(200, {"status": "ok", "documents": len(index.docids)})
                return
            if parsed.path != "/search":
                self.respond(404, {"error": "Use /health or /search?q=..."})
                return
            query = parse_qs(parsed.query)
            text = query.get("q", [""])[0]
            try:
                if len(text)>1000:
                    raise ValueError("Query exceeds 1000 characters")
                with threadpool_limits(limits=2):
                    hits = index.search(text, query.get("method", ["hybrid_rrf"])[0],
                                        int(query.get("top_k", ["5"])[0]))
                self.respond(200, {"query": text, "hits": hits})
            except ValueError as exc:
                self.respond(400, {"error": str(exc)})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local API: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
