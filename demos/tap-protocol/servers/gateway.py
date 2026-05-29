"""Gateway: client-facing edge of the tap-protocol demo.

Accepts a plain `InferenceRequest` JSON from the client, wraps it in a signed
envelope with a monotonic id, relays to the Tap, verifies the response
envelope, unwraps the response, returns it to the client.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Allow `from servers.envelope import ...` when this file is executed directly
# from anywhere via `python3 demos/tap-protocol/servers/gateway.py`.
DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from servers.envelope import (
    InferenceRequest,
    InferenceResponse,
    SignedEnvelope,
    next_id,
    sign,
    verify,
)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class GatewayHandler(BaseHTTPRequestHandler):
    tap_url: str = ""  # set on the class before serve_forever()

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"status": "ok"})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req = InferenceRequest.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad request: {exc}"})

        envelope_id = next_id()
        signed_req = sign(req.model_dump(), envelope_id)

        try:
            data = json.dumps(signed_req.model_dump()).encode("utf-8")
            outbound = Request(
                f"{self.tap_url}/request",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(outbound, timeout=300) as resp:
                resp_body = json.loads(resp.read())
        except HTTPError as exc:
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            return self._send_json(502, {"error": f"tap returned HTTP {exc.code}", "body": err_body})
        except URLError as exc:
            return self._send_json(502, {"error": f"tap unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"tap call failed: {exc}"})

        try:
            signed_resp = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})

        if not verify(signed_resp):
            return self._send_json(502, {"error": "response envelope signature invalid"})

        try:
            inner = InferenceResponse.model_validate(signed_resp.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad inner response: {exc}"})

        return self._send_json(200, inner.model_dump())

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[gateway] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Gateway")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Listen host. Pass 0.0.0.0 in start_servers.sh to expose to LAN/Internet.")
    parser.add_argument("--tap-url", default="http://127.0.0.1:8010")
    args = parser.parse_args()

    GatewayHandler.tap_url = args.tap_url.rstrip("/")
    server = ThreadedHTTPServer((args.host, args.port), GatewayHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[gateway] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[gateway] listening on {args.host}:{args.port}; tap={GatewayHandler.tap_url}")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
