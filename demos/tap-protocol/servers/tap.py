"""Tap: relays SignedEnvelope between Gateway and Host Cluster.

After the response goes back to the Gateway, fires a daemon thread that POSTs
the (request, response) pair to the Recomp Cluster's /verify. Failures are
logged but do not propagate -- verification is async.
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

DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from servers.envelope import SignedEnvelope, verify


def _async_verify(recomp_url: str, request_env: dict, response_env: dict) -> None:
    """Fire-and-forget POST to recomp /verify. Logs verdict; ignores failures."""
    try:
        body = json.dumps({
            "request_data": request_env,
            "response_data": response_env,
        }).encode("utf-8")
        req = Request(
            f"{recomp_url}/verify",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=600) as resp:
            verdict = json.loads(resp.read())
        sys.stderr.write(f"[tap] verify verdict for id={request_env.get('data', {}).get('id')}: {verdict}\n")
    except HTTPError as exc:
        sys.stderr.write(f"[tap] verify HTTP {exc.code}: {exc.reason}\n")
    except URLError as exc:
        sys.stderr.write(f"[tap] verify unreachable: {exc.reason}\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[tap] verify failed: {exc}\n")


class TapHandler(BaseHTTPRequestHandler):
    host_url: str = ""
    recomp_url: str = ""

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
            req_env = SignedEnvelope.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})

        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})

        # Forward verbatim to host cluster
        try:
            outbound = Request(
                f"{self.host_url}/request",
                data=json.dumps(req_env.model_dump()).encode("utf-8"),
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
            return self._send_json(502, {"error": f"host returned HTTP {exc.code}", "body": err_body})
        except URLError as exc:
            return self._send_json(502, {"error": f"host unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"host call failed: {exc}"})

        try:
            resp_env = SignedEnvelope.model_validate(resp_body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(502, {"error": f"bad response envelope: {exc}"})

        if not verify(resp_env):
            return self._send_json(401, {"error": "bad response signature"})

        # Return the verified response envelope to the Gateway first.
        self._send_json(200, resp_env.model_dump())

        # Then spawn the async verification tap-copy.
        threading.Thread(
            target=_async_verify,
            args=(self.recomp_url, req_env.model_dump(), resp_env.model_dump()),
            daemon=True,
        ).start()

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[tap] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Tap")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--host-url", default="http://127.0.0.1:8020")
    parser.add_argument("--recomp-url", default="http://127.0.0.1:8030")
    args = parser.parse_args()

    TapHandler.host_url = args.host_url.rstrip("/")
    TapHandler.recomp_url = args.recomp_url.rstrip("/")

    server = ThreadedHTTPServer((args.host, args.port), TapHandler)

    def _shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[tap] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[tap] listening on {args.host}:{args.port}; host={TapHandler.host_url}; recomp={TapHandler.recomp_url}")
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
