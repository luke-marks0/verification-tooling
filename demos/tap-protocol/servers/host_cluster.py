"""Host Cluster: shim that Popens the deterministic vLLM child and proxies
SignedEnvelope<InferenceRequest> -> chat/completions -> SignedEnvelope<InferenceResponse>.

`--mock` skips the child entirely and returns a canned deterministic string;
both clusters in mock mode produce the same output so /verify naturally
returns is_verified=true. Mock mode is for CPU smoke; it does not prove
determinism.

`/health` returns 200 only after warm-up succeeds (one canned chat/completions
call against the child), so callers polling /health get a single signal for
"ready to serve."
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
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

from servers.envelope import (
    InferenceRequest,
    InferenceResponse,
    SignedEnvelope,
    sign,
    verify,
)


# Fallback model id used when /v1/models cannot be queried (mock mode, etc.).
# In real-vLLM mode the actual served name is resolved from the child's
# /v1/models response at boot — vLLM serves under the path passed via
# --model, which is either "Qwen/Qwen3-1.7B" or the local snapshot path
# (when RUNNER_MODEL_PATH is set, per plan §3).
DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

class ClusterState:
    def __init__(self) -> None:
        self.is_warm: bool = False
        self.proxy_port: int = 0
        self.mock: bool = False
        self.model_id: str = DEFAULT_MODEL_ID
        self.vllm_proc: subprocess.Popen | None = None
        self.lock = threading.Lock()


STATE = ClusterState()


def _start_vllm_child(manifest_path: str, proxy_port: int, vllm_port: int, out_dir: str) -> subprocess.Popen:
    """Popen the deterministic vLLM wrapper at modules/inference/server/main.py."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "modules" / "inference" / "server" / "main.py"),
        "--manifest", manifest_path,
        "--skip-boot-validation",
        "--port", str(proxy_port),
        "--vllm-port", str(vllm_port),
        "--out-dir", out_dir,
    ]
    sys.stderr.write(f"[host_cluster] launching child: {' '.join(cmd)}\n")
    return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


def _resolve_model_id(proxy_port: int) -> str:
    """Ask vLLM what name it's serving under. With RUNNER_MODEL_PATH set the
    served name is the snapshot path, not the HF hub id — the shim must use
    whichever vLLM returns or every chat/completions call 404s."""
    try:
        with urlopen(f"http://127.0.0.1:{proxy_port}/v1/models", timeout=10) as resp:
            payload = json.loads(resp.read())
        data = payload.get("data") or []
        if data and isinstance(data[0], dict) and data[0].get("id"):
            return data[0]["id"]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[host_cluster] _resolve_model_id failed: {exc}; falling back to default\n")
    return DEFAULT_MODEL_ID


def _poll_child_health(proxy_port: int, deadline_s: float = 300.0) -> bool:
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{proxy_port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _warm_up(proxy_port: int) -> bool:
    """Send one canned chat/completions call to flush vLLM's first-iteration init."""
    body = json.dumps({
        "model": STATE.model_id,
        "messages": [{"role": "user", "content": "warmup"}],
        "max_tokens": 4,
        "temperature": 0,
        "seed": 42,
    }).encode("utf-8")
    req = Request(
        f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=300) as resp:
            _ = resp.read()
        return True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[host_cluster] warm-up failed: {exc}\n")
        return False


def _boot_thread(args: argparse.Namespace) -> None:
    """Run in a background thread so the HTTP server can come up immediately.
    Health stays 503 until warm-up completes."""
    if args.mock:
        STATE.is_warm = True
        sys.stderr.write("[host_cluster] mock mode; warm immediately\n")
        return

    proc = _start_vllm_child(args.manifest, args.proxy_port, args.vllm_port, args.out_dir)
    STATE.vllm_proc = proc

    if not _poll_child_health(args.proxy_port):
        sys.stderr.write("[host_cluster] child /health never became 200; exiting non-zero\n")
        os._exit(1)

    STATE.model_id = _resolve_model_id(args.proxy_port)
    sys.stderr.write(f"[host_cluster] child /health OK; served model_id={STATE.model_id!r}; sending warm-up\n")
    if not _warm_up(args.proxy_port):
        sys.stderr.write("[host_cluster] warm-up did not succeed; exiting non-zero\n")
        os._exit(1)

    STATE.is_warm = True
    sys.stderr.write("[host_cluster] ready\n")


# ---------------------------------------------------------------------------
# Inference paths
# ---------------------------------------------------------------------------

def _do_real_inference(prompt: str, max_tokens: int) -> str:
    """POST to local proxy and return the assistant message content."""
    body = json.dumps({
        "model": STATE.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
    }).encode("utf-8")
    req = Request(
        f"http://127.0.0.1:{STATE.proxy_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=300) as resp:
        payload = json.loads(resp.read())
    return payload["choices"][0]["message"]["content"]


def _do_mock_inference(prompt: str, max_tokens: int) -> str:
    """Canned deterministic output keyed off (prompt, max_tokens)."""
    return f"MOCK[{prompt[:32]}|max={max_tokens}]"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class HostHandler(BaseHTTPRequestHandler):

    def _send_json(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            if STATE.is_warm:
                return self._send_json(200, {"status": "ok"})
            return self._send_json(503, {"status": "warming"})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/request":
            return self._send_json(404, {"error": "not found"})
        if not STATE.is_warm:
            return self._send_json(503, {"error": "not warm"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req_env = SignedEnvelope.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad envelope: {exc}"})

        if not verify(req_env):
            return self._send_json(401, {"error": "bad request signature"})

        try:
            inner = InferenceRequest.model_validate(req_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner request: {exc}"})

        try:
            if STATE.mock:
                output = _do_mock_inference(inner.prompt, inner.max_tokens)
            else:
                output = _do_real_inference(inner.prompt, inner.max_tokens)
        except HTTPError as exc:
            return self._send_json(500, {"error": f"upstream HTTP {exc.code}"})
        except URLError as exc:
            return self._send_json(502, {"error": f"upstream unreachable: {exc.reason}"})
        except Exception as exc:  # noqa: BLE001
            return self._send_json(500, {"error": f"inference failed: {exc}"})

        resp = InferenceResponse(output=output)
        signed_resp = sign(resp.model_dump(), req_env.data.id)
        return self._send_json(200, signed_resp.model_dump())

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[host_cluster] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _install_shutdown(server: HTTPServer) -> None:
    def shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[host_cluster] shutting down\n")
        if STATE.vllm_proc and STATE.vllm_proc.poll() is None:
            STATE.vllm_proc.terminate()
            try:
                STATE.vllm_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                STATE.vllm_proc.kill()
        # server.shutdown() blocks until serve_forever returns; call it
        # from a daemon thread so the signal handler returns promptly.
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-protocol Host Cluster")
    parser.add_argument("--port", type=int, default=8020, help="Shim listen port")
    parser.add_argument("--proxy-port", type=int, default=8021, help="Internal det-vLLM proxy port")
    parser.add_argument("--vllm-port", type=int, default=8022, help="Internal vLLM port")
    parser.add_argument("--manifest", default=str(DEMO_DIR / "qwen3-1.7b-tap.manifest.json"))
    parser.add_argument("--out-dir", default="/tmp/host-cluster")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mock", action="store_true", help="Skip Popen; serve canned outputs")
    args = parser.parse_args()

    STATE.proxy_port = args.proxy_port
    STATE.mock = args.mock

    server = ThreadedHTTPServer((args.host, args.port), HostHandler)
    _install_shutdown(server)

    threading.Thread(target=_boot_thread, args=(args,), daemon=True).start()

    print(f"[host_cluster] listening on {args.host}:{args.port}; proxy_port={args.proxy_port}; mock={args.mock}")
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
