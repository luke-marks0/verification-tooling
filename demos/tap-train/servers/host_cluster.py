"""Host Cluster: runs deterministic LoRA training in-process.

`POST /train` accepts SignedEnvelope<TrainRequest>, trains a LoRA adapter
deterministically (workflows.deterministic_lora_training.train_once), hashes
the adapter directory, and returns SignedEnvelope<TrainResponse>. The adapter
itself is kept on disk and retrievable via `GET /adapter/<digest>` as a
tar.gz stream.

`--mock` skips torch/transformers entirely and returns a synthetic digest
computed from `canonical_json_bytes(TrainRequest.model_dump())`. Two clusters
in mock mode fed the same TrainRequest produce the same digest, so /verify
naturally returns is_verified=true. Mock mode is for CPU smoke; it does not
prove training determinism.

Training is single-threaded: each `/train` call acquires `STATE.train_lock`
for the duration of the run; concurrent callers block. Re-training on the
Recomp side has the same lock semantics.

`/health` returns 200 once the cluster has finished its boot-side warm-up:
- mock mode: immediate
- real mode: torch/transformers imports succeed (we do NOT pre-train on boot)
"""
from __future__ import annotations

import argparse
import io
import json
import signal
import sys
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from servers.envelope import (
    SignedEnvelope,
    TrainRequest,
    TrainResponse,
    sign,
    synthetic_mock_digest,
    verify,
)


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

class ClusterState:
    def __init__(self) -> None:
        self.is_warm: bool = False
        self.mock: bool = False
        self.adapters_dir: Path = Path("/tmp/host-cluster/adapters")
        # digest -> adapter directory on disk
        self.adapters: dict[str, Path] = {}
        # Single-threaded training: only one /train at a time.
        self.train_lock = threading.Lock()
        # Guards `adapters` dict mutations independent of train_lock.
        self.adapters_lock = threading.Lock()


STATE = ClusterState()


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------

def _boot_thread(args: argparse.Namespace) -> None:
    """Confirm imports succeed (real mode) and flip is_warm = True.

    We deliberately do NOT pre-train on boot — that would take minutes and
    block start_servers.sh. Imports alone are enough to detect a torch/peft
    install problem before the first /train hits.
    """
    if args.mock:
        STATE.is_warm = True
        sys.stderr.write("[host_cluster] mock mode; warm immediately\n")
        return

    try:
        # Import once on boot so the cost is paid before /health is asked.
        from workflows.deterministic_lora_training import train_once  # noqa: F401
        sys.stderr.write("[host_cluster] training module imports OK\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[host_cluster] training imports FAILED: {exc}\n")
        # Refuse to flip is_warm so start_servers.sh times out loud.
        return

    STATE.is_warm = True
    sys.stderr.write("[host_cluster] ready\n")


# ---------------------------------------------------------------------------
# Training paths
# ---------------------------------------------------------------------------

def _cfg_from_request(req: TrainRequest) -> dict:
    """Translate the wire-level TrainRequest into the dict shape
    `workflows.deterministic_lora_training.train_once` expects.
    """
    return {
        "base_model": req.base_model,
        "lora_rank": req.lora.r,
        "lora_alpha": req.lora.alpha,
        "lora_dropout": req.lora.dropout,
        "lora_target_modules": list(req.lora.target_modules),
        "batch_size": req.hp.batch_size,
        "max_steps": req.hp.max_steps,
        "learning_rate": req.hp.learning_rate,
        "seq_len": req.hp.seq_len,
        "seed": req.hp.seed,
        "dtype": req.hp.dtype,
        "num_examples": req.dataset.num_examples,
    }


def _build_dataset(req: TrainRequest) -> list[dict[str, str]]:
    """Materialize the dataset by named builder + seed (same on Host and Recomp)."""
    if req.dataset.builder == "benign_arithmetic":
        from workflows.deterministic_lora_training import benign_arithmetic_dataset
        return benign_arithmetic_dataset(req.dataset.num_examples, req.dataset.seed)
    raise ValueError(f"unknown dataset.builder: {req.dataset.builder}")


def _do_real_training(req: TrainRequest) -> TrainResponse:
    """Run train_once with cfg+dataset derived from the wire request; rename
    the output dir to the digest so /adapter/<digest> can find it later."""
    import tempfile

    from workflows.deterministic_lora_training import train_once

    cfg = _cfg_from_request(req)
    dataset = _build_dataset(req)

    # Train into a temp directory; rename to the digest on success.
    STATE.adapters_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="tap-train-", dir=str(STATE.adapters_dir)))
    result = train_once(tmp_dir, cfg=cfg, dataset=dataset)

    digest = result["adapter_digest"]
    final_path = STATE.adapters_dir / digest.replace(":", "_")
    # If the digest already exists (same request again), keep the older one
    # and remove the duplicate to save disk.
    if final_path.exists():
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        tmp_dir.rename(final_path)

    with STATE.adapters_lock:
        STATE.adapters[digest] = final_path

    return TrainResponse(
        adapter_digest=digest,
        final_loss=float(result["final_loss"]),
        loss_trajectory=[float(x) for x in result["loss_trajectory"]],
        n_steps=int(result["n_steps"]),
        n_params_trainable=int(result["n_params_trainable"]),
    )


def _do_mock_training(req: TrainRequest) -> TrainResponse:
    """Synthetic deterministic response. No torch import; no disk write."""
    return TrainResponse(
        adapter_digest=synthetic_mock_digest(req),
        final_loss=0.0,
        loss_trajectory=[],
        n_steps=int(req.hp.max_steps),
        n_params_trainable=0,
    )


# ---------------------------------------------------------------------------
# Adapter tarball serving
# ---------------------------------------------------------------------------

def _tar_gz_bytes(adapter_dir: Path) -> bytes:
    """Build a .tar.gz of an adapter directory, sorted for stability."""
    buf = io.BytesIO()
    # `tarfile.open(... "w:gz")` writes the gzip mtime header which is
    # nondeterministic across runs; for the demo we accept that (the digest
    # already lives in the URL, the body just has to round-trip).
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(adapter_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = path.relative_to(adapter_dir).as_posix()
            tar.add(str(path), arcname=arcname)
    return buf.getvalue()


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

    def _send_bytes(self, code: int, body: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            if STATE.is_warm:
                return self._send_json(200, {"status": "ok"})
            return self._send_json(503, {"status": "warming"})

        if self.path.startswith("/adapter/"):
            digest = self.path[len("/adapter/"):]
            if STATE.mock:
                return self._send_json(404, {"error": "mock mode: no adapter on disk"})
            with STATE.adapters_lock:
                adapter_path = STATE.adapters.get(digest)
            if adapter_path is None or not adapter_path.exists():
                return self._send_json(404, {"error": "unknown digest"})
            try:
                blob = _tar_gz_bytes(adapter_path)
            except Exception as exc:  # noqa: BLE001
                return self._send_json(500, {"error": f"tar.gz failed: {exc}"})
            filename = digest.replace(":", "_") + ".tar.gz"
            return self._send_bytes(200, blob, "application/gzip", filename=filename)

        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/train":
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
            inner = TrainRequest.model_validate(req_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner request: {exc}"})

        # Serialize training; one /train at a time.
        with STATE.train_lock:
            try:
                if STATE.mock:
                    resp = _do_mock_training(inner)
                else:
                    resp = _do_real_training(inner)
            except Exception as exc:  # noqa: BLE001
                return self._send_json(500, {"error": f"training failed: {exc}"})

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
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-train Host Cluster")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--adapters-dir", default="/tmp/host-cluster/adapters")
    parser.add_argument("--mock", action="store_true",
                        help="Skip torch/transformers; return synthetic digest keyed off the request")
    args = parser.parse_args()

    STATE.mock = args.mock
    STATE.adapters_dir = Path(args.adapters_dir)
    STATE.adapters_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadedHTTPServer((args.host, args.port), HostHandler)
    _install_shutdown(server)

    threading.Thread(target=_boot_thread, args=(args,), daemon=True).start()

    print(f"[host_cluster] listening on {args.host}:{args.port}; mock={args.mock}; adapters_dir={STATE.adapters_dir}")
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
