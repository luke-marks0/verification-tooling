"""Recomp Cluster: re-runs the LoRA training and bitwise-compares adapter digests.

Exposes `/verify` (not `/train`). Re-runs `train_once` with cfg+dataset
derived from the inbound TrainRequest payload, then compares the resulting
`adapter_digest` against the Host's `response_data.payload.adapter_digest`.

Mismatch path appends a JSON line to `${OUT_DIR}/alarm.jsonl` (opened with
"a", never truncated) and prints a single `[ALARM]` line to stderr.

`--mock` skips torch/transformers and returns `synthetic_mock_digest(req)`,
which matches Host's mock digest iff the request payloads are equal.
`--mock-output-override "<digest>"` overrides the recomputed digest to force
an alarm.

Single-threaded re-training: each /verify acquires `STATE.train_lock`.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEMO_DIR = Path(__file__).resolve().parent.parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from modules.core.common.deterministic import canonical_json_text, utc_now_iso
from servers.envelope import (
    SignedEnvelope,
    TrainRequest,
    TrainResponse,
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
        self.mock_output_override: str | None = None
        self.out_dir: Path = Path("/tmp/recomp-cluster")
        self.adapters_dir: Path = Path("/tmp/recomp-cluster/adapters")
        self.train_lock = threading.Lock()
        self.alarm_lock = threading.Lock()


STATE = ClusterState()


def _boot_thread(args: argparse.Namespace) -> None:
    """Same shape as Host: confirm imports, then flip is_warm."""
    if args.mock:
        STATE.is_warm = True
        sys.stderr.write("[recomp_cluster] mock mode; warm immediately\n")
        return

    try:
        from workflows.deterministic_lora_training import train_once  # noqa: F401
        sys.stderr.write("[recomp_cluster] training module imports OK\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[recomp_cluster] training imports FAILED: {exc}\n")
        return

    STATE.is_warm = True
    sys.stderr.write("[recomp_cluster] ready\n")


# ---------------------------------------------------------------------------
# Re-training
# ---------------------------------------------------------------------------

def _cfg_from_request(req: TrainRequest) -> dict:
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
    if req.dataset.builder == "benign_arithmetic":
        from workflows.deterministic_lora_training import benign_arithmetic_dataset
        return benign_arithmetic_dataset(req.dataset.num_examples, req.dataset.seed)
    raise ValueError(f"unknown dataset.builder: {req.dataset.builder}")


def _do_real_retraining(req: TrainRequest) -> dict:
    """Re-run train_once with identical cfg/dataset; return the train_once result.

    The adapter is written into a unique subdir of `adapters_dir` keyed off
    the digest so repeated /verify calls don't collide. We do NOT remove the
    on-disk adapter — keeping it makes mismatch diagnosis (diff against the
    Host's tarball) possible.
    """
    import tempfile

    from workflows.deterministic_lora_training import train_once

    cfg = _cfg_from_request(req)
    dataset = _build_dataset(req)

    STATE.adapters_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="recomp-train-", dir=str(STATE.adapters_dir)))
    return train_once(tmp_dir, cfg=cfg, dataset=dataset)


def _do_mock_retraining(req: TrainRequest) -> dict:
    """Return a synthetic result that matches Host's mock unless override is set."""
    digest = STATE.mock_output_override if STATE.mock_output_override is not None else synthetic_mock_digest(req)
    return {
        "adapter_digest": digest,
        "final_loss": 0.0,
        "loss_trajectory": [],
        "n_steps": int(req.hp.max_steps),
        "n_params_trainable": 0,
    }


# ---------------------------------------------------------------------------
# Alarm
# ---------------------------------------------------------------------------

def _log_alarm(record: dict) -> None:
    path = STATE.out_dir / "alarm.jsonl"
    STATE.out_dir.mkdir(parents=True, exist_ok=True)
    with STATE.alarm_lock:
        with open(path, "a", encoding="utf-8") as f:
            # canonical_json_text already terminates with \n
            f.write(canonical_json_text(record))
    sys.stderr.write(f"[ALARM] id={record.get('id')} reason={record.get('reason')}\n")


def _first_divergent_step(a: list, b: list) -> int:
    """Index of the first differing element, or -1 if equal end-to-end."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return -1


def _train_request_summary(req: TrainRequest) -> dict:
    """Compact representation of the request for the alarm record."""
    return {
        "base_model": req.base_model,
        "weights_revision": req.weights_revision,
        "lora": req.lora.model_dump(),
        "hp": req.hp.model_dump(),
        "dataset": req.dataset.model_dump(),
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class RecompHandler(BaseHTTPRequestHandler):

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
        if self.path != "/verify":
            return self._send_json(404, {"error": "not found"})
        if not STATE.is_warm:
            return self._send_json(503, {"error": "not warm"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw)
            req_env = SignedEnvelope.model_validate(body["request_data"])
            resp_env = SignedEnvelope.model_validate(body["response_data"])
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad verify body: {exc}"})

        if not verify(req_env) or not verify(resp_env):
            _log_alarm({
                "id": req_env.data.id,
                "reason": "bad_signature",
                "verified_at": utc_now_iso(),
            })
            return self._send_json(200, {"is_verified": False, "reason": "bad_signature"})

        if req_env.data.id != resp_env.data.id:
            _log_alarm({
                "id": req_env.data.id,
                "response_id": resp_env.data.id,
                "reason": "id_mismatch",
                "verified_at": utc_now_iso(),
            })
            return self._send_json(200, {"is_verified": False, "reason": "id_mismatch"})

        try:
            inner_req = TrainRequest.model_validate(req_env.data.payload)
            inner_resp = TrainResponse.model_validate(resp_env.data.payload)
        except Exception as exc:  # noqa: BLE001
            return self._send_json(400, {"error": f"bad inner payload: {exc}"})

        # Serialize re-training: one /verify at a time.
        with STATE.train_lock:
            try:
                if STATE.mock:
                    recomp_result = _do_mock_retraining(inner_req)
                else:
                    recomp_result = _do_real_retraining(inner_req)
            except Exception as exc:  # noqa: BLE001
                return self._send_json(500, {"error": f"recomp training failed: {exc}"})

        expected_digest = inner_resp.adapter_digest
        actual_digest = recomp_result["adapter_digest"]
        if expected_digest == actual_digest:
            return self._send_json(200, {"is_verified": True})

        _log_alarm({
            "id": req_env.data.id,
            "train_request_summary": _train_request_summary(inner_req),
            "expected_digest": expected_digest,
            "actual_digest": actual_digest,
            "host_final_loss": float(inner_resp.final_loss),
            "recomp_final_loss": float(recomp_result["final_loss"]) if recomp_result.get("final_loss") is not None else None,
            "host_loss_trajectory": [float(x) for x in inner_resp.loss_trajectory],
            "recomp_loss_trajectory": [float(x) for x in recomp_result.get("loss_trajectory") or []],
            "first_divergent_step": _first_divergent_step(
                list(inner_resp.loss_trajectory),
                list(recomp_result.get("loss_trajectory") or []),
            ),
            "reason": "digest_mismatch",
            "verified_at": utc_now_iso(),
        })
        return self._send_json(200, {"is_verified": False, "reason": "digest_mismatch"})

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[recomp_cluster] " + (format % args) + "\n")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _install_shutdown(server: HTTPServer) -> None:
    def shutdown(signum, frame):  # noqa: ARG001
        sys.stderr.write("[recomp_cluster] shutting down\n")
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tap-train Recomp Cluster")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--out-dir", default="/tmp/recomp-cluster",
                        help="Where alarm.jsonl is written.")
    parser.add_argument("--adapters-dir", default=None,
                        help="Where re-trained adapters are saved; defaults to <out-dir>/adapters")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-output-override", default=None,
                        help="If set in mock mode, return this digest from recomp retraining, forcing a mismatch")
    args = parser.parse_args()

    STATE.mock = args.mock
    STATE.mock_output_override = args.mock_output_override
    STATE.out_dir = Path(args.out_dir)
    STATE.out_dir.mkdir(parents=True, exist_ok=True)
    STATE.adapters_dir = Path(args.adapters_dir) if args.adapters_dir else (STATE.out_dir / "adapters")
    STATE.adapters_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadedHTTPServer((args.host, args.port), RecompHandler)
    _install_shutdown(server)

    threading.Thread(target=_boot_thread, args=(args,), daemon=True).start()

    print(f"[recomp_cluster] listening on {args.host}:{args.port}; mock={args.mock}; out_dir={STATE.out_dir}")
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
