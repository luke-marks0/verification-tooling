"""DCGM SM_ACTIVE sampler.

NVML's GPU utilisation is a binary "any kernel running" reading. To get
the actual fraction of SMs busy, we need DCGM's profiling field
``DCGM_FI_PROF_SM_ACTIVE`` (id 1002), which reports the ratio of cycles
SMs were active averaged across all SMs.

This module shells out to ``dcgmi dmon -e 1002 -d <interval_ms>`` and
parses its line-oriented output. dmon prints one row per sample:

    #Entity   SMACT
    GPU 0     0.512

We pick out the SMACT column and timestamp it locally. ``start()`` spawns
the subprocess and a parser thread; ``stop()`` terminates it; ``summary()``
returns mean/median/max over the active window.

Why subprocess and not pydcgm: pydcgm requires the daemon to be reachable
and the python bindings to be installed in the right interpreter. dcgmi
is a single binary that ships with the DCGM apt package and works as
long as ``nv-hostengine`` is running (or with ``-e`` embedded mode in
recent versions).
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class DcgmSample:
    t_ms: float
    sm_active: float  # 0.0 - 1.0


def dcgmi_available() -> bool:
    return shutil.which("dcgmi") is not None


class DcgmSmActiveSampler:
    """Stream DCGM_FI_PROF_SM_ACTIVE via `dcgmi dmon`.

    interval_ms must be ≥ 100 — DCGM profiling fields can't sample faster
    than that on most cards (the prof engine has fixed-rate counters).
    """

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100):
        if interval_ms < 100:
            interval_ms = 100
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.samples: list[DcgmSample] = []
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0_perf = 0.0

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._t0_perf = time.perf_counter()
        # `-e 1002` = DCGM_FI_PROF_SM_ACTIVE. `-d` is interval in ms.
        # `-c 0` = run forever (we kill it). `-i` = GPU index.
        cmd = ["dcgmi", "dmon", "-e", "1002", "-d", str(self.interval_ms),
               "-i", str(self.gpu_index), "-c", "0"]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, preexec_fn=os.setsid,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Id"):
                continue
            parts = line.split()
            # Expected: ['GPU', '0', '0.512']  — entity, idx, value.
            if len(parts) >= 3 and parts[0] == "GPU":
                try:
                    val = float(parts[-1])
                except ValueError:
                    continue
                t = (time.perf_counter() - self._t0_perf) * 1000.0
                self.samples.append(DcgmSample(t_ms=t, sm_active=val))

    def stop(self) -> None:
        if self._proc is None:
            return
        self._stop.set()
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._reader:
            self._reader.join(timeout=1.0)
        self._proc = None
        self._reader = None

    def stderr_text(self) -> str:
        if self._proc is None:
            return ""
        try:
            return self._proc.stderr.read() or ""
        except Exception:
            return ""

    def summary(self, threshold: float = 0.05) -> dict:
        """Aggregate. ``threshold`` filters out idle samples (below 5 % SM
        active) so the mean/median reflect the active window only.
        """
        if not self.samples:
            return {"sample_count": 0}
        active = [s.sm_active for s in self.samples if s.sm_active >= threshold]
        all_vals = [s.sm_active for s in self.samples]
        if not active:
            active = all_vals
        sa = sorted(active)
        return {
            "sample_count": len(self.samples),
            "active_sample_count": len(active),
            "sm_active_mean": float(sum(active) / len(active)),
            "sm_active_median": float(sa[len(sa) // 2]),
            "sm_active_max": float(max(active)),
            "sm_active_min": float(min(active)),
            "sm_active_all_mean": float(sum(all_vals) / len(all_vals)),
        }


# DCGM profiling fields. See nvidia-dcgm/dcgm_fields.h.
# Field IDs 1009-1012 require DCGM >= 3.x; older versions will reject them at
# `dcgmi dmon` time. tensor_active (1004) is the umbrella counter and rises
# whenever the tensor cores fire — IMMA (integer matmul), HMMA (half/bfloat
# matmul), or DFMA (double matmul). The IMMA/HMMA/DFMA breakdown lets you
# distinguish what kind of tensor work the chip is doing. pipe_int_active
# (1012) catches scalar integer arithmetic on the dedicated INT32 cores
# (separate from FP32 cores on Volta+).
DCGM_FIELDS = {
    "sm_active":          1002,  # ratio of cycles any SM was busy (averaged across all SMs)
    "sm_occupancy":       1003,  # ratio of resident warps : max possible warps per SM
    "tensor_active":      1004,  # any tensor-core work (IMMA + HMMA + DFMA)
    "dram_active":        1005,
    "fp64_active":        1006,
    "fp32_active":        1007,  # ratio of cycles FP32 pipe was active
    "fp16_active":        1008,
    "tensor_imma_active": 1009,  # tensor cores doing INT8/INT4 matmul
    "tensor_hmma_active": 1010,  # tensor cores doing FP16/BF16 matmul
    "tensor_dfma_active": 1011,  # tensor cores doing FP64 matmul (Hopper+)
    "pipe_int_active":    1012,  # scalar integer arithmetic on INT32 cores
}


class DcgmMultiFieldSampler:
    """Stream multiple DCGM profiling fields concurrently via `dcgmi dmon`.

    Used to answer Buck's per-SM-internal-saturation question (2026-04-30):
    SM_ACTIVE alone proves an SM is *scheduled*, not that the FP32/tensor
    pipes inside it are *saturated*. This sampler joins SM_ACTIVE with
    SM_OCCUPANCY (warp residency) and PIPE_FP32_ACTIVE (FP32 pipe utilisation)
    so a single sweep can verify all three layers.

    By default samples ``sm_active``, ``sm_occupancy``, ``fp32_active``,
    ``tensor_active``. Override with ``fields=[...]``.
    """

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100,
                 fields: list[str] | None = None):
        if interval_ms < 100:
            interval_ms = 100
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self.fields = fields or ["sm_active", "sm_occupancy",
                                  "fp32_active", "tensor_active"]
        for f in self.fields:
            if f not in DCGM_FIELDS:
                raise ValueError(f"unknown DCGM field {f!r}; choices: {sorted(DCGM_FIELDS)}")
        # Per-field samples: list of (t_ms, value) tuples.
        self.samples: dict[str, list[tuple[float, float]]] = {f: [] for f in self.fields}
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0_perf = 0.0

    def start(self) -> None:
        for f in self.fields:
            self.samples[f].clear()
        self._stop.clear()
        self._t0_perf = time.perf_counter()
        ids = ",".join(str(DCGM_FIELDS[f]) for f in self.fields)
        cmd = ["dcgmi", "dmon", "-e", ids, "-d", str(self.interval_ms),
               "-i", str(self.gpu_index), "-c", "0"]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, preexec_fn=os.setsid,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        assert self._proc is not None
        n_fields = len(self.fields)
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Id"):
                continue
            parts = line.split()
            # Expected: ['GPU', '<idx>', val1, val2, ..., valN]
            if len(parts) >= 2 + n_fields and parts[0] == "GPU":
                t = (time.perf_counter() - self._t0_perf) * 1000.0
                for i, f in enumerate(self.fields):
                    try:
                        val = float(parts[2 + i])
                    except ValueError:
                        continue
                    self.samples[f].append((t, val))

    def stop(self) -> None:
        if self._proc is None:
            return
        self._stop.set()
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._reader:
            self._reader.join(timeout=1.0)
        self._proc = None
        self._reader = None

    def summary(self, active_threshold_field: str = "sm_active",
                active_threshold: float = 0.05) -> dict:
        """Per-field aggregation, filtered to samples where the gating field
        (default ``sm_active``) crossed ``active_threshold``.

        Restricting to active samples is important: every field is averaged
        across all SMs, so during the launch ramp / teardown the values
        drop and would skew the mean if folded in.
        """
        gate = self.samples.get(active_threshold_field, [])
        active_idx = {i for i, (_, v) in enumerate(gate) if v >= active_threshold}
        out: dict = {"fields": self.fields, "interval_ms": self.interval_ms}
        for f in self.fields:
            vals = self.samples[f]
            if not vals:
                out[f] = {"count": 0}
                continue
            active = [v for i, (_, v) in enumerate(vals) if i in active_idx]
            all_vals = [v for _, v in vals]
            if not active:
                active = all_vals
            sa = sorted(active)
            out[f] = {
                "count": len(vals),
                "active_count": len(active),
                "mean_active": float(sum(active) / len(active)),
                "median_active": float(sa[len(sa) // 2]),
                "max_active": float(max(active)),
                "min_active": float(min(active)),
                "mean_all": float(sum(all_vals) / len(all_vals)),
            }
        return out
