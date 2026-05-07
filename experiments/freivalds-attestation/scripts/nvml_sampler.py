"""Background NVML sampler.

Spawns a thread that polls one GPU at a fixed interval and records
``(t_ms, sm_util_pct, mem_util_pct, power_w, clock_mhz, temp_c)`` per
sample. ``stop()`` flushes the thread; ``summary()`` returns aggregate
stats (median, max, mean) over the collection window.

We use the C API via ctypes because pynvml isn't always installed and we
already depend on libnvidia-ml.so.1 from the c3 stack.
"""
from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass


_NVML = None
_NVML_INIT = False
_INIT_LOCK = threading.Lock()


def _load_nvml() -> ctypes.CDLL:
    global _NVML
    if _NVML is None:
        for so in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                _NVML = ctypes.CDLL(so)
                break
            except OSError:
                continue
        if _NVML is None:
            raise OSError("libnvidia-ml.so not found")
    return _NVML


class _NvmlUtil(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _nvml_init() -> None:
    global _NVML_INIT
    with _INIT_LOCK:
        if _NVML_INIT:
            return
        rc = _load_nvml().nvmlInit_v2()
        if rc != 0:
            raise RuntimeError(f"nvmlInit_v2 failed rc={rc}")
        _NVML_INIT = True


def _device_handle(idx: int) -> ctypes.c_void_p:
    nvml = _load_nvml()
    h = ctypes.c_void_p()
    rc = nvml.nvmlDeviceGetHandleByIndex_v2(idx, ctypes.byref(h))
    if rc != 0:
        raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2({idx}) rc={rc}")
    return h


def _utilization(handle) -> tuple[int, int]:
    nvml = _load_nvml()
    u = _NvmlUtil()
    rc = nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(u))
    if rc != 0:
        return (-1, -1)
    return (int(u.gpu), int(u.memory))


def _power_watts(handle) -> float:
    nvml = _load_nvml()
    mw = ctypes.c_uint(0)
    rc = nvml.nvmlDeviceGetPowerUsage(handle, ctypes.byref(mw))
    if rc != 0:
        return -1.0
    return mw.value / 1000.0


def _clock_mhz(handle) -> int:
    nvml = _load_nvml()
    mhz = ctypes.c_uint(0)
    # NVML_CLOCK_SM = 1
    rc = nvml.nvmlDeviceGetClockInfo(handle, 1, ctypes.byref(mhz))
    if rc != 0:
        return -1
    return mhz.value


def _temp_c(handle) -> int:
    nvml = _load_nvml()
    t = ctypes.c_uint(0)
    # NVML_TEMPERATURE_GPU = 0
    rc = nvml.nvmlDeviceGetTemperature(handle, 0, ctypes.byref(t))
    if rc != 0:
        return -1
    return t.value


@dataclass
class Sample:
    t_ms: float
    sm_util: int
    mem_util: int
    power_w: float
    clock_mhz: int
    temp_c: int

    def to_dict(self) -> dict:
        return {
            "t_ms": float(self.t_ms),
            "sm_util": int(self.sm_util),
            "mem_util": int(self.mem_util),
            "power_w": float(self.power_w),
            "clock_mhz": int(self.clock_mhz),
            "temp_c": int(self.temp_c),
        }


class NvmlSampler:
    """Threaded NVML poller. Use as a context manager around a workload."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 10) -> None:
        _nvml_init()
        self.handle = _device_handle(gpu_index)
        self.interval_s = interval_ms / 1000.0
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0_perf: float = 0.0

    def __enter__(self) -> "NvmlSampler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._t0_perf = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t = (time.perf_counter() - self._t0_perf) * 1000.0
            sm, mem = _utilization(self.handle)
            self.samples.append(Sample(
                t_ms=t,
                sm_util=sm, mem_util=mem,
                power_w=_power_watts(self.handle),
                clock_mhz=_clock_mhz(self.handle),
                temp_c=_temp_c(self.handle),
            ))
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    def summary(self) -> dict:
        if not self.samples:
            return {"sample_count": 0}
        sm = sorted(s.sm_util for s in self.samples if s.sm_util >= 0)
        pw = [s.power_w for s in self.samples if s.power_w >= 0]
        cl = sorted(s.clock_mhz for s in self.samples if s.clock_mhz >= 0)
        tp = [s.temp_c for s in self.samples if s.temp_c >= 0]
        def _med(xs):
            return float(xs[len(xs)//2]) if xs else -1.0
        def _mean(xs):
            return float(sum(xs) / len(xs)) if xs else -1.0
        return {
            "sample_count": len(self.samples),
            "sm_util_median": _med(sm),
            "sm_util_max": float(sm[-1]) if sm else -1.0,
            "sm_util_min": float(sm[0]) if sm else -1.0,
            "sm_util_mean": _mean(sm),
            "power_w_mean": _mean(pw),
            "power_w_max": float(max(pw)) if pw else -1.0,
            "clock_mhz_median": _med(cl),
            "clock_mhz_max": float(cl[-1]) if cl else -1.0,
            "temp_c_max": float(max(tp)) if tp else -1.0,
            "temp_c_mean": _mean(tp),
        }
