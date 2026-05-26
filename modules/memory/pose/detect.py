"""Auto-detect wipeable memory ceilings per region.

Adapted from github.com/luke-marks0/memory-sanitization.
Uses raw ctypes to libcudart.so (no CuPy dependency) and
OS-level queries for DRAM and disk.
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU detection via ctypes
# ---------------------------------------------------------------------------

class CudaRuntime:
    """Thin ctypes wrapper around libcudart.so."""

    def __init__(self):
        self._lib = ctypes.CDLL("libcudart.so")

        self._lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self._lib.cudaSetDevice.restype = ctypes.c_int

        self._lib.cudaMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.cudaMemGetInfo.restype = ctypes.c_int

        self._lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self._lib.cudaMalloc.restype = ctypes.c_int

        self._lib.cudaFree.argtypes = [ctypes.c_void_p]
        self._lib.cudaFree.restype = ctypes.c_int

        self._lib.cudaMemset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self._lib.cudaMemset.restype = ctypes.c_int

        self._lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self._lib.cudaMemcpy.restype = ctypes.c_int

        self._lib.cudaDeviceSynchronize.argtypes = []
        self._lib.cudaDeviceSynchronize.restype = ctypes.c_int

        self._lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._lib.cudaGetDeviceCount.restype = ctypes.c_int

    def _check(self, err: int, context: str = ""):
        if err != 0:
            raise RuntimeError(f"CUDA error {err} in {context}")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self._lib.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        return count.value

    def set_device(self, device: int):
        self._check(self._lib.cudaSetDevice(device), f"cudaSetDevice({device})")

    def mem_get_info(self, device: int = 0) -> tuple[int, int]:
        """Returns (free_bytes, total_bytes) for the given GPU."""
        self.set_device(device)
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        self._check(
            self._lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)),
            f"cudaMemGetInfo(device={device})",
        )
        return int(free.value), int(total.value)

    def malloc(self, device: int, size: int) -> int:
        """Allocate GPU memory. Returns device pointer as int."""
        self.set_device(device)
        pointer = ctypes.c_void_p()
        self._check(
            self._lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(size)),
            f"cudaMalloc(device={device}, size={size})",
        )
        return int(pointer.value)

    def free(self, device: int, pointer: int):
        self.set_device(device)
        self._check(self._lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def memset(self, device: int, pointer: int, value: int, size: int):
        self.set_device(device)
        self._check(
            self._lib.cudaMemset(ctypes.c_void_p(pointer), value, ctypes.c_size_t(size)),
            "cudaMemset",
        )

    def memcpy_htod(self, device: int, dst_pointer: int, src_buffer: bytes):
        """Copy from host bytes to device pointer."""
        self.set_device(device)
        src = ctypes.c_char_p(src_buffer)
        self._check(
            self._lib.cudaMemcpy(
                ctypes.c_void_p(dst_pointer), src,
                ctypes.c_size_t(len(src_buffer)), 1,  # cudaMemcpyHostToDevice = 1
            ),
            "cudaMemcpy(H2D)",
        )

    def memcpy_dtoh(self, device: int, src_pointer: int, size: int) -> bytes:
        """Copy from device pointer to host bytes."""
        self.set_device(device)
        buf = ctypes.create_string_buffer(size)
        self._check(
            self._lib.cudaMemcpy(
                buf, ctypes.c_void_p(src_pointer),
                ctypes.c_size_t(size), 2,  # cudaMemcpyDeviceToHost = 2
            ),
            "cudaMemcpy(D2H)",
        )
        return buf.raw

    def synchronize(self, device: int):
        self.set_device(device)
        self._check(self._lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


_cuda_runtime = None

def get_cuda_runtime() -> CudaRuntime:
    global _cuda_runtime
    if _cuda_runtime is None:
        _cuda_runtime = CudaRuntime()
    return _cuda_runtime


# ---------------------------------------------------------------------------
# DRAM detection
# ---------------------------------------------------------------------------

def _numa_node_memory_bytes(node: int) -> int | None:
    """Read MemTotal for a specific NUMA node, or None if unavailable."""
    path = Path(f"/sys/devices/system/node/node{node}/meminfo")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if "MemTotal" in line:
            return int(line.split()[3]) * 1024  # kB -> bytes
    return None


def detect_dram_bytes() -> int:
    """Detect host DRAM bytes, preferring NUMA node 0 (LPDDR5X on GH200).

    On GH200, MemTotal includes both LPDDR5X and HBM. NUMA node 0 is
    LPDDR5X only, which is what we want for the DRAM region.
    """
    # Try NUMA node 0 first (excludes GPU HBM on GH200)
    node0 = _numa_node_memory_bytes(0)
    if node0 is not None and node0 > 0:
        return node0

    # Fallback: sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass

    # Fallback: /proc/meminfo
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot detect host DRAM")


def _cgroup_memory_limit() -> int | None:
    """Check cgroup memory limit (container environments)."""
    for path in (Path("/sys/fs/cgroup/memory.max"),
                 Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if not path.exists():
            continue
        raw = path.read_text().strip()
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
            if 0 < value < 2**60:
                return value
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Disk detection
# ---------------------------------------------------------------------------

def detect_disk_bytes(path: str) -> int:
    """Detect available disk bytes at the given mount point."""
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


# ---------------------------------------------------------------------------
# Combined ceiling computation
# ---------------------------------------------------------------------------

@dataclass
class MemoryCeilings:
    dram_total: int
    dram_wipeable: int
    dram_reserved: int
    dram_reserved_reason: str

    hbm_total: int
    hbm_wipeable: int
    hbm_reserved: int
    hbm_reserved_reason: str

    disk_total: int
    disk_wipeable: int
    disk_reserved: int
    disk_reserved_reason: str

    @property
    def total_wipeable(self) -> int:
        return self.dram_wipeable + self.hbm_wipeable + self.disk_wipeable

    @property
    def total_physical(self) -> int:
        return self.dram_total + self.hbm_total + self.disk_total


def compute_ceilings(
    disk_path: str = "/tmp",
    dram_fraction: float = 0.85,
    hbm_fraction: float = 0.90,
    disk_fraction: float = 0.95,
    gpu_device: int = 0,
) -> MemoryCeilings:
    """Compute wipeable memory ceilings using hardware queries + target fractions.

    Args:
        disk_path: Mount point for disk region.
        dram_fraction: Fraction of LPDDR5X node 0 to use (0.85 = 85%).
        hbm_fraction: Fraction of free GPU HBM to use (0.90 = 90%).
        disk_fraction: Fraction of available disk to use (0.95 = 95%).
        gpu_device: CUDA device index.
    """
    block_size = 4096

    # --- DRAM ---
    dram_total = detect_dram_bytes()
    cgroup_limit = _cgroup_memory_limit()
    if cgroup_limit is not None:
        dram_total = min(dram_total, cgroup_limit)
    dram_wipeable = int(dram_total * dram_fraction)
    dram_wipeable = (dram_wipeable // block_size) * block_size  # align
    dram_reserved = dram_total - dram_wipeable

    # --- HBM ---
    try:
        hbm_free, hbm_total = get_cuda_runtime().mem_get_info(gpu_device)
        hbm_wipeable = int(hbm_free * hbm_fraction)
        hbm_wipeable = (hbm_wipeable // block_size) * block_size
        hbm_reserved = hbm_total - hbm_wipeable
    except (RuntimeError, OSError):
        hbm_total = 0
        hbm_free = 0
        hbm_wipeable = 0
        hbm_reserved = 0

    # --- Disk ---
    disk_avail = detect_disk_bytes(disk_path)
    disk_total_st = os.statvfs(disk_path)
    disk_total = disk_total_st.f_blocks * disk_total_st.f_frsize
    disk_wipeable = int(disk_avail * disk_fraction)
    disk_wipeable = (disk_wipeable // block_size) * block_size
    disk_reserved = disk_total - disk_wipeable

    return MemoryCeilings(
        dram_total=dram_total,
        dram_wipeable=dram_wipeable,
        dram_reserved=dram_reserved,
        dram_reserved_reason=(
            "Linux kernel text/data/page tables, slab caches, network stack, "
            "systemd, Python runtime, and AES noise generation buffers. "
            f"Target fraction: {dram_fraction:.0%} of NUMA node 0 ({dram_total / (1024**3):.1f} GiB). "
            "Remaining headroom covers page table overhead (~0.2% of mapped region) "
            "and kernel working set."
        ),
        hbm_total=hbm_total,
        hbm_wipeable=hbm_wipeable,
        hbm_reserved=hbm_reserved,
        hbm_reserved_reason=(
            "CUDA driver context, GPU page tables, ECC metadata, and "
            "runtime allocator overhead. "
            f"Target fraction: {hbm_fraction:.0%} of free HBM ({hbm_free / (1024**3):.1f} GiB free "
            f"out of {hbm_total / (1024**3):.1f} GiB total). "
            "Overallocating causes Xid faults requiring cold power cycle."
        ) if hbm_total > 0 else "No GPU detected.",
        disk_total=disk_total,
        disk_wipeable=disk_wipeable,
        disk_reserved=disk_reserved,
        disk_reserved_reason=(
            "Filesystem superblock, inode table, journal, reserved blocks, "
            "and NVMe controller wear-leveling metadata. "
            f"Target fraction: {disk_fraction:.0%} of available space "
            f"({disk_avail / (1024**3):.1f} GiB available)."
        ),
    )
