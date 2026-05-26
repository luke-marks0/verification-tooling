"""Host DRAM memory region using mmap.

Allocates anonymous (non-file-backed) memory. On Linux, this comes from
the kernel's virtual memory system and is backed by physical DRAM.

On GH200, the OS sees both LPDDR5X (CPU) and HBM3 (GPU) as system memory
under different NUMA nodes. We pin the allocation to the CPU NUMA node
(node 0) to prevent Linux from spilling into HBM, which CuPy needs.
"""

import ctypes
import ctypes.util
import mmap
import os


def _bind_numa_node0(addr: int, size: int) -> None:
    """Bind a memory range to NUMA node 0 (LPDDR5X on GH200).

    Uses mbind(2) with MPOL_BIND to force pages onto node 0.
    No-op on non-Linux or if libnuma is unavailable.
    """
    if os.uname().sysname != "Linux":
        return
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        # mbind(addr, len, mode=MPOL_BIND, nodemask, maxnode, flags=MPOL_MF_MOVE)
        MPOL_BIND = 2
        MPOL_MF_MOVE = 2
        # nodemask: bit 0 set = node 0
        nodemask = ctypes.c_ulong(1)
        ret = libc.mbind(
            ctypes.c_void_p(addr),
            ctypes.c_ulong(size),
            ctypes.c_int(MPOL_BIND),
            ctypes.byref(nodemask),
            ctypes.c_ulong(64),
            ctypes.c_uint(MPOL_MF_MOVE),
        )
        if ret != 0:
            errno = ctypes.get_errno()
            # Non-fatal: fall back to default allocation
            pass
    except (OSError, AttributeError):
        pass


class DramRegion:
    def __init__(self, size_bytes: int, block_size: int = 4096, numa_node: int = 0):
        self.block_size = block_size
        self.num_blocks = size_bytes // block_size
        self._size = self.num_blocks * block_size
        # MAP_ANONYMOUS | MAP_PRIVATE: not backed by a file
        self._buf = mmap.mmap(-1, self._size)
        # Pin to LPDDR5X NUMA node to avoid spilling into GPU HBM
        buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        _bind_numa_node0(buf_addr, self._size)

    def write_block(self, index: int, data: bytes) -> None:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        self._buf[offset : offset + self.block_size] = data

    def write_range(self, start_index: int, data: bytes) -> None:
        """Write a contiguous range of blocks in one operation."""
        offset = start_index * self.block_size
        self._buf[offset : offset + len(data)] = data

    def read_block(self, index: int) -> bytes:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        return self._buf[offset : offset + self.block_size]

    def close(self):
        self._buf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
