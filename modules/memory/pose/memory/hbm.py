"""GPU HBM memory region using raw ctypes to libcudart.so.

No CuPy or numpy dependency. Allocates device memory via cudaMalloc,
transfers data via cudaMemcpy. This avoids CuPy's memory pool overhead
and gives us exact control over GPU memory.
"""

from ..detect import get_cuda_runtime


class HbmRegion:
    def __init__(self, size_bytes: int, block_size: int = 4096, device: int = 0):
        self.block_size = block_size
        self.num_blocks = size_bytes // block_size
        self._size = self.num_blocks * block_size
        self._device = device
        self._rt = get_cuda_runtime()
        self._ptr = self._rt.malloc(device, self._size)

    def write_block(self, index: int, data: bytes) -> None:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        self._rt.memcpy_htod(self._device, self._ptr + offset, data)

    def write_range(self, start_index: int, data: bytes) -> None:
        """Write a contiguous range of blocks in one H2D transfer."""
        offset = start_index * self.block_size
        self._rt.memcpy_htod(self._device, self._ptr + offset, data)

    def read_block(self, index: int) -> bytes:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        return self._rt.memcpy_dtoh(self._device, self._ptr + offset, self.block_size)

    def close(self):
        if hasattr(self, "_ptr") and self._ptr:
            self._rt.free(self._device, self._ptr)
            self._ptr = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
