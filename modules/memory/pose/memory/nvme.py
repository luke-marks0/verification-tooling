"""NVMe SSD memory region using direct I/O.

Writes to a preallocated file with O_DIRECT to bypass the OS page cache.
This ensures data is physically written to the SSD, not just cached in DRAM.

O_DIRECT requires:
- Reads/writes aligned to 512-byte (or 4096-byte) boundaries
- Buffer sizes that are multiples of the alignment
Our 4096-byte block size satisfies both.
"""

import mmap
import os


class NvmeRegion:
    def __init__(self, filepath: str, num_blocks: int, block_size: int = 4096):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._size = num_blocks * block_size
        self._filepath = filepath
        self._use_direct = hasattr(os, "O_DIRECT")

        flags = os.O_RDWR | os.O_CREAT
        if self._use_direct:
            flags |= os.O_DIRECT
        self._fd = os.open(filepath, flags, 0o600)
        os.ftruncate(self._fd, self._size)

        # O_DIRECT requires page-aligned buffers. We allocate a reusable
        # aligned buffer via mmap (guaranteed page-aligned by the kernel).
        if self._use_direct:
            self._aligned_buf = mmap.mmap(-1, block_size)

    def write_block(self, index: int, data: bytes) -> None:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        os.lseek(self._fd, index * self.block_size, os.SEEK_SET)
        if self._use_direct:
            self._aligned_buf[:self.block_size] = data
            os.write(self._fd, self._aligned_buf[:self.block_size])
        else:
            os.write(self._fd, data)

    def write_range(self, start_index: int, data: bytes) -> None:
        """Write a contiguous range of blocks in one I/O operation."""
        os.lseek(self._fd, start_index * self.block_size, os.SEEK_SET)
        if self._use_direct:
            # Allocate a temporary page-aligned buffer for the bulk write
            size = len(data)
            aligned = mmap.mmap(-1, size)
            aligned[:] = data
            os.write(self._fd, aligned)
            aligned.close()
        else:
            os.write(self._fd, data)

    def read_block(self, index: int) -> bytes:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        os.lseek(self._fd, index * self.block_size, os.SEEK_SET)
        if self._use_direct:
            # O_DIRECT requires the read buffer to be page-aligned too.
            # Read into the aligned mmap buffer, then copy out.
            n = os.readv(self._fd, [self._aligned_buf])
            return bytes(self._aligned_buf[:n])
        return os.read(self._fd, self.block_size)

    def close(self):
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
