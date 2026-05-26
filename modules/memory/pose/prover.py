"""Prover: stores noise blocks streamed from the verifier, responds to challenges.

The prover NEVER has access to the PRF seed. It receives opaque blocks from
the verifier's noise_stream() and stores them. It can only respond to
challenges by reading from its stored memory.
"""

from typing import Iterator


class Prover:
    def __init__(self, regions: dict, block_size: int = 4096):
        """
        Args:
            regions: Dict mapping region name -> region object.
                     Each region must have .num_blocks, .write_block(), .read_block().
                     Insertion order determines the memory map layout.
            block_size: Block size in bytes.
        """
        self.block_size = block_size
        self._regions = regions
        self._region_list = list(regions.items())

        self._offsets = {}  # region_name -> global_start_index
        offset = 0
        for name, r in self._region_list:
            self._offsets[name] = offset
            offset += r.num_blocks
        self.total_blocks = offset

    def fill(self, block_stream: Iterator[bytes]) -> None:
        """Fill all memory regions from an opaque block stream.

        The stream is produced by the verifier. The prover does not know
        the seed or how blocks are generated -- it just stores them.
        """
        global_idx = 0
        for name, region in self._region_list:
            for _ in range(region.num_blocks):
                block = next(block_stream)
                region.write_block(global_idx - self._offsets[name], block)
                global_idx += 1

    def fill_region(self, name: str, block_stream: Iterator[bytes]) -> None:
        """Fill a single named region from the stream.

        Batches blocks into chunks for efficient bulk writes.
        """
        region = self._regions[name]
        # 4096 blocks * 4096 bytes = 16 MB per batch
        batch_blocks = 4096
        remaining = region.num_blocks
        start_idx = 0

        while remaining > 0:
            n = min(batch_blocks, remaining)
            batch = bytearray()
            for _ in range(n):
                batch.extend(next(block_stream))
            if hasattr(region, "write_range"):
                region.write_range(start_idx, bytes(batch))
            else:
                for i in range(n):
                    off = i * self.block_size
                    region.write_block(start_idx + i, batch[off:off + self.block_size])
            start_idx += n
            remaining -= n

    def fill_region_bulk(self, name: str, chunk_iter) -> None:
        """Fill a region from pre-generated bulk chunks.

        Accepts an iterator of (start_block_index, chunk_bytes) tuples,
        as produced by noise.generate_noise_bulk(). Each chunk is ~256 MiB,
        written in a single call — no per-block Python loop.
        """
        region = self._regions[name]
        for start_idx, chunk in chunk_iter:
            local_start = start_idx - self._offsets.get(name, 0)
            if hasattr(region, "write_range"):
                region.write_range(local_start, chunk)
            else:
                # Fallback: slice into blocks
                for i in range(len(chunk) // self.block_size):
                    off = i * self.block_size
                    region.write_block(local_start + i, chunk[off:off + self.block_size])

    def respond(self, global_index: int) -> bytes:
        """Return the block at the given global index (memory read only)."""
        for name, region in self._region_list:
            start = self._offsets[name]
            if start <= global_index < start + region.num_blocks:
                return region.read_block(global_index - start)
        raise IndexError(f"Block {global_index} out of range [0, {self.total_blocks})")
