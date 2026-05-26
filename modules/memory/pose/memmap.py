"""Maps global block indices to (region, local_index) pairs.

The protocol treats all memory as a flat array of blocks. This module
translates between the flat view and the three physical memory regions.
"""


class MemoryMap:
    def __init__(self, dram_blocks: int, hbm_blocks: int, nvme_blocks: int):
        self.dram_blocks = dram_blocks
        self.hbm_blocks = hbm_blocks
        self.nvme_blocks = nvme_blocks
        self.total_blocks = dram_blocks + hbm_blocks + nvme_blocks

        self._hbm_start = dram_blocks
        self._nvme_start = dram_blocks + hbm_blocks

    def resolve(self, global_index: int) -> tuple[str, int]:
        """Map a global block index to (region_name, local_index)."""
        if global_index < 0 or global_index >= self.total_blocks:
            raise IndexError(
                f"Block {global_index} out of range [0, {self.total_blocks})"
            )
        if global_index < self._hbm_start:
            return ("dram", global_index)
        if global_index < self._nvme_start:
            return ("hbm", global_index - self._hbm_start)
        return ("nvme", global_index - self._nvme_start)
