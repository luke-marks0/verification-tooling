"""Verifier: streams noise to prover, issues challenges, checks responses.

The verifier holds the PRF seed. It never exposes the seed to the prover.
The prover only receives opaque noise blocks via noise_stream().
"""

import os
import secrets
from typing import Iterator

from .noise import generate_block, generate_blocks


class Verifier:
    def __init__(self, total_blocks: int, block_size: int = 4096, seed: bytes | None = None):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.seed = seed or os.urandom(32)

    def noise_stream(self) -> Iterator[bytes]:
        """Generate the noise block stream for the prover to store.

        The prover calls this to receive blocks. It never sees the seed.
        """
        yield from generate_blocks(
            self.seed, start=0, count=self.total_blocks,
            block_size=self.block_size,
        )

    def challenge(self) -> int:
        """Pick a random block index to challenge."""
        return secrets.randbelow(self.total_blocks)

    def verify(self, index: int, response: bytes) -> bool:
        """Check the prover's response by regenerating from the secret seed."""
        expected = generate_block(self.seed, index, self.block_size)
        return response == expected
