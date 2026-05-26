"""PRF-based noise generation using AES-256-CTR.

The noise for the entire memory is conceptually one long AES-CTR stream.
Block index `i` maps to byte offset `i * block_size` in that stream.
This makes generation both seekable (any block independently) and efficient
(sequential blocks share cipher state).
"""

from typing import Iterator

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


_MAX_CTR = 2**128 - 1  # AES-CTR counter space


def generate_block(seed: bytes, index: int, block_size: int = 4096) -> bytes:
    """Generate a single pseudorandom block.

    Args:
        seed: 32-byte AES-256 key (the verifier's secret).
        index: Block index (0-based).
        block_size: Size of each block in bytes. Must be a multiple of 16.
    """
    aes_blocks_per_block = block_size // 16
    counter_start = index * aes_blocks_per_block
    if counter_start + aes_blocks_per_block > _MAX_CTR:
        raise OverflowError(
            f"Block {index} would overflow the 128-bit AES-CTR counter space. "
            f"Reduce block count or block size."
        )
    nonce = counter_start.to_bytes(16, byteorder="big")
    cipher = Cipher(algorithms.AES256(seed), modes.CTR(nonce))
    enc = cipher.encryptor()
    return enc.update(b"\x00" * block_size) + enc.finalize()


def generate_blocks(
    seed: bytes, start: int, count: int, block_size: int = 4096
) -> Iterator[bytes]:
    """Generate a contiguous range of blocks efficiently.

    Uses a single AES-CTR cipher instance for the entire range.
    """
    counter_start = start * (block_size // 16)
    nonce = counter_start.to_bytes(16, byteorder="big")
    cipher = Cipher(algorithms.AES256(seed), modes.CTR(nonce))
    enc = cipher.encryptor()
    for _ in range(count):
        yield enc.update(b"\x00" * block_size)
    enc.finalize()


# Size of bulk noise chunks: 256 MiB. One AES call per chunk.
_CHUNK_BYTES = 256 * 1024 * 1024


def generate_noise_bulk(
    seed: bytes, start_block: int, num_blocks: int, block_size: int = 4096
) -> Iterator[tuple[int, bytes]]:
    """Generate noise in large contiguous chunks for fast fills.

    Yields (start_block_index, chunk_bytes) tuples. Each chunk is generated
    by a single AES-CTR call (~256 MiB), eliminating per-block Python overhead.
    """
    counter_start = start_block * (block_size // 16)
    nonce = counter_start.to_bytes(16, byteorder="big")
    cipher = Cipher(algorithms.AES256(seed), modes.CTR(nonce))
    enc = cipher.encryptor()

    blocks_per_chunk = _CHUNK_BYTES // block_size
    remaining = num_blocks
    chunk_start = 0

    while remaining > 0:
        n = min(blocks_per_chunk, remaining)
        chunk = enc.update(b"\x00" * n * block_size)
        yield (start_block + chunk_start, chunk)
        chunk_start += n
        remaining -= n

    enc.finalize()


def _generate_one_chunk(args):
    """Generate a single chunk in a subprocess. Used by multiprocessing.Pool."""
    seed, chunk_start_block, chunk_num_blocks, block_size = args
    counter_start = chunk_start_block * (block_size // 16)
    nonce = counter_start.to_bytes(16, byteorder="big")
    cipher = Cipher(algorithms.AES256(seed), modes.CTR(nonce))
    enc = cipher.encryptor()
    data = enc.update(b"\x00" * chunk_num_blocks * block_size)
    enc.finalize()
    return (chunk_start_block, data)


def generate_noise_multicore(
    seed: bytes, start_block: int, num_blocks: int,
    block_size: int = 4096, num_workers: int = 8,
) -> Iterator[tuple[int, bytes]]:
    """Generate noise using multiple CPU cores via multiprocessing.

    Splits the block range into 256 MiB chunks and distributes them across
    a process pool. Each process runs its own AES-CTR cipher (separate address
    space, no GIL contention). Results are yielded in order as they complete.

    Uses imap (ordered) so chunks can be written sequentially to memory.
    """
    from multiprocessing import Pool

    blocks_per_chunk = _CHUNK_BYTES // block_size

    # Build work items: one per chunk
    work = []
    remaining = num_blocks
    offset = start_block
    while remaining > 0:
        n = min(blocks_per_chunk, remaining)
        work.append((seed, offset, n, block_size))
        offset += n
        remaining -= n

    # Process pool — imap preserves order and streams results one at a time
    with Pool(processes=num_workers) as pool:
        yield from pool.imap(_generate_one_chunk, work)
