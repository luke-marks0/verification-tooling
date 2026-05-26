"""Generate noise via /dev/urandom and stream from file.

For the devurandom method:
1. pregen_urandom() writes noise to a file using parallel dd processes
2. stream_from_file() yields chunks from the file for filling regions
3. verify_from_file() reads a specific block for challenge verification
"""

import os
import subprocess

_CHUNK = 256 * 1024 * 1024  # 256 MiB


def pregen_urandom(path: str, total_bytes: int, num_cores: int = 16) -> None:
    """Pre-generate noise file by reading /dev/urandom in parallel.

    Each core writes to a non-overlapping region of the output file
    using dd with seek offsets. No serial concatenation.
    """
    # Pre-create file
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.ftruncate(fd, total_bytes)
    os.close(fd)

    if total_bytes < _CHUNK:
        # Small file: single dd with exact size as bs
        cmd = (f"dd if=/dev/urandom bs={total_bytes} count=1 "
               f"of={path} conv=notrunc 2>/dev/null")
        subprocess.run(cmd, shell=True, check=True)
    else:
        per_core = (total_bytes // num_cores // _CHUNK) * _CHUNK
        if per_core == 0:
            per_core = _CHUNK
            num_cores = 1

        procs = []
        for i in range(num_cores):
            seek_blocks = i * (per_core // _CHUNK)
            count = per_core // _CHUNK
            if count == 0:
                continue
            cmd = (f"dd if=/dev/urandom bs={_CHUNK} count={count} "
                   f"of={path} seek={seek_blocks} conv=notrunc 2>/dev/null")
            procs.append(subprocess.Popen(cmd, shell=True))

        for p in procs:
            p.wait()

    # Truncate to exact size (parallel dd may overshoot due to rounding)
    os.truncate(path, total_bytes)


def stream_from_file(
    path: str, offset: int, size: int, chunk_size: int = _CHUNK
):
    """Yield chunks from the noise file for filling a region.

    Generator — yields one chunk at a time to avoid loading everything
    into memory at once. Uses O_DIRECT where available to bypass page cache.
    """
    import mmap as _mmap

    flags = os.O_RDONLY
    use_direct = hasattr(os, "O_DIRECT")
    if use_direct:
        flags |= os.O_DIRECT

    fd = os.open(path, flags)
    os.lseek(fd, offset, os.SEEK_SET)

    if use_direct:
        # O_DIRECT needs page-aligned buffer for reads
        aligned_buf = _mmap.mmap(-1, chunk_size)

    remaining = size
    while remaining > 0:
        n = min(chunk_size, remaining)
        if use_direct:
            nbytes = os.readv(fd, [aligned_buf])
            data = bytes(aligned_buf[:min(nbytes, n)])
        else:
            data = os.read(fd, n)
        if not data:
            break
        yield data
        remaining -= len(data)

    os.close(fd)
    if use_direct:
        aligned_buf.close()


def verify_from_file(path: str, block_index: int, block_size: int = 4096) -> bytes:
    """Read a single block from the noise file for challenge verification."""
    fd = os.open(path, os.O_RDONLY)
    os.lseek(fd, block_index * block_size, os.SEEK_SET)
    data = os.read(fd, block_size)
    os.close(fd)
    return data
