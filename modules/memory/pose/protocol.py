"""Full PoSE-DB protocol orchestrator."""

import time
from dataclasses import dataclass, field

from .noise import generate_noise_bulk
from .verifier import Verifier
from .prover import Prover


@dataclass
class RegionMetrics:
    name: str
    total_bytes: int           # Physical capacity of this region
    reserved_bytes: int        # Bytes we could NOT wipe
    reserved_reason: str       # Why those bytes are unwipeable
    wiped_bytes: int           # Bytes actually overwritten with noise
    fill_time_s: float         # Time to fill this region
    fill_throughput_gbps: float  # GB/s during fill


@dataclass
class ProtocolResult:
    passed: bool
    rounds_passed: int
    rounds_total: int
    bytes_wiped: int            # Total across all regions
    bytes_total: int            # Total physical memory on node
    bytes_reserved: int         # Total unwipeable
    coverage: float             # bytes_wiped / bytes_total
    fill_time_s: float          # Wall time for entire fill phase
    verify_time_s: float        # Wall time for challenge-response phase
    resume_time_s: float        # Time to resume basic operation after wipe
    region_metrics: list[RegionMetrics] = field(default_factory=list)
    seed: bytes = b""


def run_protocol(
    regions: dict,
    region_info: dict | None = None,
    block_size: int = 4096,
    num_rounds: int = 1000,
) -> ProtocolResult:
    """Run the unconditional PoSE-DB protocol.

    Args:
        regions: {"dram": DramRegion, "hbm": HbmRegion, "nvme": NvmeRegion}
        region_info: Optional per-region metadata for the report:
            {"dram": {"total_bytes": ..., "reserved_bytes": ..., "reserved_reason": ...}, ...}
        block_size: Block size in bytes.
        num_rounds: Number of challenge-response rounds.
    """
    prover = Prover(regions=regions, block_size=block_size)
    verifier = Verifier(total_blocks=prover.total_blocks, block_size=block_size)

    # --- Fill phase (timed per-region) ---
    # Verifier streams noise per-region so we can time each independently.
    # The prover never receives the seed -- only opaque blocks.
    t_fill_start = time.monotonic()
    region_metrics = []
    global_offset = 0
    for name, region in regions.items():
        # Generate noise in ~256 MiB bulk chunks — one AES call per chunk,
        # eliminating per-block Python overhead.
        chunk_iter = generate_noise_bulk(
            verifier.seed, start_block=global_offset,
            num_blocks=region.num_blocks, block_size=block_size,
        )
        t0 = time.monotonic()
        prover.fill_region_bulk(name, chunk_iter)
        elapsed = time.monotonic() - t0
        global_offset += region.num_blocks
        wiped = region.num_blocks * block_size
        info = (region_info or {}).get(name, {})
        region_metrics.append(RegionMetrics(
            name=name,
            total_bytes=info.get("total_bytes", wiped),
            reserved_bytes=info.get("reserved_bytes", 0),
            reserved_reason=info.get("reserved_reason", ""),
            wiped_bytes=wiped,
            fill_time_s=elapsed,
            fill_throughput_gbps=wiped / elapsed / (1024**3) if elapsed > 0 else 0,
        ))
    fill_time = time.monotonic() - t_fill_start

    # --- Challenge-response phase ---
    t0 = time.monotonic()
    rounds_passed = 0
    for _ in range(num_rounds):
        idx = verifier.challenge()
        response = prover.respond(idx)
        if verifier.verify(idx, response):
            rounds_passed += 1
    verify_time = time.monotonic() - t0

    # --- Resume operation timing ---
    # Measures time to: release DRAM, re-init CUDA context, sync NVMe
    t0 = time.monotonic()
    for name, region in regions.items():
        region.close()
    # Re-verify GPU is usable
    try:
        from pose.detect import get_cuda_runtime
        get_cuda_runtime().synchronize(0)
    except Exception:
        pass
    resume_time = time.monotonic() - t0

    bytes_wiped = sum(rm.wiped_bytes for rm in region_metrics)
    bytes_total = sum(rm.total_bytes for rm in region_metrics)
    bytes_reserved = sum(rm.reserved_bytes for rm in region_metrics)

    return ProtocolResult(
        passed=(rounds_passed == num_rounds),
        rounds_passed=rounds_passed,
        rounds_total=num_rounds,
        bytes_wiped=bytes_wiped,
        bytes_total=bytes_total,
        bytes_reserved=bytes_reserved,
        coverage=bytes_wiped / bytes_total if bytes_total > 0 else 0,
        fill_time_s=fill_time,
        verify_time_s=verify_time,
        resume_time_s=resume_time,
        region_metrics=region_metrics,
        seed=verifier.seed,
    )
