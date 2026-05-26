"""Generate the Phase 1 wipe report from a ProtocolResult."""

from datetime import datetime, timezone
from .protocol import ProtocolResult


def generate_report(result: ProtocolResult) -> dict:
    """Convert a ProtocolResult into a JSON-serializable report dict.

    This is the final deliverable for Phase 1. It must contain:
    - Wipe time (total + per-region with throughput)
    - Resume time
    - Memory inventory (total, wiped, reserved per region + reasons)
    - Verification result (rounds passed, coverage)
    """
    return {
        "protocol": "unconditional-pose-db",
        "prf": "AES-256-CTR",
        "block_size_bytes": 4096,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        # --- Key metrics ---
        "wipe_time_s": result.fill_time_s,
        "resume_time_s": result.resume_time_s,

        # --- Memory inventory ---
        "memory_inventory": {
            "total_bytes": result.bytes_total,
            "total_gb": round(result.bytes_total / (1024**3), 2),
            "wiped_bytes": result.bytes_wiped,
            "wiped_gb": round(result.bytes_wiped / (1024**3), 2),
            "reserved_bytes": result.bytes_reserved,
            "reserved_gb": round(result.bytes_reserved / (1024**3), 3),
            "regions": [
                {
                    "name": rm.name,
                    "total_bytes": rm.total_bytes,
                    "total_gb": round(rm.total_bytes / (1024**3), 2),
                    "wiped_bytes": rm.wiped_bytes,
                    "wiped_gb": round(rm.wiped_bytes / (1024**3), 2),
                    "reserved_bytes": rm.reserved_bytes,
                    "reserved_gb": round(rm.reserved_bytes / (1024**3), 3),
                    "reserved_reason": rm.reserved_reason,
                    "fill_time_s": round(rm.fill_time_s, 3),
                    "fill_throughput_gbps": round(rm.fill_throughput_gbps, 2),
                }
                for rm in result.region_metrics
            ],
        },

        # --- Coverage ---
        "coverage_pct": round(result.coverage * 100, 4),

        # --- Verification ---
        "verification": {
            "passed": result.passed,
            "rounds_passed": result.rounds_passed,
            "rounds_total": result.rounds_total,
            "verify_time_s": round(result.verify_time_s, 4),
        },

        # --- Limitations / stubs (must be empty if everything is real) ---
        "limitations": result.limitations if hasattr(result, "limitations") else [],
    }
