# network — deterministic network egress

**Purpose.** Turn application-layer bytes into bitwise-reproducible L2 frames, so
two independent servers emit byte-identical packets on the wire. The value isn't
the ~2–3k LOC — it's knowing *which* mitigations to apply at *which* layer (no
timestamps, no SACK, deterministic ISN/IP-ID, MRF-compliant framing).

**Interface.**

```python
from modules.network import egress_frames, create_net_stack

# Headline: payload + config -> deterministic frames.
frames = egress_frames(payload, manifest=manifest, lockfile=lockfile,
                       dst_mac="02:00:00:00:00:02", backend="sim")

# Lower-level: a reusable stack across many responses.
stack = create_net_stack(manifest, lockfile, backend="sim")
frames = stack.process_response(conn_index=0, response_bytes=payload)
digest = stack.capture_digest()   # SHA-256 over all emitted frames
```

**Artifacts.** Consumes `manifest.v1` (+ `lockfile.v1`); produces raw L2 frames
and a capture digest (recorded in `run_bundle.v1.observables.network_egress`).

**Backends / requirements.**
- `sim` — pure Python, no hardware. Used in CI and tests. ✅ mature
- `dpdk` — real NIC via `native/libnetdet`; needs DPDK 24.11 + a supported NIC. ⚠️ rough

**Example.** See `workflows/deterministic_inference_server.py` (proves the same
payload yields identical frames across two calls).

**Underlying code.** `pkg/networkdet/` (frame/tcp/ip/ethernet/capture/warden),
`native/libnetdet/` (DPDK C lib). This module re-exports `pkg.networkdet`.
