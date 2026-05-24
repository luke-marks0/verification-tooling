"""Pydantic wire models for the prover ↔ verifier protocol.

The JSON schemas under schemas/prover_graph.v1.schema.json,
schemas/replay_request.v1.schema.json, schemas/replay_evidence.v1.schema.json,
and schemas/verifier_transcript_entry.v1.schema.json are the wire contracts.
These models are the runtime types the prover and verifier code build on top
of. Both must agree — tests/unit/test_proverdet_wire.py exercises the
intersection.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from modules.core.common.deterministic import canonical_json_text

Sha256Digest = Annotated[
    str, Field(pattern=r"^sha256:[0-9a-f]{64}$", description="sha256:<64-hex>")
]
HexString = Annotated[str, Field(pattern=r"^[0-9a-f]+$")]
IsoDateTime = Annotated[str, Field(min_length=1)]


class _Frozen(BaseModel):
    """Strict-but-frozen base; rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# -- Graph types --


class Task(_Frozen):
    task_id: str = Field(min_length=1)
    pod_id: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    claimed_flops: int = Field(ge=0)


class Artifact(_Frozen):
    artifact_id: str = Field(min_length=1)
    commitment: Sha256Digest
    size_bytes: int = Field(ge=0)


class Transmission(_Frozen):
    transmission_id: str = Field(min_length=1)
    sender_pod_id: str = Field(min_length=1)
    receiver_pod_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    tap_signature: HexString


class Graph(_Frozen):
    graph_version: Literal["v1-placeholder"] = "v1-placeholder"
    run_id: str = Field(min_length=1)
    produced_at: IsoDateTime
    tasks: list[Task]
    artifacts: list[Artifact]
    transmissions: list[Transmission]

    def to_canonical(self) -> str:
        return canonical_json_text(self.model_dump(exclude_none=True))


# -- Replay types --


class TaskTarget(_Frozen):
    kind: Literal["task"] = "task"
    task_id: str = Field(min_length=1)


class ArtifactTarget(_Frozen):
    kind: Literal["artifact"] = "artifact"
    artifact_id: str = Field(min_length=1)


ReplayTarget = Annotated[
    TaskTarget | ArtifactTarget,
    Field(discriminator="kind"),
]


class ErasureSpec(_Frozen):
    challenge_seed: HexString
    deadline_ms: int = Field(ge=1)
    rounds: int = Field(ge=1)


class ProofOfWorkSpec(_Frozen):
    matmul_dim: int = Field(ge=1)
    dtype: Literal["bf16", "fp16", "int8"]
    rounds: int = Field(ge=1)
    report_every_ms: int = Field(ge=1)


class ReplayRequest(_Frozen):
    replay_id: str = Field(min_length=1)
    pod_id: str = Field(min_length=1)
    target: ReplayTarget
    erasure: ErasureSpec
    proof_of_work: ProofOfWorkSpec
    auxiliary: list[str]

    def to_canonical(self) -> str:
        return canonical_json_text(self.model_dump(exclude_none=True))


class ReplayOutput(_Frozen):
    commitment: Sha256Digest
    bytes_b64: str


class ErasureEvidence(_Frozen):
    rounds: int = Field(ge=0)
    passed: int = Field(ge=0)
    log_path: str = Field(min_length=1)


class PowStreamEntry(_Frozen):
    t_ms: int = Field(ge=0)
    freivalds_attestation_id: str = Field(min_length=1)
    matmul_dim: int = Field(ge=1)
    rounds: int = Field(ge=1)
    dtype: Literal["bf16", "fp16", "int8"]


class ReplayEvidence(_Frozen):
    replay_id: str = Field(min_length=1)
    produced_at: IsoDateTime
    output: ReplayOutput
    erasure_evidence: ErasureEvidence
    pow_stream: list[PowStreamEntry]
    errors: list[str] | None = None

    def to_canonical(self) -> str:
        return canonical_json_text(self.model_dump(exclude_none=True))


# -- Transcript --


class TranscriptEntry(_Frozen):
    seq: int = Field(ge=0)
    direction: Literal["sent", "received"]
    endpoint: str = Field(min_length=1)
    timestamp: IsoDateTime
    payload_digest: Sha256Digest
    status_code: int | None = Field(default=None, ge=0)
    payload_path: str | None = Field(default=None, min_length=1)

    def to_canonical(self) -> str:
        return canonical_json_text(self.model_dump(exclude_none=True))


__all__ = [
    "Artifact",
    "ArtifactTarget",
    "ErasureEvidence",
    "ErasureSpec",
    "Graph",
    "HexString",
    "IsoDateTime",
    "PowStreamEntry",
    "ProofOfWorkSpec",
    "ReplayEvidence",
    "ReplayOutput",
    "ReplayRequest",
    "ReplayTarget",
    "Sha256Digest",
    "Task",
    "TaskTarget",
    "TranscriptEntry",
    "Transmission",
]
