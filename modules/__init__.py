"""Capability modules + the composition pipeline.

This is the curated, function-oriented surface of the deterministic serving
stack. Each subdirectory is one capability (build, inference, network, memory,
attestation, utils) with a documented interface; :class:`Pipeline` composes
them. See ``modules/README.md`` for the capability map.
"""
from modules.pipeline import Pipeline

__all__ = ["Pipeline"]
