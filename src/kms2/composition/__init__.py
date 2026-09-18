"""Explicit dependency composition for KMS2 runtime components."""

from kms2.composition.semantic import build_semantic_graph
from kms2.composition.source import build_source_graph

__all__ = ['build_semantic_graph', 'build_source_graph']
