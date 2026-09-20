"""Explicit dependency composition for KMS2 runtime components."""

from kms2.composition.global_semantic import build_global_semantic_graph
from kms2.composition.source_processing import build_source_processing_graph
from kms2.composition.source_semantic import build_source_semantic_graph

__all__ = [
    'build_global_semantic_graph',
    'build_source_processing_graph',
    'build_source_semantic_graph',
]
