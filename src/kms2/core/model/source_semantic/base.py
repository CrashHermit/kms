"""Shared source semantic occurrence model."""

from kms2.core.model.base import Vertex


class _SemanticOccurrence(Vertex):
    """Common provenance carried by one raw semantic occurrence."""

    source_uuid: str
    source_block_uuid: str
