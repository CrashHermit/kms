"""Visual asset models attached to source content."""

from kms2.core.model.base import Vertex


class VisualAsset(Vertex):
    """One visual asset attached to source content."""

    path: str


__all__ = ['VisualAsset']
