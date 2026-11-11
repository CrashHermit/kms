"""User identity graph model."""

from kms.core.models import Vertex


class User(Vertex):
    """A person who owns sources."""

    name: str
