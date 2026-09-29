"""User, deck, and scheduler settings graph models."""

from kms2.core.model.base import Vertex


class User(Vertex):
    """A person who owns sources and decks."""

    name: str


class Deck(Vertex):
    """A user's collection for future learning cards."""

    name: str


class DeckSettings(Vertex):
    """Serialized scheduler configuration for a deck."""

    scheduler_json: str
