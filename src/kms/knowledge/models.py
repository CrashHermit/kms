"""Propositions and arguments in the knowledge graph."""

from kms.core.models import Vertex


class Proposition(Vertex):
    """An identifiable claim, with a short name and complete statement."""

    name: str
    statement: str


class Argument(Vertex):
    """Propositions that jointly support a conclusion.

    Premise and conclusion UUIDs identify propositions without embedding their
    statements. Persist these associations as graph relationships.
    """

    premise_uuids: list[str]
    conclusion_uuid: str
