"""Proposition and argument candidates shared by knowledge creators."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class NewProposition(BaseModel):
    """A new claim identified locally within one creation result."""

    id: str
    name: str
    statement: str


class ExistingPropositionReference(BaseModel):
    """A reference to an existing proposition."""

    kind: Literal['existing'] = 'existing'
    uuid: str


class NewPropositionReference(BaseModel):
    """A reference to a new proposition in the same result."""

    kind: Literal['new'] = 'new'
    id: str


PropositionReference = Annotated[
    ExistingPropositionReference | NewPropositionReference,
    Field(discriminator='kind'),
]


class NewArgument(BaseModel):
    """An inference from premises to a conclusion."""

    premises: list[PropositionReference]
    conclusion: PropositionReference
