"""Typed request and results for proposition identity auditing."""

from pydantic import BaseModel, ConfigDict

from kms.knowledge.candidates import NewProposition
from kms.knowledge.models import Proposition


class IdentityRequest(BaseModel):
    """One proposed claim and the only propositions it may reuse."""

    model_config = ConfigDict(extra='forbid')

    candidate: NewProposition
    matches: list[Proposition]


class IdentityDecision(BaseModel):
    """Select an equivalent supplied proposition, or create a new one."""

    model_config = ConfigDict(extra='forbid')

    match_uuid: str | None


class AuditResult(BaseModel):
    """Canonical propositions for local ids and newly minted vertices."""

    model_config = ConfigDict(extra='forbid')

    canonical: dict[str, Proposition]
    new_propositions: list[Proposition]
