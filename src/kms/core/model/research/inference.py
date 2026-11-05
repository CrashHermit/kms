"""Shared-source research actions and model-facing material."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.source import SourceRevisionReference
from kms.core.model.teaching.positions import Position


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class WebSearchAction(_StrictModel):
    kind: Literal['web_search']
    query: str


class SubmitSourcesAction(_StrictModel):
    kind: Literal['submit_sources']
    result_positions: list[Position]


class FinishResearchAction(_StrictModel):
    """Finish source acquisition without handing selected evidence to teaching."""

    kind: Literal['finish']


ResearchAction = Annotated[
    WebSearchAction | SubmitSourcesAction | FinishResearchAction,
    Field(discriminator='kind'),
]


class ResearchTask(_StrictModel):
    question: str
    context: str


class ResearchSearchResult(_StrictModel):
    position: Position
    title: str | None
    url: str
    snippet: str


class ResearchFailure(_StrictModel):
    operation: str
    url: str | None = None
    message: str


class ResearchToolObservation(_StrictModel):
    kind: Literal[
        'web_search',
        'submit_sources',
        'invalid_action',
    ]
    summary: str
    result_positions: list[Position] = Field(default_factory=list)
    failures: list[ResearchFailure] = Field(default_factory=list)


class ResearchContext(_StrictModel):
    task: ResearchTask
    search_results: list[ResearchSearchResult] = Field(default_factory=list)
    observations: list[ResearchToolObservation] = Field(default_factory=list)
    tools_remaining: int


class ResearchResult(_StrictModel):
    """Final handoff containing only persisted, researcher-selected sources."""

    source_revisions: list[SourceRevisionReference] = Field(
        default_factory=list
    )
    failures: list[ResearchFailure] = Field(default_factory=list)
