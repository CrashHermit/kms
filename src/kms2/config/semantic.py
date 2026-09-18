"""Semantic-stage configuration models."""

from pydantic import BaseModel, Field

from kms2.config.inference import (
    ContextWindowSettings,
    NoRetryTextInferenceSettings,
    TextInferenceSettings,
)


class TermDescriptionSettings(BaseModel):
    """Language model and context-window settings for typed descriptions."""

    inference: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )


class _SourceHubSettings(BaseModel):
    """Shared similarity, filtering, and synthesis settings for source hubs."""

    inference: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    judge: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    candidate_limit: int = 32
    minimum_similarity: float = 0.86
    reranker_token_budget: int = 4096
    judge_token_budget: int = 8192
    judge_batch_size: int = 16
    reranker_acceptance_threshold: float = 0.90
    reranker_rejection_threshold: float = 0.60
    max_iterations: int = 100
    min_association_strength: float = 0.2
    minimum_community_size: int = 2


class SourceEntityHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for entities."""


class SourceEventHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for events."""


class SourcePredicateHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for predicates."""


class SourceStatementHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for statements."""


class SourceProcedureHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for procedures."""


class SourceTripletHubSettings(BaseModel):
    """Inference settings for exact source-local triplet hubs."""

    inference: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )


class SemanticSettings(BaseModel):
    """Language model and context-window settings for semantic extraction."""

    fact_extraction: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    triplet_decomposition: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )
    source_entity_description: TermDescriptionSettings = Field(
        default_factory=TermDescriptionSettings
    )
    source_event_description: TermDescriptionSettings = Field(
        default_factory=TermDescriptionSettings
    )
    source_predicate_description: TermDescriptionSettings = Field(
        default_factory=TermDescriptionSettings
    )
    source_statement_description: TermDescriptionSettings = Field(
        default_factory=TermDescriptionSettings
    )
    source_procedure_description: TermDescriptionSettings = Field(
        default_factory=TermDescriptionSettings
    )
    source_entity_hubs: SourceEntityHubSettings = Field(
        default_factory=SourceEntityHubSettings
    )
    source_event_hubs: SourceEventHubSettings = Field(
        default_factory=SourceEventHubSettings
    )
    source_predicate_hubs: SourcePredicateHubSettings = Field(
        default_factory=SourcePredicateHubSettings
    )
    source_triplet_hubs: SourceTripletHubSettings = Field(
        default_factory=SourceTripletHubSettings
    )
    source_statement_hubs: SourceStatementHubSettings = Field(
        default_factory=SourceStatementHubSettings
    )
    source_procedure_hubs: SourceProcedureHubSettings = Field(
        default_factory=SourceProcedureHubSettings
    )
