"""Semantic-stage configuration models."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    HubSynthesisBudgetSettings,
    ImagePayloadSettings,
    NoRetryTextInferenceSettings,
    TermDescriptionSettings,
    TextInferenceSettings,
)


class _SourceHubSettings(BaseModel):
    """Shared similarity, filtering, and synthesis settings for source hubs."""

    inference: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    synthesis_budget: HubSynthesisBudgetSettings = Field(
        default_factory=HubSynthesisBudgetSettings
    )
    judge: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    candidate_limit: int = 32
    minimum_similarity: float = 0.86
    judge_token_budget: int = 8192
    judge_batch_size: int = 16
    reranker_acceptance_threshold: float = 0.90
    reranker_rejection_threshold: float = 0.60
    max_iterations: int = 100
    min_association_strength: float = 0.2
    minimum_community_size: int = 2


class SourceStatementHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for statements."""


class SourceProcedureHubSettings(_SourceHubSettings):
    """Similarity, filtering, community, and synthesis settings for procedures."""


class SourceEntityHubSettings(BaseModel):
    """Similarity and judgment budgets for source-local entity grouping."""

    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )
    judge: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    candidate_limit: int = 32
    minimum_similarity: float = 0.86
    reranker_rejection_threshold: float = 0.60
    judge_token_budget: int = 8192
    judge_batch_size: int = 16
    safety_margin_tokens: int = 256


class SourceSemanticSettings(BaseModel):
    """Language model settings for source semantic indexing."""

    source_entity_extraction: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    source_entity_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=1024,
            forward_budget=1024,
            target_budget=2048,
        )
    )
    source_entity_image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    source_entity_description: TermDescriptionSettings = Field(
        default_factory=TermDescriptionSettings
    )
    source_entity_hubs: SourceEntityHubSettings = Field(
        default_factory=SourceEntityHubSettings
    )
    source_statement_hubs: SourceStatementHubSettings = Field(
        default_factory=SourceStatementHubSettings
    )
    source_procedure_hubs: SourceProcedureHubSettings = Field(
        default_factory=SourceProcedureHubSettings
    )
