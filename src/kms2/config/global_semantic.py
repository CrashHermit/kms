"""Global semantic-stage configuration models."""

from pydantic import BaseModel, Field

from kms2.config.inference import (
    NoRetryTextInferenceSettings,
    TextInferenceSettings,
)


class GlobalPredicateHubSettings(BaseModel):
    """Similarity, filtering, community, and synthesis settings."""

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


class GlobalEntityHubSettings(BaseModel):
    """Entity global-hub similarity, filtering, community, and synthesis settings."""

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


class GlobalEventHubSettings(BaseModel):
    """Event global-hub similarity, filtering, community, and synthesis settings."""

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


class GlobalStatementHubSettings(BaseModel):
    """Statement global-hub similarity, filtering, community, and synthesis settings."""

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


class GlobalProcedureHubSettings(BaseModel):
    """Procedure global-hub similarity, filtering, community, and synthesis settings."""

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


class GlobalSemanticSettings(BaseModel):
    """Settings for global semantic consolidation."""

    global_entity_hubs: GlobalEntityHubSettings = Field(
        default_factory=GlobalEntityHubSettings
    )
    global_event_hubs: GlobalEventHubSettings = Field(
        default_factory=GlobalEventHubSettings
    )
    global_predicate_hubs: GlobalPredicateHubSettings = Field(
        default_factory=GlobalPredicateHubSettings
    )
    global_statement_hubs: GlobalStatementHubSettings = Field(
        default_factory=GlobalStatementHubSettings
    )
    global_procedure_hubs: GlobalProcedureHubSettings = Field(
        default_factory=GlobalProcedureHubSettings
    )


__all__ = [
    'GlobalEntityHubSettings',
    'GlobalEventHubSettings',
    'GlobalPredicateHubSettings',
    'GlobalProcedureHubSettings',
    'GlobalSemanticSettings',
    'GlobalStatementHubSettings',
]
