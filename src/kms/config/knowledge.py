"""Configuration for proposition extraction, identity, and search."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import TextInferenceSettings, VisionInferenceSettings
from kms.config.source_search import SourceSearchSettings


class KnowledgeExtractionSettings(BaseModel):
    """Inference and payload budget for grounded knowledge extraction."""

    model_config = ConfigDict(extra='forbid')

    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(num_retries=0)
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


class PropositionIdentitySettings(BaseModel):
    """Inference, payload, and retrieval limits for identity preparation."""

    model_config = ConfigDict(extra='forbid')

    inference: TextInferenceSettings = Field(
        default_factory=lambda: TextInferenceSettings(num_retries=0)
    )
    input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    candidate_limit: int = Field(default=32, gt=0)
    result_limit: int = Field(default=8, gt=0)


class KnowledgeSettings(BaseModel):
    """All runtime budgets for persistent knowledge workflows."""

    model_config = ConfigDict(extra='forbid')

    extraction: KnowledgeExtractionSettings = Field(
        default_factory=KnowledgeExtractionSettings
    )
    identity: PropositionIdentitySettings = Field(
        default_factory=PropositionIdentitySettings
    )
    search: SourceSearchSettings = Field(default_factory=SourceSearchSettings)
