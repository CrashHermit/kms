"""Operational settings for procedure boundary splitting."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    NoRetryVisionInferenceSettings,
)


class ProcedureSplitterSettings(BaseModel):
    router: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    splitter: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    batch_size: int = Field(default=32, gt=0)
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(target_budget=2048)
    )
    router_input_token_budget: int = Field(default=8192, gt=0)
    splitter_input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
