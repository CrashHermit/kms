"""Settings for procedure source-unit discovery."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    NoRetryVisionInferenceSettings,
)


class ProcedureFinderSettings(BaseModel):
    start_router: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    boundary_router: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    start_batch_size: int = Field(default=16, gt=0)
    boundary_batch_size: int = Field(default=16, gt=0)
    start_input_token_budget: int = Field(default=4096, gt=0)
    boundary_input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300, forward_budget=300
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300, forward_budget=300
        )
    )
