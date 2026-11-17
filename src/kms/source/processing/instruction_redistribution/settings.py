"""Inference settings for instruction detection and applicability."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    NoRetryVisionInferenceSettings,
)


class InstructionDetectionSettings(BaseModel):
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
            backward_budget=0, forward_budget=0
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300, forward_budget=0
        )
    )


class InstructionApplicabilitySettings(BaseModel):
    inference: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    batch_size: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200, forward_budget=500
        )
    )


class InstructionRedistributionSettings(BaseModel):
    detection: InstructionDetectionSettings = Field(
        default_factory=InstructionDetectionSettings
    )
    applicability: InstructionApplicabilitySettings = Field(
        default_factory=InstructionApplicabilitySettings
    )
