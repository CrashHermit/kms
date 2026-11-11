from pydantic import BaseModel, Field

from kms.config.inference import TextInferenceSettings


class FormattingSettings(BaseModel):
    """Inference and whole-block batching budgets for source formatting."""

    inference: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    input_token_budget: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=32, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
