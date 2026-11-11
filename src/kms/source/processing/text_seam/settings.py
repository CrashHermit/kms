from pydantic import BaseModel, Field

from kms.config.inference import TextInferenceSettings


class TextSeamSettings(BaseModel):
    """Language model settings for text seam judging and rewriting."""

    judge: TextInferenceSettings = Field(default_factory=TextInferenceSettings)
    rewriter: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    judge_batch_size: int = Field(default=16, gt=0)
    rewrite_batch_size: int = Field(default=16, gt=0)
    judge_input_token_budget: int = Field(default=4096, gt=0)
    rewrite_input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
