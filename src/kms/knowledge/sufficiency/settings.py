"""Text inference settings for knowledge sufficiency judgment."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import NoRetryTextInferenceSettings


class SufficiencySettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    inference: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    input_token_budget: int = Field(default=16384, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
