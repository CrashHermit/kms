"""Inference settings for goal clarification."""

from pydantic import BaseModel, ConfigDict, Field

from kms.config.inference import TextInferenceSettings


class GoalClarificationSettings(BaseModel):
    """Text inference for interpreting the learner's goal."""

    model_config = ConfigDict(extra='forbid')

    inference: TextInferenceSettings = Field(
        default_factory=lambda: TextInferenceSettings(
            max_tokens=2048, num_retries=0
        )
    )
