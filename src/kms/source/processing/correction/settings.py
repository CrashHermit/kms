from pydantic import BaseModel, Field

from kms.config.inference import VisionInferenceSettings


class ContentCorrectionSettings(BaseModel):
    """Inference and whole-block multimodal batching budgets for OCR correction."""

    inference: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    input_token_budget: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=4, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
