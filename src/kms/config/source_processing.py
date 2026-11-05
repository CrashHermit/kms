"""Settings for source capture and processing stages."""

from pathlib import Path

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    TextInferenceSettings,
    VisionInferenceSettings,
)


class ContentCorrectionSettings(BaseModel):
    """Inference and whole-block multimodal batching budgets for OCR correction."""

    inference: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    input_token_budget: int = Field(default=8192, gt=0)
    batch_size: int = Field(default=4, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


class FormattingSettings(BaseModel):
    """Inference and whole-block batching budgets for source formatting."""

    inference: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    input_token_budget: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=32, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


class ImageDescriptionSettings(BaseModel):
    """Language model and context-window settings for image descriptions."""

    inference: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200,
            forward_budget=200,
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    batch_size: int = Field(default=4, gt=0)
    input_token_budget: int = Field(default=2048, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


class ImageFilterSettings(BaseModel):
    """Vision inference and context settings for image retention decisions."""

    inference: VisionInferenceSettings = Field(
        default_factory=lambda: VisionInferenceSettings(max_tokens=1024)
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=1024, forward_budget=1024
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    batch_size: int = Field(default=4, gt=0)
    input_token_budget: int = Field(default=8192, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


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


class ImageSeamSettings(BaseModel):
    """Vision inference and text-metadata batching for image seams."""

    inference: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    batch_size: int = Field(default=4, gt=0)
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
    input_token_budget: int = Field(default=2048, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)


class SourceProcessingSettings(BaseModel):
    """Settings for the KMS2 source-processing pipeline."""

    source_directory: Path = Path('output/sources')

    content_correction: ContentCorrectionSettings = Field(
        default_factory=ContentCorrectionSettings
    )
    formatting: FormattingSettings = Field(default_factory=FormattingSettings)
    text_seam: TextSeamSettings = Field(default_factory=TextSeamSettings)
    image_seam: ImageSeamSettings = Field(default_factory=ImageSeamSettings)
    image_description: ImageDescriptionSettings = Field(
        default_factory=ImageDescriptionSettings
    )
    image_filter: ImageFilterSettings = Field(
        default_factory=ImageFilterSettings
    )
