"""Settings for statement and procedure partitioning."""

from pydantic import BaseModel, Field

from kms.config.inference import ImagePayloadSettings, VisionInferenceSettings


class StatementProcedureSettings(BaseModel):
    """Language model settings for statement and procedure construction."""

    role_typer: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    statement_partitioner: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    procedure_partitioner: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    role_batch_size: int = Field(default=16, gt=0)
    statement_batch_size: int = Field(default=16, gt=0)
    procedure_batch_size: int = Field(default=16, gt=0)
    role_input_token_budget: int = Field(default=4096, gt=0)
    statement_input_token_budget: int = Field(default=4096, gt=0)
    procedure_input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )
