"""Source-pedagogical stage configuration models."""

from pydantic import BaseModel, Field

from kms.config.inference import (
    ContextWindowSettings,
    ImagePayloadSettings,
    NoRetryTextInferenceSettings,
    TextInferenceSettings,
)


class ExerciseSplitterSettings(BaseModel):
    """Language model and context-window settings for exercise splitting."""

    router: TextInferenceSettings = Field(default_factory=TextInferenceSettings)
    splitter: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    batch_size: int = Field(default=32, gt=0)
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(target_budget=2048)
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )


class InstructionFinderSettings(BaseModel):
    """Language model and context-window settings for instruction discovery."""

    start_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    boundary_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
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
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )


class ExerciseFinderSettings(BaseModel):
    """Language model and context-window settings for exercise discovery."""

    start_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    boundary_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
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
            backward_budget=0, forward_budget=0
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )


class PedagogicalFinderSettings(BaseModel):
    """Language model and context-window settings for pedagogical discovery."""

    start_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    boundary_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
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
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )


class StatementProcedureSettings(BaseModel):
    """Language model settings for statement and procedure construction."""

    role_typer: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    statement_partitioner: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    procedure_partitioner: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
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


class MaterialReconstructionSettings(BaseModel):
    """Inference and packet-budget settings for full material reconstruction."""

    inference: NoRetryTextInferenceSettings = Field(
        default_factory=NoRetryTextInferenceSettings
    )
    batch_size: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )


class InstructionGovernanceSettings(BaseModel):
    """Language model and context-window settings for instruction governance."""

    inference: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    batch_size: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200, forward_budget=500
        )
    )
    image_payload: ImagePayloadSettings = Field(
        default_factory=ImagePayloadSettings
    )


class SourcePedagogicalSettings(BaseModel):
    """Settings for source-pedagogical preparation."""

    exercise_splitter: ExerciseSplitterSettings = Field(
        default_factory=ExerciseSplitterSettings
    )
    instruction_finder: InstructionFinderSettings = Field(
        default_factory=InstructionFinderSettings
    )
    exercise_finder: ExerciseFinderSettings = Field(
        default_factory=ExerciseFinderSettings
    )
    pedagogical_finder: PedagogicalFinderSettings = Field(
        default_factory=PedagogicalFinderSettings
    )
    statement_procedure: StatementProcedureSettings = Field(
        default_factory=StatementProcedureSettings
    )
    instruction_governance: InstructionGovernanceSettings = Field(
        default_factory=InstructionGovernanceSettings
    )
    source_statement_reconstruction: MaterialReconstructionSettings = Field(
        default_factory=MaterialReconstructionSettings
    )
    source_procedure_reconstruction: MaterialReconstructionSettings = Field(
        default_factory=MaterialReconstructionSettings
    )
