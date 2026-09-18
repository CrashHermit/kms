"""Source-stage configuration models."""

from pydantic import BaseModel, Field

from kms2.config.inference import (
    ContextWindowSettings,
    TextInferenceSettings,
    VisionInferenceSettings,
)


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


class ExerciseSplitterSettings(BaseModel):
    """Language model and context-window settings for exercise splitting."""

    router: TextInferenceSettings = Field(default_factory=TextInferenceSettings)
    splitter: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    context_window: ContextWindowSettings = Field(
        default_factory=ContextWindowSettings
    )


class InstructionFinderSettings(BaseModel):
    """Language model and context-window settings for instruction discovery."""

    start_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    boundary_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300,
            forward_budget=0,
        )
    )


class PedagogicalFinderSettings(BaseModel):
    """Language model and context-window settings for pedagogical discovery."""

    start_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    boundary_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300,
            forward_budget=300,
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300,
            forward_budget=300,
        )
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


class ExerciseFinderSettings(BaseModel):
    """Language model and context-window settings for exercise discovery."""

    start_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    boundary_router: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
        )
    )


class InstructionGovernanceSettings(BaseModel):
    """Language model and context-window settings for instruction governance."""

    inference: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200,
            forward_budget=500,
        )
    )


class TextSeamSettings(BaseModel):
    """Language model settings for text seam judging and rewriting."""

    judge: TextInferenceSettings = Field(default_factory=TextInferenceSettings)
    rewriter: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )


class SourceSettings(BaseModel):
    """Settings for the KMS2 source-processing pipeline."""

    content_correction: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    formatting: TextInferenceSettings = Field(
        default_factory=TextInferenceSettings
    )
    text_seam: TextSeamSettings = Field(default_factory=TextSeamSettings)
    image_seam: VisionInferenceSettings = Field(
        default_factory=VisionInferenceSettings
    )
    image_description: ImageDescriptionSettings = Field(
        default_factory=ImageDescriptionSettings
    )
    exercise_splitter: ExerciseSplitterSettings = Field(
        default_factory=ExerciseSplitterSettings
    )
    instruction_finder: InstructionFinderSettings = Field(
        default_factory=InstructionFinderSettings
    )
    pedagogical_finder: PedagogicalFinderSettings = Field(
        default_factory=PedagogicalFinderSettings
    )
    statement_procedure: StatementProcedureSettings = Field(
        default_factory=StatementProcedureSettings
    )
    exercise_finder: ExerciseFinderSettings = Field(
        default_factory=ExerciseFinderSettings
    )
    instruction_governance: InstructionGovernanceSettings = Field(
        default_factory=InstructionGovernanceSettings
    )
