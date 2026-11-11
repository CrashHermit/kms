"""Settings for source processing and shared material reconstruction."""

from pathlib import Path

from pydantic import BaseModel, Field

from kms.config.inference import (
    ImagePayloadSettings,
    NoRetryVisionInferenceSettings,
)
from kms.source.processing.correction.settings import ContentCorrectionSettings
from kms.source.processing.exercise_finder.settings import (
    ExerciseFinderSettings,
)
from kms.source.processing.exercise_splitter.settings import (
    ExerciseSplitterSettings,
)
from kms.source.processing.formatting.settings import FormattingSettings
from kms.source.processing.image_filter.settings import ImageFilterSettings
from kms.source.processing.image_seam.settings import ImageSeamSettings
from kms.source.processing.instruction_finder.settings import (
    InstructionFinderSettings,
)
from kms.source.processing.instruction_governance.settings import (
    InstructionGovernanceSettings,
)
from kms.source.processing.pedagogical_finder.settings import (
    PedagogicalFinderSettings,
)
from kms.source.processing.statement_procedure.settings import (
    StatementProcedureSettings,
)
from kms.source.processing.text_seam.settings import TextSeamSettings


class MaterialReconstructionSettings(BaseModel):
    """Inference and packet-budget settings for full material reconstruction."""

    inference: NoRetryVisionInferenceSettings = Field(
        default_factory=NoRetryVisionInferenceSettings
    )
    batch_size: int = Field(default=16, gt=0)
    input_token_budget: int = Field(default=4096, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)
    image_payload: ImagePayloadSettings = Field(
        default_factory=lambda: ImagePayloadSettings(max_images=5)
    )


class SourceProcessingSettings(BaseModel):
    """Settings for the KMS2 source-processing pipeline."""

    source_directory: Path = Path('output/sources')

    content_correction: ContentCorrectionSettings = Field(
        default_factory=ContentCorrectionSettings
    )
    formatting: FormattingSettings = Field(default_factory=FormattingSettings)
    text_seam: TextSeamSettings = Field(default_factory=TextSeamSettings)
    image_seam: ImageSeamSettings = Field(default_factory=ImageSeamSettings)
    image_filter: ImageFilterSettings = Field(
        default_factory=ImageFilterSettings
    )
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
