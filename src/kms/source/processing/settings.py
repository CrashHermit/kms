"""Settings for source processing."""

from pathlib import Path

from pydantic import BaseModel, Field

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
from kms.source.processing.instruction_redistribution.settings import (
    InstructionRedistributionSettings,
)
from kms.source.processing.procedure_finder.settings import (
    ProcedureFinderSettings,
)
from kms.source.processing.procedure_splitter.settings import (
    ProcedureSplitterSettings,
)
from kms.source.processing.text_seam.settings import TextSeamSettings


class SourceProcessingSettings(BaseModel):
    """Settings for the source-processing pipeline."""

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
    procedure_splitter: ProcedureSplitterSettings = Field(
        default_factory=ProcedureSplitterSettings
    )
    exercise_finder: ExerciseFinderSettings = Field(
        default_factory=ExerciseFinderSettings
    )
    procedure_finder: ProcedureFinderSettings = Field(
        default_factory=ProcedureFinderSettings
    )
    instruction_redistribution: InstructionRedistributionSettings = Field(
        default_factory=InstructionRedistributionSettings
    )
