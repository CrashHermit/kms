"""Source retrieval and evidence contracts."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from kms.media.models import Material
from kms.source.context import project_source_block, project_source_unit
from kms.source.models.source import Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind


class SourceImageReference(BaseModel):
    """Authorized path and digest for one owned source image block."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    block_uuid: str
    image_path: str
    image_sha256: str


class SourceBlockRead(BaseModel):
    """An owned source block and its containing unit identity."""

    source: Source
    unit_uuid: str
    block: SourceBlock

    @property
    def material(self) -> Material:
        """Project block text and native image without image I/O."""
        return project_source_block(self.block).material


class SourceSearchHit(SourceBlockRead):
    """An owned source block with embedding similarity."""

    similarity: float


class SourceUnitRead(BaseModel):
    """An owned source unit with its complete ordered blocks."""

    source: Source
    unit: SourceUnit

    @property
    def material(self) -> Material:
        """Project ordered unit text and native images without image I/O."""
        return project_source_unit(self.unit)


class SourceUnitMaterial(BaseModel):
    """A unit's identity and complete ordered multimodal material."""

    uuid: str
    kind: SourceUnitKind
    material: Material


class SourceUnitWindow(BaseModel):
    """Owned source material in order around one target unit."""

    source: Source
    before: list[SourceUnitMaterial]
    target: SourceUnitMaterial
    after: list[SourceUnitMaterial]


class SourceUnitSearchHit(SourceUnitRead):
    """An owned source unit with embedding similarity."""

    similarity: float


@dataclass(frozen=True, slots=True)
class SourceImage:
    """Canonical image bytes authorized through an owning source block."""

    block_uuid: str
    data: bytes
