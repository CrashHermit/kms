"""Runtime source registries shared by acquisition and teaching workflows."""

from pydantic import BaseModel, ConfigDict, Field

from kms.core.model.block import SourceBlock
from kms.core.model.source import Source
from kms.core.model.visual_asset import VisualAsset


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class SourceWorkspaceBlockEntry(_StrictModel):
    """Stable workspace position for one block in one source revision."""

    source: Source
    page_indexes: list[int]
    block: SourceBlock


class SourceWorkspaceImageEntry(_StrictModel):
    """Stable workspace position for one image and its source blocks."""

    asset: VisualAsset
    source: Source
    block_positions: list[int] = Field(default_factory=list)
    data: bytes | None = None


class SourceWorkspace(_StrictModel):
    """Runtime block and image registries."""

    block_registry: list[SourceWorkspaceBlockEntry] = Field(
        default_factory=list
    )
    image_registry: list[SourceWorkspaceImageEntry] = Field(
        default_factory=list
    )


class SourceWorkspaceSourceEntry(_StrictModel):
    """Private research catalog summary for one persisted source revision."""

    source: Source
    page_count: int
    block_count: int
    image_count: int


class ResearchWorkspace(SourceWorkspace):
    """Research-run catalog and runtime media registries."""

    acquired_sources: list[SourceWorkspaceSourceEntry] = Field(
        default_factory=list
    )
