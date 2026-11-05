"""Request-local readable text and native-image references."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class MultimodalContent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    text: str | None = None
    image_positions: list[Annotated[int, Field(strict=True, ge=0)]] = Field(
        default_factory=list
    )
