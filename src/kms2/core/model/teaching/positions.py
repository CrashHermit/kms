"""Shared non-negative teaching-context position type."""

from typing import Annotated

from pydantic import Field

Position = Annotated[int, Field(ge=0)]
