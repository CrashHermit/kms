"""Learner-session conversation contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class SessionMessage(BaseModel):
    """One message in the learner's session conversation."""

    model_config = ConfigDict(extra='forbid')

    role: Literal['assistant', 'user']
    content: str
