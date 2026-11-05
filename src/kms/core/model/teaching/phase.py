"""Teaching interaction phases shared by session and inference models."""

from enum import StrEnum


class TeachingPhase(StrEnum):
    """Current pedagogical phase selected by the teaching workflow."""

    PROBING = 'probing'
    PLANNING = 'planning'
    TEACHING = 'teaching'
