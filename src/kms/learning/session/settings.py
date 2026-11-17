"""Configuration for durable learning-session media."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class LearningSessionSettings(BaseModel):
    """Artifact storage location for durable learner session media."""

    model_config = ConfigDict(extra='forbid')

    artifact_directory: Path = Path('output/learning/artifacts')
