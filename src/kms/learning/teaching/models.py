"""Durable learner-facing teaching material."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from kms.core.models import Vertex
from kms.learning.materials import Material
from kms.learning.references import GroundingReferences, LearningTarget


class Teaching(Vertex):
    """One persisted explanation grounded in an existing learning target."""

    kind: Literal['teaching'] = 'teaching'
    understanding: str
    objective: str
    target: LearningTarget
    material: Material
    grounding: GroundingReferences
    assessment_uuids: list[str]


class PresentedTeaching(BaseModel):
    """Learner-visible material from a successfully persisted teaching step."""

    model_config = ConfigDict(extra='forbid')

    teaching_uuid: str
    material: Material
