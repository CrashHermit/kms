"""Teaching-plan graph models and relationship transports."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from kms2.core.model.base import Vertex


class TeachingPlanStep(Vertex):
    """One immutable step in one teaching-plan revision."""

    title: str
    objective: str


class TeachingPlanDependency(BaseModel):
    """One prerequisite edge between steps in a plan revision."""

    model_config = ConfigDict(extra='forbid')

    step_uuid: str
    prerequisite_uuid: str


class TeachingPlan(Vertex):
    """One immutable teaching-plan revision with its step DAG."""

    summary: str
    created_at: datetime
    steps: list[TeachingPlanStep]
    dependencies: list[TeachingPlanDependency]


class TeachingPlanAcceptance(BaseModel):
    """Relationship data for a learner accepting an existing plan."""

    model_config = ConfigDict(extra='forbid')

    plan_uuid: str
    learner_turn_uuid: str
