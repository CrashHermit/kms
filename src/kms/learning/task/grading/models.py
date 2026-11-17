"""Request and state contracts for grading a stored free-response attempt."""

from typing import TypedDict

from pydantic import BaseModel, ConfigDict

from kms.learning.task.models import Task, TaskResponse


class TaskGradingRequest(BaseModel):
    """The authoritative task and recorded response to assess."""

    model_config = ConfigDict(extra='forbid')

    task: Task
    response: TaskResponse


class GradingState(TypedDict):
    """Owner-scoped identity for grading one persisted response."""

    user_uuid: str
    session_uuid: str
    response_uuid: str
