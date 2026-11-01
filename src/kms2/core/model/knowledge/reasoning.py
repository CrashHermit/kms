"""One-hop records describing immutable joint-premise arguments."""

from pydantic import BaseModel, ConfigDict

from kms2.core.model.knowledge.proposition import ReasoningStep


class ReasoningStepRecord(BaseModel):
    """A reasoning step with its complete direct graph references."""

    model_config = ConfigDict(extra='forbid')

    step: ReasoningStep
    premise_uuids: list[str]
    conclusion_uuid: str
    supporting_block_uuids: list[str]
