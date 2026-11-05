"""One-hop records describing immutable joint-premise arguments."""

import json

from pydantic import BaseModel, ConfigDict

from kms.core.model.knowledge.proposition import Argument


def argument_text(
    premises: list[str],
    conclusion: str,
    explanation: str,
) -> str:
    """Serialize the complete argument consistently for search and previews."""
    return json.dumps(
        {
            'premises': sorted(set(premises)),
            'conclusion': conclusion,
            'explanation': explanation,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )


class ArgumentRecord(BaseModel):
    """An argument with its complete direct graph references."""

    model_config = ConfigDict(extra='forbid')

    argument: Argument
    premise_uuids: list[str]
    conclusion_uuid: str
    supporting_block_uuids: list[str]
