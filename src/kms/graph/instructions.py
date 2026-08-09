from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

INSTRUCTION_LABEL = 'Instruction'


def instruction_uuid(source: str, node_id: int) -> str:
    return uuid5(NAMESPACE_URL, f'{source}#instruction#{node_id}').hex


def instruction_properties(
    instruction: models.Instruction, source: str
) -> dict:
    properties = {
        'uuid': instruction_uuid(source, instruction.node_id),
        'source': nodes.source_uuid(source),
        'text': instruction.text,
        'directive': instruction.directive,
        'index': instruction.node_id,
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def instruction_rows(
    instructions: list[models.Instruction], source: str
) -> list[dict]:
    return [
        instruction_properties(instruction, source)
        for instruction in instructions
    ]


def governs_pairs(
    instructions: list[models.Instruction], source: str
) -> list[dict]:
    return [
        {
            'instruction': instruction_uuid(source, instruction.node_id),
            'node': nodes.node_uuid(source, member_id),
        }
        for instruction in instructions
        for member_id in instruction.members
    ]

