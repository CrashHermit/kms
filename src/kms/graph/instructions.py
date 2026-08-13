from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

INSTRUCTION_LABEL = 'Instruction'


def instruction_uuid(source: str, block: list[int]) -> str:
    return uuid5(
        NAMESPACE_URL, f'{source}#instruction#{nodes.block_key(block)}'
    ).hex


def instruction_properties(
    instruction: models.Instruction, source: str
) -> dict:
    properties = {
        'uuid': instruction_uuid(source, instruction.block),
        'source': nodes.source_uuid(source),
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


def instruction_member_pairs(
    instructions: list[models.Instruction], source: str
) -> list[dict]:
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'instruction': instruction_uuid(source, instruction.block),
        }
        for instruction in instructions
        for node_id in instruction.members
    ]
