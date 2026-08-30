"""Row and edge builders for the instruction layer."""

from kms.core import identity, models
from kms.graph import nodes

INSTRUCTION_LABEL = 'Instruction'


def instruction_uuid(source: str, block: list[int]) -> str:
    """Returns the deterministic uuid for an instruction block."""
    return identity.instruction_uuid(source, block)


def instruction_properties(
    instruction: models.Instruction, source: str
) -> dict:
    """Builds the property dict used to persist an instruction."""
    if instruction.uuid is None:
        raise ValueError('instruction is missing its assigned uuid')
    if instruction.uuid != instruction_uuid(source, instruction.block):
        raise ValueError('instruction uuid does not match its block')
    properties = {
        'uuid': instruction.uuid,
        'source': nodes.source_uuid(source),
    }
    return {
        key: value for key, value in properties.items() if value is not None
    }


def instruction_rows(
    instructions: list[models.Instruction], source: str
) -> list[dict]:
    """Builds the row dicts for all instructions in a source."""
    return [
        instruction_properties(instruction, source)
        for instruction in instructions
    ]


def instruction_member_pairs(
    instructions: list[models.Instruction],
    node_stream: list[models.SourceNode],
    source: str,
) -> list[dict]:
    """Builds instruction→member node edge pairs."""
    pairs: list[dict] = []
    for instruction in instructions:
        if instruction.uuid is None:
            raise ValueError('instruction is missing its assigned uuid')
        for member_position in instruction.member_positions:
            if not 0 <= member_position < len(node_stream):
                raise ValueError(
                    f'instruction member position {member_position} is outside '
                    f'the node stream'
                )
            node = node_stream[member_position]
            pairs.append(
                {
                    'node': nodes.node_uuid(source, node),
                    'instruction': instruction.uuid,
                }
            )
    return pairs


def instruction_governance_pairs(
    statements: list[models.Statement],
    instructions: list[models.Instruction],
    source: str,
) -> list[dict]:
    """Builds instruction→statement governance edge pairs."""
    instruction_ids = {
        instruction.uuid
        for instruction in instructions
        if instruction.uuid is not None
    }
    pairs: list[dict] = []
    for statement in statements:
        if statement.uuid is None:
            raise ValueError('statement is missing its assigned uuid')
        for instruction_uuid_value in statement.instruction_uuids:
            if instruction_uuid_value not in instruction_ids:
                raise ValueError(
                    f'statement {statement.uuid} references unknown '
                    f'instruction {instruction_uuid_value}'
                )
            pairs.append(
                {
                    'instruction': instruction_uuid_value,
                    'statement': statement.uuid,
                }
            )
    return pairs


