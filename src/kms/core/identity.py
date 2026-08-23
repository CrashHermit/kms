"""Canonical deterministic identities for the construction data model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

if TYPE_CHECKING:
    from kms.core import models


def _source(source: str) -> str:
    """Require the raw source key used by every identity formula."""
    if not source or not source.strip():
        raise ValueError('identity generation requires a non-empty source')
    return source


def _block_key(block: list[int]) -> str:
    return '#'.join(str(node_id) for node_id in block)


def _node_provenance_key(node: models.Node) -> str:
    """Build a stable provenance key for a node from its source attributes."""
    doc_idx = node.document_index if node.document_index is not None else 0
    prov_idx = node.index
    return f'{doc_idx}#{prov_idx}'


def source_uuid(source: str) -> str:
    return uuid5(NAMESPACE_URL, _source(source)).hex


def node_uuid(source: str, node: models.Node) -> str:
    """Durable UUID for a node from its source provenance, not list position."""
    return uuid5(
        NAMESPACE_URL, f'{_source(source)}#node#{_node_provenance_key(node)}'
    ).hex


def split_child_uuid(parent_uuid: str, child_index: int) -> str:
    """Returns a deterministic durable UUID for one split child node."""
    if not parent_uuid:
        raise ValueError('split child identity requires a parent uuid')
    if child_index < 0:
        raise ValueError('split child identity requires a non-negative index')
    return uuid5(
        NAMESPACE_URL, f'{parent_uuid}#split-child#{child_index}'
    ).hex


def instruction_uuid(source: str, block: list[int]) -> str:
    return uuid5(
        NAMESPACE_URL, f'{_source(source)}#instruction#{_block_key(block)}'
    ).hex


def statement_uuid(source: str, block: list[int]) -> str:
    return uuid5(
        NAMESPACE_URL, f'{_source(source)}#statement#{_block_key(block)}'
    ).hex


def procedure_uuid(
    source: str,
    block: list[int],
    index: int,
    statement_uuid_value: str | None = None,
) -> str:
    key = f'{_source(source)}#procedure#{_block_key(block)}#{index}'
    if statement_uuid_value is not None:
        key = f'{key}#{statement_uuid_value}'
    return uuid5(NAMESPACE_URL, key).hex


def triplet_uuid(
    source: str,
    node_position: int,
    subject: str,
    predicate: str,
    object: str,
) -> str:
    return uuid5(
        NAMESPACE_URL,
        f'{_source(source)}#triplet#{node_position}#{subject}#{predicate}#{object}',
    ).hex


def entity_uuid(source: str, node_position: int, name: str) -> str:
    return uuid5(
        NAMESPACE_URL, f'{_source(source)}#entity#{node_position}#{name}'
    ).hex


def predicate_uuid(triplet_occurrence_uuid: str) -> str:
    return uuid5(NAMESPACE_URL, f'{triplet_occurrence_uuid}#predicate').hex


def assign_node_uuids(nodes: list[models.Node], source: str) -> None:
    """Assign durable UUIDs to all nodes from their source provenance."""
    _source(source)
    for node in nodes:
        if node.uuid is None:
            node.uuid = node_uuid(source, node)


def assign_instruction_ids(
    instructions: list[models.Instruction], source: str
) -> None:
    """Assign instruction identities after instruction span detection."""
    _source(source)
    for instruction in instructions:
        instruction.uuid = instruction_uuid(source, instruction.block)


def assign_statement_procedure_ids(
    statements: list[models.Statement],
    procedures: list[models.Procedure],
    source: str,
) -> None:
    """Assign all statement/procedure identities at construction boundary."""
    _source(source)
    statements_by_block = {
        tuple(statement.block): statement for statement in statements
    }
    for statement in statements:
        statement.uuid = statement_uuid(source, statement.block)

    procedures_by_block: dict[tuple[int, ...], list[models.Procedure]] = {}
    for procedure in procedures:
        procedures_by_block.setdefault(tuple(procedure.block), []).append(
            procedure
        )
    for block, block_procedures in procedures_by_block.items():
        statement = statements_by_block.get(block)
        statement_id = statement.uuid if statement is not None else None
        for index, procedure in enumerate(block_procedures):
            procedure.index = index
            procedure.statement_uuid = statement_id
            procedure.uuid = procedure_uuid(
                source,
                procedure.block,
                index,
                statement_uuid_value=statement_id,
            )


def assign_triplet_ids(triplets: list[models.Triplet], source: str) -> None:
    """Assign triplet, entity, and predicate occurrence identities."""
    _source(source)
    for triplet in triplets:
        if not triplet.evidence_positions:
            raise ValueError('triplets require at least one evidence node')
        triplet.occurrence_uuids = {}
        triplet.entity_uuids = {}
        triplet.predicate_uuids = {}
        for node_position in triplet.evidence_positions:
            occurrence_id = triplet_uuid(
                source,
                node_position,
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
            triplet.occurrence_uuids[node_position] = occurrence_id
            triplet.entity_uuids[(node_position, triplet.subject)] = (
                entity_uuid(source, node_position, triplet.subject)
            )
            triplet.entity_uuids[(node_position, triplet.object)] = entity_uuid(
                source, node_position, triplet.object
            )
            triplet.predicate_uuids[node_position] = predicate_uuid(
                occurrence_id
            )


def validate_assigned_ids(bundle: models.ConstructionBundle) -> None:
    """Fail fast when durable construction identities are incomplete."""
    errors: list[str] = []
    for index, instruction in enumerate(bundle.instructions):
        expected = instruction_uuid(bundle.source.key or '', instruction.block)
        if instruction.uuid != expected:
            errors.append(f'instruction {index} has invalid uuid')
    for index, statement in enumerate(bundle.statements):
        expected = statement_uuid(bundle.source.key or '', statement.block)
        if statement.uuid != expected:
            errors.append(f'statement {index} has invalid uuid')
    for index, procedure in enumerate(bundle.procedures):
        expected = procedure_uuid(
            bundle.source.key or '',
            procedure.block,
            procedure.index,
            statement_uuid_value=procedure.statement_uuid,
        )
        if procedure.uuid != expected:
            errors.append(f'procedure {index} has invalid uuid')
    for index, triplet in enumerate(bundle.triplets):
        for node_position in triplet.evidence_positions:
            expected = triplet_uuid(
                bundle.source.key or '',
                node_position,
                triplet.subject,
                triplet.predicate,
                triplet.object,
            )
            if triplet.occurrence_uuids.get(node_position) != expected:
                errors.append(
                    f'triplet {index} occurrence {node_position} has invalid uuid'
                )
    if errors:
        raise ValueError('; '.join(errors))
