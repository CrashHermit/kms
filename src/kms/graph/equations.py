"""
Graph representation of the equation layer — ``:Equation`` nodes extracted
from content units.

The equation extractor runs as part of the combined equation/variable node
and produces ``models.Equation`` entries (latex, name, domain). This module
maps one onto its Neo4j form.

Representation: every equation carries the ``:Equation`` label. Its identity
includes the KIND of the unit it was extracted from (a statement hub, a
procedure hub, or a plain node), the unit's BLOCK (the block id set, or a
one-node block for a plain node), and its index within that unit's equation
list — the kind namespaces the block, because a statement and a procedure
from the SAME block carry the same block list.

Attachment: an equation hangs off the unit it was extracted from — its
owning ``:Statement``, ``:Procedure`` or plain ``:Node`` — every unit is an
equally valid extraction source, and each one owns its equations via
``:HAS_EQUATION``.

Identity: deterministic uuid5 over ``(source, unit_kind, block, index)``,
disjoint from node/statement/procedure/variable uuids via the ``equation#``
segment.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.procedures import procedure_uuid
from kms.graph.statements import statement_uuid

EQUATION_LABEL = 'Equation'


def equation_uuid(
    source: str, unit_kind: str, block: list[int], index: int
) -> str:
    """Stable, deterministic vertex key for an equation.

    Args:
        source: The stable book identity.
        unit_kind: The unit's kind (``models.UNIT_STATEMENT``,
            ``models.UNIT_PROCEDURE`` or ``models.UNIT_NODE``) — namespaces
            the block, which a statement and a procedure of one block share.
        block: The unit's block — the PCF block id set for a hub, a one-node
            block for a plain node.
        index: The equation's position within the unit's equation list.

    Returns:
        The equation's hex uuid, disjoint from every other uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#equation#{unit_kind}#{nodes.block_key(block)}#{index}',
    ).hex


def equation_properties(
    equation: models.Equation,
    source: str,
    unit_kind: str,
    block: list[int],
    index: int,
) -> dict:
    """The Neo4j property map for one equation."""
    props = {
        'uuid': equation_uuid(source, unit_kind, block, index),
        'source': nodes.source_uuid(source),
        'latex': equation.latex,
        'name': equation.name,
        'domain': equation.domain,
    }
    return {key: value for key, value in props.items() if value is not None}


def equation_rows(
    equations: list[tuple[str, list[int], list[models.Equation]]],
    source: str,
) -> list[dict]:
    """Every equation's property map across the units, one flat list.

    Args:
        equations: The raw channel entries — ``(unit_kind, block,
            [Equation])`` triples.
        source: The stable book identity.

    Returns:
        One property map per equation.
    """
    return [
        equation_properties(equation, source, unit_kind, block, i)
        for unit_kind, block, eq_list in equations
        for i, equation in enumerate(eq_list)
    ]


def unit_container(
    unit_kind: str,
    block: list[int],
    procedure_by_block: dict[tuple[int, ...], models.Procedure],
    source: str,
) -> tuple[str | None, str]:
    """Resolve a unit to its container vertex for the attachment edges.

    Args:
        unit_kind: The unit's kind.
        block: The unit's block.
        procedure_by_block: The procedure overlay keyed by block tuple.
        source: The stable book identity.

    Returns:
        The ``(container_uuid, container_label)`` pair, or ``(None, …)``
        when the unit's procedure cannot be resolved.
    """
    if unit_kind == models.UNIT_STATEMENT:
        return statement_uuid(source, block), 'statement'
    if unit_kind == models.UNIT_PROCEDURE:
        procedure = procedure_by_block.get(tuple(block))
        if procedure is None:
            return None, 'procedure'
        return procedure_uuid(source, block, procedure.index), 'procedure'
    return nodes.node_uuid(source, block[0]), 'node'


def equation_pairs(
    equations: list[tuple[str, list[int], list[models.Equation]]],
    procedures: list[models.Procedure],
    source: str,
) -> list[dict]:
    """The ``{container, container_label, equation}`` uuid pairs for the
    attachment edges.

    An equation hangs off the unit it was extracted from, resolved by the
    unit kind the channel carries: ``:HAS_EQUATION`` from a ``:Statement``,
    ``:Procedure`` or plain ``:Node`` — every unit is an equally valid
    extraction source.

    Args:
        equations: The raw channel entries — ``(unit_kind, block,
            [Equation])`` triples.
        procedures: The procedure overlay, to resolve procedure units.
        source: The stable book identity.

    Returns:
        One ``{container, container_label, equation}`` per equation.
    """
    procedure_by_block = {
        tuple(procedure.block): procedure for procedure in procedures
    }
    pairs: list[dict] = []
    for unit_kind, block, eq_list in equations:
        for i, _equation in enumerate(eq_list):
            container, container_label = unit_container(
                unit_kind, block, procedure_by_block, source
            )
            if container is None:
                continue
            pairs.append(
                {
                    'container': container,
                    'container_label': container_label,
                    'equation': equation_uuid(source, unit_kind, block, i),
                }
            )
    return pairs
