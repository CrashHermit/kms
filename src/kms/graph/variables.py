"""
Graph representation of the variable binding layer — ``:Variable`` nodes
extracted from content units.

The variable extractor runs over every content unit in the stream and produces
``models.Variable`` entries (symbol, meaning, kind). This module maps one onto
its Neo4j form, mirroring ``graph.statements`` and ``graph.procedures``: pure
mapping, free of the neo4j driver.

Representation: every variable carries the ``:Variable`` label and hangs off
the unit it was extracted from via ``:HAS_VARIABLE``: the ``:Equation`` when
the binding lives inside one, the owning ``:Statement`` or ``:Procedure`` hub
or the plain ``:Node`` otherwise. Its identity includes the KIND and BLOCK of
the unit it was extracted from — the kind namespaces the block, because a
statement and a procedure of one block carry the same block list.

Identity: deterministic uuid5 over ``(source, unit_kind, block, symbol)``,
disjoint from node/statement/procedure/equation uuids via the ``variable#``
segment.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.equations import equation_uuid, unit_container

VARIABLE_LABEL = 'Variable'


def variable_uuid(
    source: str, unit_kind: str, block: list[int], symbol: str
) -> str:
    """Stable, deterministic vertex key for a variable.

    Args:
        source: The stable book identity.
        unit_kind: The unit's kind — namespaces the block, which a statement
            and a procedure of one block share.
        block: The unit's block — the PCF block id set for a hub, a one-node
            block for a plain node.
        symbol: The variable's symbol.

    Returns:
        The variable's hex uuid, disjoint from every node/statement/
        procedure uuid.
    """
    return uuid5(
        NAMESPACE_URL,
        f'{source}#variable#{unit_kind}#{nodes.block_key(block)}#{symbol}',
    ).hex


def variable_properties(
    variable: models.Variable, source: str, unit_kind: str, block: list[int]
) -> dict:
    """The Neo4j property map for one variable.

    Args:
        variable: The variable to map.
        source: The stable book identity.
        unit_kind: The kind of unit the variable came from.
        block: The block of the unit the variable came from.

    Returns:
        The property map, with None values omitted.
    """
    props = {
        'uuid': variable_uuid(source, unit_kind, block, variable.symbol),
        'source': nodes.source_uuid(source),
        'symbol': variable.symbol,
        'meaning': variable.meaning,
        'kind': variable.kind,
    }
    return {key: value for key, value in props.items() if value is not None}


def variable_rows(
    variables: list[tuple[str, list[int], list[models.Variable]]],
    source: str,
) -> list[dict]:
    """Every variable's property map across the units, one flat list.

    Args:
        variables: The raw channel entries — ``(unit_kind, block,
            [Variable])`` triples.
        source: The stable book identity.

    Returns:
        One property map per variable.
    """
    return [
        variable_properties(variable, source, unit_kind, block)
        for unit_kind, block, bindings in variables
        for variable in bindings
    ]


def has_variable_pairs(
    variables: list[tuple[str, list[int], list[models.Variable]]],
    procedures: list[models.Procedure],
    source: str,
) -> list[dict]:
    """The ``{variable, container, container_label}`` uuid pairs for the
    ``:HAS_VARIABLE`` edges.

    A variable hangs off the unit it was extracted from: the ``:Equation``
    when ``equation_index`` is set, otherwise the unit's hub or plain node
    resolved by the unit kind the channel carries.
    """
    procedure_by_block = {
        tuple(procedure.block): procedure for procedure in procedures
    }
    pairs: list[dict] = []
    for unit_kind, block, bindings in variables:
        for variable in bindings:
            variable_key = variable_uuid(
                source, unit_kind, block, variable.symbol
            )
            if variable.equation_index is not None:
                container = equation_uuid(
                    source, unit_kind, block, variable.equation_index
                )
                container_label = 'equation'
            else:
                container, container_label = unit_container(
                    unit_kind, block, procedure_by_block, source
                )
                if container is None:
                    continue
            pairs.append(
                {
                    'variable': variable_key,
                    'container': container,
                    'container_label': container_label,
                }
            )
    return pairs
