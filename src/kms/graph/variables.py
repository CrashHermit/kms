"""
Graph representation of the variable binding layer — ``:Variable`` nodes
extracted from provenance nodes.

The variable extractor runs over every node in the stream and produces
``models.Variable`` entries (symbol, meaning, kind). This module maps one onto
its Neo4j form, mirroring ``graph.statements`` and ``graph.procedures``: pure
mapping, free of the neo4j driver.

Representation: every variable carries the ``:Variable`` label and hangs off
the unit it was extracted from via ``:HAS_VARIABLE``: the ``:Equation`` when
the binding lives inside one (``equation_index`` set), or the ``:Node``
otherwise. Statement and procedure hubs inherit variables through their
``:MEMBER_OF`` edges.

Identity: deterministic uuid5 over ``(source, node_id, symbol)``, disjoint
from node/statement/procedure/equation uuids via the ``variable#`` segment.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes
from kms.graph.equations import equation_uuid

VARIABLE_LABEL = 'Variable'


def variable_uuid(source: str, node_id: int, symbol: str) -> str:
    """Stable, deterministic vertex key for a variable.

    Args:
        source: The stable book identity.
        node_id: The provenance node the variable was extracted from.
        symbol: The variable's symbol.

    Returns:
        The variable's hex uuid, disjoint from every node/statement/
        procedure uuid.
    """
    return uuid5(
        NAMESPACE_URL, f'{source}#variable#{node_id}#{symbol}'
    ).hex


def variable_properties(
    variable: models.Variable, source: str, node_id: int
) -> dict:
    """The Neo4j property map for one variable.

    Args:
        variable: The variable to map.
        source: The stable book identity.
        node_id: The provenance node the variable came from.

    Returns:
        The property map, with None values omitted.
    """
    props = {
        'uuid': variable_uuid(source, node_id, variable.symbol),
        'source': nodes.source_uuid(source),
        'symbol': variable.symbol,
        'meaning': variable.meaning,
        'kind': variable.kind,
    }
    return {key: value for key, value in props.items() if value is not None}


def variable_rows(
    variables: list[tuple[int, list[models.Variable]]],
    source: str,
) -> list[dict]:
    """Every variable's property map across the nodes, one flat list.

    Args:
        variables: The raw channel entries — ``(node_id, [Variable])`` pairs.
        source: The stable book identity.

    Returns:
        One property map per variable.
    """
    return [
        variable_properties(variable, source, node_id)
        for node_id, bindings in variables
        for variable in bindings
    ]


def has_variable_pairs(
    variables: list[tuple[int, list[models.Variable]]],
    source: str,
) -> list[dict]:
    """The ``{variable, container, container_label}`` uuid pairs for the
    ``:HAS_VARIABLE`` edges.

    A variable hangs off the ``:Equation`` when ``equation_index`` is set,
    otherwise off the ``:Node`` it was extracted from. Statement and
    procedure hubs inherit variables through ``:MEMBER_OF``.
    """
    pairs: list[dict] = []
    for node_id, bindings in variables:
        for variable in bindings:
            variable_key = variable_uuid(
                source, node_id, variable.symbol
            )
            if variable.equation_index is not None:
                container = equation_uuid(
                    source, node_id, variable.equation_index
                )
                container_label = 'equation'
            else:
                container = nodes.node_uuid(source, node_id)
                container_label = 'node'
            pairs.append(
                {
                    'variable': variable_key,
                    'container': container,
                    'container_label': container_label,
                }
            )
    return pairs
