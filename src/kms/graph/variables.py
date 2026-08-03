"""
Graph representation of the variable binding layer — ``:Variable`` nodes
extracted from provenance nodes.

The variable extractor runs over every node in the stream and produces
``models.Variable`` entries (symbol, meaning, kind, and the bound value when
the text assigns one). This module maps one onto its Neo4j form, mirroring
``graph.statements`` and ``graph.procedures``: pure mapping, free of the neo4j
driver.

Representation: every variable carries the ``:Variable`` label and hangs
off the ``:Node`` it was extracted from via ``:HAS_VARIABLE``. Statement and
procedure hubs inherit variables through their ``:MEMBER_OF`` edges. (The
``:Equation`` tier was folded into facts — ADR 0001 — so there is no
second container to attach to.)

Identity: deterministic uuid5 over ``(source, node_id, symbol)`` plus the
binding's own text — the bound value and the meaning, empty when absent.
A single block routinely binds one symbol several ways ("$m = 3$" and
"$m = -3$", or "$X$ a set" and "$X$ the set containing 0"); without the
distinguishing text they collapse onto one vertex and the last write wins.
Disjoint from node/statement/procedure uuids via the ``variable#``
segment.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

VARIABLE_LABEL = 'Variable'


def variable_uuid(
    source: str,
    node_id: int,
    symbol: str,
    value: str | None = None,
    meaning: str | None = None,
) -> str:
    """Stable, deterministic vertex key for a variable.

    ``(source, node_id, symbol)`` is not unique within one block: a single
    block routinely binds one symbol several ways — substitutionally
    ("$-m$ when ⓐ $m = 3$ ⓑ $m = -3$") or definitionally ("$X$ is a set",
    "$X$ is the set containing 0", "$X$ is the set containing a and b").
    Without the binding's own text every such pair MERGEs onto the same
    vertex and the last write wins. Both the bound value and the meaning
    are always part of the key (empty when absent), so two bindings differ
    whenever ANY of their parts differ.

    Args:
        source: The stable book identity.
        node_id: The provenance node the variable was extracted from.
        symbol: The variable's symbol.
        value: The value bound to the symbol here, if any.
        meaning: What the symbol stands for here, if any.

    Returns:
        The variable's hex uuid, disjoint from every node/statement/
        procedure uuid.
    """
    key = (
        f'{source}#variable#{node_id}#{symbol}'
        f'#{(value or "").strip()}#{(meaning or "").strip()}'
    )
    return uuid5(NAMESPACE_URL, key).hex


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
        'uuid': variable_uuid(
            source, node_id, variable.symbol, variable.value, variable.meaning
        ),
        'source': nodes.source_uuid(source),
        'symbol': variable.symbol,
        'meaning': variable.meaning,
        'kind': variable.kind,
        'value': variable.value,
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
    """The ``{variable, container}`` uuid pairs for the ``:HAS_VARIABLE``
    edges.

    Every variable hangs off the ``:Node`` it was extracted from — a single
    container kind, since the ``:Equation`` tier is gone. Statement and
    procedure hubs inherit variables through ``:MEMBER_OF``.
    """
    pairs: list[dict] = []
    for node_id, bindings in variables:
        for variable in bindings:
            variable_key = variable_uuid(
                source,
                node_id,
                variable.symbol,
                variable.value,
                variable.meaning,
            )
            pairs.append(
                {
                    'variable': variable_key,
                    'container': nodes.node_uuid(source, node_id),
                }
            )
    return pairs
