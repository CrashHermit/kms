"""
Graph representation of the equation layer — ``:Equation`` nodes extracted
from provenance nodes.

The equation extractor runs as part of the combined equation/variable node
and produces ``models.Equation`` entries (latex, name, domain). This module
maps one onto its Neo4j form.

Representation: every equation carries the ``:Equation`` label. Its identity
includes the provenance node it was extracted from and its index within that
node's equation list — a node id is already unique, so no unit-kind
namespace is needed.

Attachment: an equation hangs off the ``:Node`` it was extracted from via
``:HAS_EQUATION``. Statement and procedure hubs inherit equations through
their ``:MEMBER_OF`` edges — every equation is reachable from every hub that
covers its node.

Identity: deterministic uuid5 over ``(source, node_id, index)``, disjoint
from node/statement/procedure/variable uuids via the ``equation#`` segment.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

EQUATION_LABEL = 'Equation'


def equation_uuid(source: str, node_id: int, index: int) -> str:
    """Stable, deterministic vertex key for an equation.

    Args:
        source: The stable book identity.
        node_id: The provenance node the equation was extracted from.
        index: The equation's position within the node's equation list.

    Returns:
        The equation's hex uuid, disjoint from every other uuid.
    """
    return uuid5(NAMESPACE_URL, f'{source}#equation#{node_id}#{index}').hex


def equation_properties(
    equation: models.Equation,
    source: str,
    node_id: int,
    index: int,
) -> dict:
    """The Neo4j property map for one equation."""
    props = {
        'uuid': equation_uuid(source, node_id, index),
        'source': nodes.source_uuid(source),
        'latex': equation.latex,
        'name': equation.name,
        'domain': equation.domain,
    }
    return {key: value for key, value in props.items() if value is not None}


def equation_rows(
    equations: list[tuple[int, list[models.Equation]]],
    source: str,
) -> list[dict]:
    """Every equation's property map across the nodes, one flat list.

    Args:
        equations: The raw channel entries — ``(node_id, [Equation])`` pairs.
        source: The stable book identity.

    Returns:
        One property map per equation.
    """
    return [
        equation_properties(equation, source, node_id, i)
        for node_id, eq_list in equations
        for i, equation in enumerate(eq_list)
    ]


def equation_pairs(
    equations: list[tuple[int, list[models.Equation]]],
    source: str,
) -> list[dict]:
    """The ``{node, equation}`` uuid pairs for ``:HAS_EQUATION`` edges.

    An equation hangs off the ``:Node`` it was extracted from. Statement and
    procedure hubs inherit it through ``:MEMBER_OF`` — no separate
    label-bucketed edges to hub tiers.

    Args:
        equations: The raw channel entries — ``(node_id, [Equation])`` pairs.
        source: The stable book identity.

    Returns:
        One ``{node, equation}`` per equation.
    """
    return [
        {
            'node': nodes.node_uuid(source, node_id),
            'equation': equation_uuid(source, node_id, i),
        }
        for node_id, eq_list in equations
        for i, _equation in enumerate(eq_list)
    ]
