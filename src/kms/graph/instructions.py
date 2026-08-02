"""
Graph representation of the instruction layer — ``:Instruction`` hubs over the
exercise nodes a shared lead-in governs.

The instruction finder tags a grouped-exercise lead-in; the distributor asks
which of the following exercises it governs and emits one
``models.Instruction`` per lead-in. This module maps one onto its Neo4j form,
mirroring ``graph.statements`` and ``graph.procedures``: pure mapping, free of
the neo4j driver.

Representation: a hub carrying the lead-in's own text, pointing at each
exercise it governs via ``(:Instruction)-[:GOVERNS]->(:Node)``. The edge runs
from the hub outward, unlike ``:MEMBER_OF``, because it is a claim the
instruction makes about those nodes rather than a grouping they belong to: an
exercise is not "part of" its lead-in the way a block is part of a statement,
and the same exercise keeps its own statement membership independently.

Two text properties, deliberately distinct:

* ``text`` — the lead-in exactly as the page prints it ("In the following
  exercises, simplify."). This is the page's own sentence, and the reason the
  hub exists: the node the sentence came from is dropped from the stream, so
  without the hub it would leave the graph entirely.
* ``directive`` — the model's normalised imperative ("simplify"), useful for
  applying to an exercise and never mistakable for something the page says.

Keeping them apart is the point of the tier. The distributor used to prepend
the normalised form onto every governed node's ``content``, which both
duplicated it and put synthesized text inside the layer defined as the
verbatim page.

Identity: deterministic uuid5 over ``(source, node_id)`` — the lead-in's own
id in the flattened stream, frozen before the node is removed. One lead-in is
one hub, so the id is already unique, and the ``instruction#`` segment keeps
these uuids disjoint from node/statement/procedure/equation/variable uuids.
"""

from uuid import NAMESPACE_URL, uuid5

from kms.core import models
from kms.graph import nodes

INSTRUCTION_LABEL = 'Instruction'


def instruction_uuid(source: str, node_id: int) -> str:
    """Stable, deterministic vertex key for an instruction hub.

    Args:
        source: The stable book identity.
        node_id: The lead-in node's id in the flattened stream.

    Returns:
        The instruction's hex uuid, disjoint from every other tier's.
    """
    return uuid5(NAMESPACE_URL, f'{source}#instruction#{node_id}').hex


def instruction_properties(
    instruction: models.Instruction, source: str
) -> dict:
    """The Neo4j property map for one instruction hub.

    Args:
        instruction: The instruction to map.
        source: The stable book identity.

    Returns:
        The property map, with None values omitted.
    """
    props = {
        'uuid': instruction_uuid(source, instruction.node_id),
        'source': nodes.source_uuid(source),
        'text': instruction.text,
        'directive': instruction.directive,
        'index': instruction.node_id,
    }
    return {key: value for key, value in props.items() if value is not None}


def instruction_rows(
    instructions: list[models.Instruction], source: str
) -> list[dict]:
    """Every instruction's property map, one flat list.

    Args:
        instructions: The instruction overlay.
        source: The stable book identity.

    Returns:
        One property map per instruction.
    """
    return [
        instruction_properties(instruction, source)
        for instruction in instructions
    ]


def governs_pairs(
    instructions: list[models.Instruction], source: str
) -> list[dict]:
    """The ``{instruction, node}`` uuid pairs for the ``:GOVERNS`` edges.

    Args:
        instructions: The instruction overlay.
        source: The stable book identity.

    Returns:
        One pair per governed node, in document order.
    """
    return [
        {
            'instruction': instruction_uuid(source, instruction.node_id),
            'node': nodes.node_uuid(source, member_id),
        }
        for instruction in instructions
        for member_id in instruction.members
    ]
