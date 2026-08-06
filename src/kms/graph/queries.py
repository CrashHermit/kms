"""
Every Cypher statement in the graph tier — the query module.

``writer`` is the write half of the I/O layer: it composes rows and edge
pairs from the mapping modules and hands them to the batched MERGE
queries defined here. This module is the query half: every Cypher string
lives in one place, named and parameterised, so the writer embeds no
Cypher and the read side has a single home.

Write queries are the batched, idempotent upserts (``MERGE ... ON CREATE
SET created_at = $now SET ..., modified_at = $now``) that ``writer``
runs. Read queries are named lookbacks that return plain data, used by
construction passes that must see what is already in the graph.

Every function takes the injected ``session_factory`` (the same callable
the persister receives — an async context manager with
``run(query, **params)``) and returns plain values, never driver objects,
so the queries are unit-testable against a scripted session and the neo4j
driver stays quarantined in ``db``. Scoping by book uses the raw source
identity and maps it through ``nodes.source_uuid``, matching ``writer``.
"""

from collections.abc import Callable

from kms.graph.definitions import DEFINITION_LABEL
from kms.graph.entities import ENTITY_LABEL
from kms.graph.entity_hubs import ENTITY_HUB_LABEL
from kms.graph.facts import FACT_LABEL
from kms.graph.instructions import INSTRUCTION_LABEL
from kms.graph.nodes import NODE_LABEL, SOURCE_LABEL, source_uuid
from kms.graph.predicate_hubs import PREDICATE_HUB_LABEL
from kms.graph.predicates import PREDICATE_LABEL
from kms.graph.procedures import ACT_LABEL, PROCEDURE_LABEL
from kms.graph.statements import STATEMENT_LABEL
from kms.graph.triplets import TRIPLET_LABEL

# ============================================================================
# Write queries — batched, idempotent upserts run by ``writer``
# ============================================================================


MERGE_SOURCE = (
    f'MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += $props, s.modified_at = $now'
)


def merge_nodes_query(label: str | None) -> str:
    """The batched ``:Node`` upsert for one per-type label group.

    Args:
        label: The per-type label (e.g. ``'Paragraph'``), or None to keep
            the plain ``:Node`` label.

    Returns:
        The Cypher for one batched MERGE of ``$rows``.
    """
    query = (
        f'UNWIND $rows AS row '
        f'MERGE (n:{NODE_LABEL} {{uuid: row.uuid}}) '
        f'ON CREATE SET n.created_at = $now '
        f'SET n += row, n.modified_at = $now'
    )
    if label:
        query += f' SET n:{label}'
    return query


MERGE_HEAD = (
    f'MATCH (s:{SOURCE_LABEL} {{uuid: $source}}), '
    f'(n:{NODE_LABEL} {{uuid: $head}}) '
    f'MERGE (s)-[r:HEAD]->(n) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_NEXT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (a:{NODE_LABEL} {{uuid: pair.from}}), '
    f'(b:{NODE_LABEL} {{uuid: pair.to}}) '
    f'MERGE (a)-[r:NEXT]->(b) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_STATEMENTS = (
    f'UNWIND $rows AS row '
    f'MERGE (s:{STATEMENT_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += row, s.modified_at = $now'
)

MERGE_STATEMENT_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
    f'(s:{STATEMENT_LABEL} {{uuid: pair.statement}}) '
    f'MERGE (n)-[r:MEMBER_OF]->(s) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_INSTRUCTIONS = (
    f'UNWIND $rows AS row '
    f'MERGE (i:{INSTRUCTION_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET i.created_at = $now '
    f'SET i += row, i.modified_at = $now'
)

MERGE_GOVERNS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (i:{INSTRUCTION_LABEL} {{uuid: pair.instruction}}), '
    f'(n:{NODE_LABEL} {{uuid: pair.node}}) '
    f'MERGE (i)-[r:GOVERNS]->(n) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_PROCEDURES = (
    f'UNWIND $rows AS row '
    f'MERGE (p:{PROCEDURE_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET p.created_at = $now '
    f'SET p += row, p.modified_at = $now'
)

MERGE_ACTS = (
    f'UNWIND $rows AS row '
    f'MERGE (a:{ACT_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET a.created_at = $now '
    f'SET a += row, a.modified_at = $now'
)

MERGE_PROCEDURE_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
    f'(p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
    f'MERGE (n)-[r:MEMBER_OF]->(p) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_FIRST = (
    f'UNWIND $pairs AS pair '
    f'MATCH (p:{PROCEDURE_LABEL} {{uuid: pair.procedure}}), '
    f'(a:{ACT_LABEL} {{uuid: pair.act}}) '
    f'MERGE (p)-[r:FIRST]->(a) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_THEN = (
    f'UNWIND $pairs AS pair '
    f'MATCH (a:{ACT_LABEL} {{uuid: pair.from}}), '
    f'(b:{ACT_LABEL} {{uuid: pair.to}}) '
    f'MERGE (a)-[r:THEN]->(b) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_FACTS = (
    f'UNWIND $rows AS row '
    f'MERGE (f:{FACT_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET f.created_at = $now '
    f'SET f += row, f.modified_at = $now'
)

MERGE_EVIDENCE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{NODE_LABEL} {{uuid: pair.node}}), '
    f'(f:{FACT_LABEL} {{uuid: pair.fact}}) '
    f'MERGE (n)-[r:EVIDENCE_FOR]->(f) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_ENTITIES = (
    f'UNWIND $rows AS row '
    f'MERGE (e:{ENTITY_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET e.created_at = $now '
    f'SET e += row, e.modified_at = $now'
)

MERGE_TRIPLETS = (
    f'UNWIND $rows AS row '
    f'MERGE (t:{TRIPLET_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET t.created_at = $now '
    f'SET t += row, t.modified_at = $now'
)

MERGE_YIELDS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (f:{FACT_LABEL} {{uuid: pair.fact}}), '
    f'(t:{TRIPLET_LABEL} {{uuid: pair.triplet}}) '
    f'MERGE (f)-[r:YIELDS]->(t) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_PREDICATES = (
    f'UNWIND $rows AS row '
    f'MERGE (p:{PREDICATE_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET p.created_at = $now '
    f'SET p += row, p.modified_at = $now'
)

MERGE_HAS_PREDICATE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(p:{PREDICATE_LABEL} {{uuid: pair.predicate}}) '
    f'MERGE (t)-[r:HAS_PREDICATE]->(p) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_HAS_SUBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(e:{ENTITY_LABEL} {{uuid: pair.entity}}) '
    f'MERGE (t)-[r:HAS_SUBJECT]->(e) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_HAS_OBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(e:{ENTITY_LABEL} {{uuid: pair.entity}}) '
    f'MERGE (t)-[r:HAS_OBJECT]->(e) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)


MERGE_ENTITY_HUBS = (
    f'UNWIND $rows AS row '
    f'MERGE (h:{ENTITY_HUB_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET h.created_at = $now '
    f'SET h += row, h.modified_at = $now'
)

MERGE_PREDICATE_HUBS = (
    f'UNWIND $rows AS row '
    f'MERGE (h:{PREDICATE_HUB_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET h.created_at = $now '
    f'SET h += row, h.modified_at = $now'
)

MERGE_DEFINITIONS = (
    f'UNWIND $rows AS row '
    f'MERGE (d:{DEFINITION_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET d.created_at = $now '
    f'SET d += row, d.modified_at = $now'
)

MERGE_CANONICAL_ENTITY = (
    f'UNWIND $pairs AS pair '
    f'MATCH (e:{ENTITY_LABEL} {{uuid: pair.entity}}), '
    f'(h:{ENTITY_HUB_LABEL} {{uuid: pair.hub}}) '
    f'MERGE (e)-[r:CANONICAL]->(h) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_CANONICAL_PREDICATE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (p:{PREDICATE_LABEL} {{uuid: pair.predicate}}), '
    f'(h:{PREDICATE_HUB_LABEL} {{uuid: pair.hub}}) '
    f'MERGE (p)-[r:CANONICAL]->(h) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_HAS_DEFINITION = (
    f'UNWIND $pairs AS pair '
    f'MATCH (h {{uuid: pair.hub}}), '
    f'(d:{DEFINITION_LABEL} {{uuid: pair.definition}}) '
    f'MERGE (h)-[r:HAS_DEFINITION]->(d) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)


# ============================================================================
# Read queries — named lookbacks returning plain data
# ============================================================================


async def relation_types(
    session_factory: Callable,
    *,
    source: str | None = None,
) -> list[str]:
    """Every distinct relation type present in the graph.

    The living schema's ground truth: relation types are whatever edges the
    graph actually contains. The relation pass's write-time canonicalisation
    compares a proposed type against this list and reuses the existing name
    when one matches.

    Args:
        session_factory: A callable that returns an async context manager
            with a ``run(query, **params)`` method.
        source: The book identity to scope to, or None for the whole graph.

    Returns:
        The distinct relation type names, sorted.
    """
    params: dict[str, str] = {}
    if source is not None:
        params['source'] = source_uuid(source)
        cypher = (
            'MATCH (n {source: $source})-[r]->() '
            'RETURN DISTINCT type(r) AS type ORDER BY type'
        )
    else:
        cypher = (
            'MATCH ()-[r]->() RETURN DISTINCT type(r) AS type ORDER BY type'
        )

    async with session_factory() as session:
        result = await session.run(cypher, **params)
        records = await result.all()
    return [record['type'] for record in records]
