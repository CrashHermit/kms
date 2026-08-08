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

from kms.graph.community import COMMUNITY_LABEL
from kms.graph.definitions import DEFINITION_LABEL
from kms.graph.entities import ENTITY_LABEL
from kms.graph.entity_hubs import ENTITY_HUB_LABEL
from kms.graph.fact_hubs import FACT_HUB_LABEL
from kms.graph.facts import FACT_LABEL
from kms.graph.instructions import INSTRUCTION_LABEL
from kms.graph.nodes import NODE_LABEL, SOURCE_LABEL, source_uuid
from kms.graph.predicate_hubs import PREDICATE_HUB_LABEL
from kms.graph.predicates import PREDICATE_LABEL
from kms.graph.procedures import ACT_LABEL, PROCEDURE_LABEL
from kms.graph.statements import STATEMENT_LABEL
from kms.graph.triplet_hubs import TRIPLET_HUB_LABEL
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

MERGE_COMMUNITIES = (
    f'UNWIND $rows AS row '
    f'MERGE (c:{COMMUNITY_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET c.created_at = $now '
    f'SET c += row, c.modified_at = $now'
)

MERGE_COMMUNITY_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (c:{COMMUNITY_LABEL} {{uuid: pair.community}}), '
    f'(h {{uuid: pair.hub}}) '
    f'MERGE (c)-[r:HAS_MEMBER]->(h) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_COMMUNITY_EVIDENCE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (c:{COMMUNITY_LABEL} {{uuid: pair.community}}), '
    f'(t:{TRIPLET_LABEL} {{uuid: pair.triplet}}) '
    f'MERGE (c)-[r:COMMUNITY_EVIDENCE]->(t) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_TRIPLET_HUBS = (
    f'UNWIND $rows AS row '
    f'MERGE (th:{TRIPLET_HUB_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET th.created_at = $now '
    f'SET th += row, th.modified_at = $now'
)

MERGE_FACT_HUBS = (
    f'UNWIND $rows AS row '
    f'MERGE (fh:{FACT_HUB_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET fh.created_at = $now '
    f'SET fh += row, fh.modified_at = $now'
)

MERGE_HAS_FACT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (th:{TRIPLET_HUB_LABEL} {{uuid: pair.triplet_hub}}), '
    f'(fh:{FACT_HUB_LABEL} {{uuid: pair.fact_hub}}) '
    f'MERGE (th)-[r:HAS_FACT]->(fh) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_CANONICAL_SUBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (eh:{ENTITY_HUB_LABEL} {{uuid: pair.hub}}), '
    f'(th:{TRIPLET_HUB_LABEL} {{uuid: pair.triplet_hub}}) '
    f'MERGE (eh)-[r:CANONICAL_SUBJECT]->(th) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_CANONICAL_PREDICATE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (ph:{PREDICATE_HUB_LABEL} {{uuid: pair.hub}}), '
    f'(th:{TRIPLET_HUB_LABEL} {{uuid: pair.triplet_hub}}) '
    f'MERGE (ph)-[r:CANONICAL_PREDICATE]->(th) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_CANONICAL_OBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (eh:{ENTITY_HUB_LABEL} {{uuid: pair.hub}}), '
    f'(th:{TRIPLET_HUB_LABEL} {{uuid: pair.triplet_hub}}) '
    f'MERGE (eh)-[r:CANONICAL_OBJECT]->(th) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_SUPPORTED_BY = (
    f'UNWIND $pairs AS pair '
    f'MATCH (th:{TRIPLET_HUB_LABEL} {{uuid: pair.triplet_hub}}), '
    f'(t:{TRIPLET_LABEL} {{uuid: pair.triplet}}) '
    f'MERGE (th)-[r:SUPPORTED_BY]->(t) '
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


async def uncanonicalized_entity_spokes(
    session_factory: Callable,
    *,
    source: str,
) -> list[dict]:
    """Every ``:Entity`` that has an embedding but no ``:CANONICAL``
    edge yet — the new spokes awaiting canonicalization.

    Args:
        session_factory: The injected session factory.
        source: The stable book identity to scope to.

    Returns:
        One dict per spoke: ``{uuid, name, description, embedding}``.
    """
    cypher = (
        f'MATCH (e:{ENTITY_LABEL})\n'
        f'WHERE e.source = $source\n'
        f'  AND e.embedding IS NOT NULL\n'
        f'  AND NOT EXISTS {{ (e)-[:CANONICAL]->(:{ENTITY_HUB_LABEL}) }}\n'
        f'RETURN e.uuid AS uuid, e.name AS name, '
        f'e.description AS description, e.embedding AS embedding'
    )
    async with session_factory() as session:
        result = await session.run(
            cypher, source=source_uuid(source)
        )
        return [
            {
                'uuid': record['uuid'],
                'name': record['name'],
                'description': record.get('description'),
                'embedding': record['embedding'],
            }
            async for record in result
        ]


async def uncanonicalized_predicate_spokes(
    session_factory: Callable,
    *,
    source: str,
) -> list[dict]:
    """Every ``:Predicate`` that has an embedding but no
    ``:CANONICAL`` edge yet.

    Args:
        session_factory: The injected session factory.
        source: The stable book identity to scope to.

    Returns:
        One dict per spoke:
        ``{uuid, predicate, description, embedding}``.
    """
    cypher = (
        f'MATCH (p:{PREDICATE_LABEL})\n'
        f'WHERE p.source = $source\n'
        f'  AND p.embedding IS NOT NULL\n'
        f'  AND NOT EXISTS {{ (p)-[:CANONICAL]->(:{PREDICATE_HUB_LABEL}) }}\n'
        f'RETURN p.uuid AS uuid, p.predicate AS predicate, '
        f'p.description AS description, p.embedding AS embedding'
    )
    async with session_factory() as session:
        result = await session.run(
            cypher, source=source_uuid(source)
        )
        return [
            {
                'uuid': record['uuid'],
                'predicate': record['predicate'],
                'description': record.get('description'),
                'embedding': record['embedding'],
            }
            async for record in result
        ]


async def candidate_entity_hubs(
    session_factory: Callable,
    *,
    query_embedding: list[float],
    source: str | None = None,
    top_k: int = 5,
    min_score: float = 0.7,
) -> list[dict]:
    """Vector-search existing ``:EntityHub`` definitions for
    candidates similar to *query_embedding*.

    Args:
        session_factory: The injected session factory.
        query_embedding: The centroid of the new cluster's spoke
            embeddings.
        source: Optional source scope (None = cross-source).
        top_k: Maximum candidates to return.
        min_score: Minimum similarity score.

    Returns:
        One dict per candidate:
        ``{hub_uuid, display_name, aliases, definition_text,
          definition_embedding, score}``.
    """
    cypher = (
        f'CALL db.index.vector.queryNodes(\n'
        f'  "definition_embedding", $k, $query_embedding\n'
        f') YIELD node, score\n'
        f'WHERE score >= $min_score\n'
    )
    if source is not None:
        cypher += (
            f'MATCH (h:{ENTITY_HUB_LABEL} {{source: $source}})\n'
            f'  -[:HAS_DEFINITION]->(node)\n'
        )
    else:
        cypher += (
            f'MATCH (h:{ENTITY_HUB_LABEL})\n'
            f'  -[:HAS_DEFINITION]->(node)\n'
        )
    cypher += (
        f'RETURN h.uuid AS hub_uuid, '
        f'h.display_name AS display_name, '
        f'h.aliases AS aliases, '
        f'node.text AS definition_text, '
        f'node.embedding AS definition_embedding, '
        f'score\n'
        f'ORDER BY score DESC\n'
        f'LIMIT $top_k'
    )
    params = {
        'query_embedding': query_embedding,
        'k': top_k,
        'min_score': min_score,
        'top_k': top_k,
    }
    if source is not None:
        params['source'] = source_uuid(source)
    async with session_factory() as session:
        result = await session.run(cypher, **params)
        return [
            {
                'hub_uuid': record['hub_uuid'],
                'display_name': record['display_name'],
                'aliases': record.get('aliases') or [],
                'definition_text': record['definition_text'],
                'definition_embedding': record.get(
                    'definition_embedding'
                ),
                'score': record['score'],
            }
            async for record in result
        ]


async def candidate_predicate_hubs(
    session_factory: Callable,
    *,
    query_embedding: list[float],
    source: str | None = None,
    top_k: int = 5,
    min_score: float = 0.7,
) -> list[dict]:
    """Vector-search existing ``:PredicateHub`` definitions.

    Same shape as ``candidate_entity_hubs``.
    """
    cypher = (
        f'CALL db.index.vector.queryNodes(\n'
        f'  "definition_embedding", $k, $query_embedding\n'
        f') YIELD node, score\n'
        f'WHERE score >= $min_score\n'
    )
    if source is not None:
        cypher += (
            f'MATCH (h:{PREDICATE_HUB_LABEL} {{source: $source}})\n'
            f'  -[:HAS_DEFINITION]->(node)\n'
        )
    else:
        cypher += (
            f'MATCH (h:{PREDICATE_HUB_LABEL})\n'
            f'  -[:HAS_DEFINITION]->(node)\n'
        )
    cypher += (
        f'RETURN h.uuid AS hub_uuid, '
        f'h.display_name AS display_name, '
        f'h.aliases AS aliases, '
        f'node.text AS definition_text, '
        f'node.embedding AS definition_embedding, '
        f'score\n'
        f'ORDER BY score DESC\n'
        f'LIMIT $top_k'
    )
    params = {
        'query_embedding': query_embedding,
        'k': top_k,
        'min_score': min_score,
        'top_k': top_k,
    }
    if source is not None:
        params['source'] = source_uuid(source)
    async with session_factory() as session:
        result = await session.run(cypher, **params)
        return [
            {
                'hub_uuid': record['hub_uuid'],
                'display_name': record['display_name'],
                'aliases': record.get('aliases') or [],
                'definition_text': record['definition_text'],
                'definition_embedding': record.get(
                    'definition_embedding'
                ),
                'score': record['score'],
            }
            async for record in result
        ]


async def canonical_hub_triplets(
    session_factory: Callable,
    *,
    source: str,
) -> list[dict]:
    """Every canonical triplet at hub level — the EntityHub /
    PredicateHub graph used for community detection.

    Walks from each ``:Triplet`` through its ``:CANONICAL`` edges to
    find the hub for each of subject, predicate, and object.

    Args:
        session_factory: The injected session factory.
        source: The stable book identity to scope to.

    Returns:
        One dict per canonical triplet:
        ``{triplet_uuid, subj_hub, subj_name, subj_def,
           pred_hub, pred_name, pred_def,
           obj_hub, obj_name, obj_def}``.
    """
    from kms.graph.entity_hubs import ENTITY_HUB_LABEL
    from kms.graph.predicate_hubs import PREDICATE_HUB_LABEL
    from kms.graph.triplets import TRIPLET_LABEL

    cypher = (
        f'MATCH (t:{TRIPLET_LABEL})\n'
        f'WHERE t.source = $source\n'
        f'MATCH (t)-[:HAS_SUBJECT]->(es:Entity)\n'
        f'  -[:CANONICAL]->(sh:{ENTITY_HUB_LABEL})\n'
        f'MATCH (t)-[:HAS_PREDICATE]->(pp:Predicate)\n'
        f'  -[:CANONICAL]->(ph:{PREDICATE_HUB_LABEL})\n'
        f'MATCH (t)-[:HAS_OBJECT]->(eo:Entity)\n'
        f'  -[:CANONICAL]->(oh:{ENTITY_HUB_LABEL})\n'
        f'OPTIONAL MATCH (sh)-[:HAS_DEFINITION]->(sd:Definition)\n'
        f'OPTIONAL MATCH (ph)-[:HAS_DEFINITION]->(pd:Definition)\n'
        f'OPTIONAL MATCH (oh)-[:HAS_DEFINITION]->(od:Definition)\n'
        f'RETURN t.uuid AS triplet_uuid,\n'
        f'  sh.uuid AS subj_hub, sh.display_name AS subj_name,\n'
        f'  sd.text AS subj_def,\n'
        f'  ph.uuid AS pred_hub, ph.display_name AS pred_name,\n'
        f'  pd.text AS pred_def,\n'
        f'  oh.uuid AS obj_hub, oh.display_name AS obj_name,\n'
        f'  od.text AS obj_def'
    )
    async with session_factory() as session:
        result = await session.run(
            cypher, source=source_uuid(source)
        )
        return [
            {
                'triplet_uuid': record['triplet_uuid'],
                'subj_hub': record['subj_hub'],
                'subj_name': record['subj_name'],
                'subj_def': record.get('subj_def'),
                'pred_hub': record['pred_hub'],
                'pred_name': record['pred_name'],
                'pred_def': record.get('pred_def'),
                'obj_hub': record['obj_hub'],
                'obj_name': record['obj_name'],
                'obj_def': record.get('obj_def'),
            }
            async for record in result
        ]


async def vector_search_communities(
    session_factory: Callable,
    *,
    query_embedding: list[float],
    source: str,
    top_k: int = 20,
) -> list[dict]:
    """Vector search the ``community_summary`` index.

    Args:
        session_factory: The injected session factory.
        query_embedding: The query vector.
        source: The stable book identity to scope to.
        top_k: Maximum results.

    Returns:
        One dict per result:
        ``{uuid, summary_text, summary_embedding, score}``.
    """
    cypher = (
        f'CALL db.index.vector.queryNodes(\n'
        f'  "community_summary", $k, $query_embedding\n'
        f') YIELD node, score\n'
        f'WHERE node.source = $source\n'
        f'RETURN node.uuid AS uuid, '
        f'node.summary_text AS summary_text, '
        f'node.summary_embedding AS summary_embedding, '
        f'score\n'
        f'ORDER BY score DESC\n'
        f'LIMIT $top_k'
    )
    async with session_factory() as session:
        result = await session.run(
            cypher,
            query_embedding=query_embedding,
            k=top_k,
            source=source_uuid(source),
            top_k=top_k,
        )
        return [
            {
                'uuid': record['uuid'],
                'summary_text': record['summary_text'],
                'summary_embedding': record.get('summary_embedding'),
                'score': record['score'],
            }
            async for record in result
        ]


async def vector_search_hub_definitions(
    session_factory: Callable,
    *,
    query_embedding: list[float],
    source: str,
    kind: str,
    top_k: int = 20,
) -> list[dict]:
    """Vector search the ``definition_embedding`` index for hubs.

    Args:
        session_factory: The injected session factory.
        query_embedding: The query vector.
        source: The stable book identity.
        kind: ``'entity'`` or ``'predicate'``.
        top_k: Maximum results.

    Returns:
        One dict per result:
        ``{hub_uuid, display_name, definition_text, score}``.
    """
    hub_label = (
        ENTITY_HUB_LABEL if kind == 'entity' else PREDICATE_HUB_LABEL
    )
    cypher = (
        f'CALL db.index.vector.queryNodes(\n'
        f'  "definition_embedding", $k, $query_embedding\n'
        f') YIELD node, score\n'
        f'MATCH (h:{hub_label} {{source: $source}})\n'
        f'  -[:HAS_DEFINITION]->(node)\n'
        f'RETURN h.uuid AS hub_uuid, '
        f'h.display_name AS display_name, '
        f'node.text AS definition_text, '
        f'score\n'
        f'ORDER BY score DESC\n'
        f'LIMIT $top_k'
    )
    async with session_factory() as session:
        result = await session.run(
            cypher,
            query_embedding=query_embedding,
            k=top_k,
            source=source_uuid(source),
            top_k=top_k,
        )
        return [
            {
                'hub_uuid': record['hub_uuid'],
                'display_name': record['display_name'],
                'definition_text': record['definition_text'],
                'score': record['score'],
            }
            async for record in result
        ]


async def vector_search_nodes(
    session_factory: Callable,
    *,
    query_embedding: list[float],
    source: str,
    top_k: int = 20,
) -> list[dict]:
    """Vector search the ``node_content`` index on ``:Node``.

    Args:
        session_factory: The injected session factory.
        query_embedding: The query vector.
        source: The stable book identity.
        top_k: Maximum results.

    Returns:
        One dict per result:
        ``{uuid, content, type, index, segment_index, score}``.
    """
    cypher = (
        f'CALL db.index.vector.queryNodes(\n'
        f'  "node_content", $k, $query_embedding\n'
        f') YIELD node, score\n'
        f'WHERE node.source = $source\n'
        f'RETURN node.uuid AS uuid, '
        f'node.content AS content, '
        f'node.type AS type, '
        f'node.index AS index, '
        f'node.segment_index AS segment_index, '
        f'score\n'
        f'ORDER BY score DESC\n'
        f'LIMIT $top_k'
    )
    async with session_factory() as session:
        result = await session.run(
            cypher,
            query_embedding=query_embedding,
            k=top_k,
            source=source_uuid(source),
            top_k=top_k,
        )
        return [
            {
                'uuid': record['uuid'],
                'content': record['content'],
                'type': record['type'],
                'index': record['index'],
                'segment_index': record.get('segment_index'),
                'score': record['score'],
            }
            async for record in result
        ]
