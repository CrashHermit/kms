"""Cypher merge queries and read queries for the graph tier.

The MERGE_* constants are the idempotent upsert statements used by
``writer``; the functions build parameterized queries and run reads.
"""

from collections.abc import Callable

from kms import config
from kms.core import models
from kms.graph import (
    entities,
    events,
    hubs,
    instructions,
    learning,
    local_event_hubs,
    names,
    nodes,
    predicates,
    procedures,
    statements,
    triplets,
)
from kms.graph import local_entity_hubs as entity_hubs
from kms.graph import local_predicate_hubs as predicate_hubs
from kms.graph import (
    local_procedure_hubs as procedure_hubs,
)
from kms.graph import (
    local_statement_hubs as statement_hubs,
)

MERGE_SOURCE = (
    f'MERGE (s:{nodes.SOURCE_LABEL} {{uuid: $uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += $props, s.modified_at = $now'
)


def merge_nodes_query(label: str | None) -> str:
    """Builds the MERGE statement for AST node rows.

    Args:
        label: Optional extra label to apply to each merged node
            (e.g. 'Paragraph', 'Heading').
    """
    query = (
        f'UNWIND $rows AS row '
        f'MERGE (n:{nodes.NODE_LABEL} {{uuid: row.uuid}}) '
        f'ON CREATE SET n.created_at = $now '
        f'SET n += row, n.modified_at = $now'
    )
    if label:
        query += f' SET n:{label}'
    return query


MERGE_VISUAL_ASSETS = (
    f'UNWIND $rows AS row '
    f'MERGE (a:{nodes.VISUAL_ASSET_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET a.created_at = $now '
    f'SET a += row, a.modified_at = $now'
)


MERGE_NODE_ASSETS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{nodes.NODE_LABEL} {{uuid: pair.node}}), '
    f'(a:{nodes.VISUAL_ASSET_LABEL} {{uuid: pair.asset}}) '
    f'MERGE (n)-[r:HAS_ASSET]->(a) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)


MERGE_HEAD = (
    f'MATCH (s:{nodes.SOURCE_LABEL} {{uuid: $source}}), '
    f'(n:{nodes.NODE_LABEL} {{uuid: $head}}) '
    f'MERGE (s)-[r:HEAD]->(n) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_NEXT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (a:{nodes.NODE_LABEL} {{uuid: pair.from}}), '
    f'(b:{nodes.NODE_LABEL} {{uuid: pair.to}}) '
    f'MERGE (a)-[r:NEXT]->(b) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_STATEMENTS = (
    f'UNWIND $rows AS row '
    f'MERGE (s:{statements.STATEMENT_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += row, s.modified_at = $now'
)

MERGE_STATEMENT_ENRICHMENT = (
    f'UNWIND $rows AS row '
    f'MATCH (s:{statements.STATEMENT_LABEL} {{uuid: row.uuid}}) '
    f'SET s.description = row.description, s.embedding = row.embedding'
)

MERGE_STATEMENT_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{nodes.NODE_LABEL} {{uuid: pair.node}}), '
    f'(s:{statements.STATEMENT_LABEL} {{uuid: pair.statement}}) '
    f'MERGE (n)-[r:MEMBER_OF]->(s) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_INSTRUCTIONS = (
    f'UNWIND $rows AS row '
    f'MERGE (i:{instructions.INSTRUCTION_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET i.created_at = $now '
    f'SET i += row, i.modified_at = $now'
)

MERGE_INSTRUCTION_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{nodes.NODE_LABEL} {{uuid: pair.node}}), '
    f'(i:{instructions.INSTRUCTION_LABEL} {{uuid: pair.instruction}}) '
    f'MERGE (n)-[r:MEMBER_OF]->(i) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_INSTRUCTION_GOVERNANCE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (i:{instructions.INSTRUCTION_LABEL} {{uuid: pair.instruction}}), '
    f'(s:{statements.STATEMENT_LABEL} {{uuid: pair.statement}}) '
    f'MERGE (i)-[r:GOVERNS]->(s) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_PROCEDURES = (
    f'UNWIND $rows AS row '
    f'MERGE (p:{procedures.PROCEDURE_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET p.created_at = $now '
    f'SET p += row, p.modified_at = $now'
)

MERGE_PROCEDURE_ENRICHMENT = (
    f'UNWIND $rows AS row '
    f'MATCH (p:{procedures.PROCEDURE_LABEL} {{uuid: row.uuid}}) '
    f'SET p.procedure = row.procedure, p.embedding = row.embedding '
    f'REMOVE p.description'
)

MERGE_PROCEDURE_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{nodes.NODE_LABEL} {{uuid: pair.node}}), '
    f'(p:{procedures.PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
    f'MERGE (n)-[r:MEMBER_OF]->(p) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_TRIPLETS = (
    f'UNWIND $rows AS row '
    f'MERGE (t:{triplets.TRIPLET_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET t.created_at = $now '
    f'SET t += row, t.modified_at = $now'
)

MERGE_TRIPLET_EVIDENCE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{nodes.NODE_LABEL} {{uuid: pair.node}}), '
    f'(t:{triplets.TRIPLET_LABEL} {{uuid: pair.triplet}}) '
    f'MERGE (n)-[r:SUPPORTS]->(t) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_ENTITIES = (
    f'UNWIND $rows AS row '
    f'MERGE (e:{entities.ENTITY_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET e.created_at = $now '
    f'SET e += row, e.modified_at = $now'
)

MERGE_EVENTS = (
    f'UNWIND $rows AS row '
    f'MERGE (e:{events.EVENT_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET e.created_at = $now '
    f'SET e += row, e.modified_at = $now'
)


MERGE_PREDICATES = (
    f'UNWIND $rows AS row '
    f'MERGE (p:{predicates.PREDICATE_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET p.created_at = $now '
    f'SET p += row, p.modified_at = $now'
)

MERGE_ENTITY_NAMES = (
    f'UNWIND $rows AS row '
    f'MERGE (n:{names.ENTITY_NAME_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET n.created_at = $now '
    f'SET n += row, n.modified_at = $now'
)

MERGE_EVENT_NAMES = (
    f'UNWIND $rows AS row '
    f'MERGE (n:{names.EVENT_NAME_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET n.created_at = $now '
    f'SET n += row, n.modified_at = $now'
)


MERGE_PREDICATE_NAMES = (
    f'UNWIND $rows AS row '
    f'MERGE (n:{names.PREDICATE_NAME_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET n.created_at = $now '
    f'SET n += row, n.modified_at = $now'
)

MERGE_HAS_ENTITY_NAME = (
    f'UNWIND $pairs AS pair '
    f'MATCH (c:{entities.ENTITY_LABEL} {{uuid: pair.component}}), '
    f'(n:{names.ENTITY_NAME_LABEL} {{uuid: pair.name}}) '
    f'MERGE (c)-[:HAS_NAME]->(n)'
)

MERGE_HAS_EVENT_NAME = (
    f'UNWIND $pairs AS pair '
    f'MATCH (c:{events.EVENT_LABEL} {{uuid: pair.component}}), '
    f'(n:{names.EVENT_NAME_LABEL} {{uuid: pair.name}}) '
    f'MERGE (c)-[:HAS_NAME]->(n)'
)

MERGE_HAS_PREDICATE_NAME = (
    f'UNWIND $pairs AS pair '
    f'MATCH (c:{predicates.PREDICATE_LABEL} {{uuid: pair.component}}), '
    f'(n:{names.PREDICATE_NAME_LABEL} {{uuid: pair.name}}) '
    f'MERGE (c)-[:HAS_NAME]->(n)'
)


def merge_hubs_query(label: str) -> str:
    """Builds the MERGE statement for hub rows of one label.

    Args:
        label: The hub label to merge (EntityHub or PredicateHub).
    """
    return (
        f'UNWIND $rows AS row '
        f'MERGE (h:{label} {{uuid: row.uuid}}) '
        f'ON CREATE SET h.created_at = $now '
        f'SET h += row, h.modified_at = $now'
    )


def merge_statement_hubs_query() -> str:
    return merge_hubs_query(statement_hubs.hub_label())


def merge_procedure_hubs_query() -> str:
    return merge_hubs_query(procedure_hubs.hub_label())


def merge_statement_hub_memberships_query() -> str:
    base_label = statement_hubs.base_label()
    hub_label = statement_hubs.hub_label()
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (b:{base_label} {{uuid: pair.base}}), '
        f'(h:{hub_label} {{uuid: pair.hub}}) '
        f'MERGE (b)-[r:CANONICAL]->(h) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def merge_procedure_hub_memberships_query() -> str:
    base_label = procedure_hubs.base_label()
    hub_label = procedure_hubs.hub_label()
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (b:{base_label} {{uuid: pair.base}}), '
        f'(h:{hub_label} {{uuid: pair.hub}}) '
        f'MERGE (b)-[r:CANONICAL]->(h) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def merge_global_statement_hubs_query() -> str:
    global_label = statement_hubs.hub_label(tier='global')
    return (
        f'UNWIND $rows AS row '
        f'MERGE (h:{global_label} {{uuid: row.uuid}}) '
        f'ON CREATE SET h.created_at = $now '
        f'SET h += row, h.modified_at = $now'
    )


def merge_global_statement_alignments_query() -> str:
    local_label = statement_hubs.hub_label()
    global_label = statement_hubs.hub_label(tier='global')
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (s:{local_label} {{uuid: pair.source_hub}}), '
        f'(g:{global_label} {{uuid: pair.meta_hub}}) '
        f'MERGE (s)-[r:ALIGNS_TO]->(g) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def merge_global_procedure_hubs_query() -> str:
    global_label = procedure_hubs.hub_label(tier='global')
    return (
        f'UNWIND $rows AS row '
        f'MERGE (h:{global_label} {{uuid: row.uuid}}) '
        f'ON CREATE SET h.created_at = $now '
        f'SET h += row, h.modified_at = $now'
    )


def merge_global_procedure_alignments_query() -> str:
    local_label = procedure_hubs.hub_label()
    global_label = procedure_hubs.hub_label(tier='global')
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (s:{local_label} {{uuid: pair.source_hub}}), '
        f'(g:{global_label} {{uuid: pair.meta_hub}}) '
        f'MERGE (s)-[r:ALIGNS_TO]->(g) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def delete_global_statement_hubs_query() -> str:
    return (
        f'MATCH (h:{statement_hubs.hub_label(tier="global")}) DETACH DELETE h'
    )


def delete_global_procedure_hubs_query() -> str:
    return (
        f'MATCH (h:{procedure_hubs.hub_label(tier="global")}) DETACH DELETE h'
    )


def delete_statement_hubs_query() -> str:
    label = statement_hubs.hub_label()
    return (
        f'MATCH (h:{label}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}}) '
        f'WHERE src.key = $source '
        f'DETACH DELETE h'
    )


def delete_procedure_hubs_query() -> str:
    label = procedure_hubs.hub_label()
    return (
        f'MATCH (h:{label}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}}) '
        f'WHERE src.key = $source DETACH DELETE h'
    )


def merge_name_hubs_query(kind: str) -> str:
    """Builds the MERGE for source-local lexical hub rows."""
    label = names.name_hub_label(kind)
    return (
        f'UNWIND $rows AS row '
        f'MERGE (h:{label} {{uuid: row.uuid}}) '
        f'ON CREATE SET h.created_at = $now '
        f'SET h += row, h.modified_at = $now'
    )


def merge_meta_name_hubs_query(kind: str) -> str:
    """Builds the MERGE for cross-source lexical hub rows."""
    label = names.name_hub_label(kind, tier='meta')
    return (
        f'UNWIND $rows AS row '
        f'MERGE (h:{label} {{uuid: row.uuid}}) '
        f'ON CREATE SET h.created_at = $now '
        f'SET h += row, h.modified_at = $now'
    )


def merge_name_hub_alignments_query(kind: str) -> str:
    """Builds source lexical-hub→meta lexical-hub alignments."""
    source_label = names.name_hub_label(kind)
    meta_label = names.name_hub_label(kind, tier='meta')
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (s:{source_label} {{uuid: pair.source_hub}}), '
        f'(m:{meta_label} {{uuid: pair.meta_hub}}) '
        f'MERGE (s)-[r:ALIGNS_TO]->(m) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def delete_meta_name_hubs_query(kind: str) -> str:
    """Builds cleanup for the disposable meta lexical tier."""
    return (
        f'MATCH (h:{names.name_hub_label(kind, tier="meta")}) DETACH DELETE h'
    )


def merge_name_hub_memberships_query(kind: str) -> str:
    """Builds lexical occurrence→hub membership edges."""
    occurrence_label = names.name_label(kind)
    hub_label = names.name_hub_label(kind)
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (n:{occurrence_label} {{uuid: pair.name}}), '
        f'(h:{hub_label} {{uuid: pair.hub}}) '
        f'MERGE (n)-[:LEXICAL_CANONICAL]->(h)'
    )


def delete_name_hubs_query(kind: str) -> str:
    """Builds cleanup for source-local lexical hubs."""
    return (
        f'MATCH (h:{names.name_hub_label(kind)} {{source: $source_uuid}}) '
        f'DETACH DELETE h'
    )


def delete_semantic_name_links_query(kind: str) -> str:
    """Builds cleanup for semantic-hub→lexical-hub projections."""
    semantic_label = (
        entity_hubs.hub_label()
        if kind == 'entity'
        else predicate_hubs.hub_label()
    )
    return (
        f'MATCH (h:{semantic_label} {{source: $source_uuid}}) '
        f'-[r:HAS_NAME_HUB]->() DELETE r'
    )


def merge_semantic_name_links_query(kind: str) -> str:
    """Builds the occurrence-projection query for semantic hubs."""
    domain = entity_hubs if kind == 'entity' else predicate_hubs
    component_label = domain.component_label()
    semantic_label = domain.hub_label()
    occurrence_label = names.name_label(kind)
    name_hub_label = names.name_hub_label(kind)
    return (
        f'MATCH (h:{semantic_label} {{source: $source_uuid}})'
        f'<-[:CANONICAL]-(c:{component_label})'
        f'-[:HAS_NAME]->(n:{occurrence_label})'
        f'-[:LEXICAL_CANONICAL]->(nh:{name_hub_label}) '
        f'MERGE (h)-[:HAS_NAME_HUB]->(nh)'
    )


def merge_triplet_hubs_query(tier: str) -> str:
    """Builds the MERGE for derived TripletHub rows."""
    label = hubs.triplet_hub_label(tier)
    return (
        f'UNWIND $rows AS row '
        f'MERGE (h:{label} {{uuid: row.uuid}}) '
        f'ON CREATE SET h.created_at = $now '
        f'SET h += row, h.modified_at = $now'
    )


def merge_triplet_hub_edges_query(tier: str) -> str:
    """Builds role edges for a source or meta TripletHub."""
    label = hubs.triplet_hub_label(tier)
    if tier == 'source':
        return (
            f'UNWIND $rows AS row '
            f'MATCH (h:{label} {{uuid: row.hub}}), '
            f'(s {{uuid: row.subject_hub}}), '
            f'(p:{hubs.PREDICATE_HUB_LABEL} {{uuid: row.predicate_hub}}), '
            f'(o {{uuid: row.object_hub}}) '
            f'WHERE (s:{hubs.ENTITY_HUB_LABEL} OR '
            f's:{hubs.LOCAL_EVENT_HUB_LABEL}) AND '
            f'(o:{hubs.ENTITY_HUB_LABEL} OR '
            f'o:{hubs.LOCAL_EVENT_HUB_LABEL}) '
            f'MERGE (h)-[:HAS_SUBJECT_HUB]->(s) '
            f'MERGE (h)-[:HAS_PREDICATE_HUB]->(p) '
            f'MERGE (h)-[:HAS_OBJECT_HUB]->(o)'
        )
    return (
        f'UNWIND $rows AS row '
        f'MATCH (h:{label} {{uuid: row.hub}}), '
        f'(s:{hubs.META_ENTITY_HUB_LABEL} {{uuid: row.subject_hub}}), '
        f'(p:{hubs.META_PREDICATE_HUB_LABEL} {{uuid: row.predicate_hub}}), '
        f'(o:{hubs.META_ENTITY_HUB_LABEL} {{uuid: row.object_hub}}) '
        f'MERGE (h)-[:HAS_GLOBAL_SUBJECT_HUB]->(s) '
        f'MERGE (h)-[:HAS_GLOBAL_PREDICATE_HUB]->(p) '
        f'MERGE (h)-[:HAS_GLOBAL_OBJECT_HUB]->(o)'
    )


def merge_triplet_hub_evidence_query() -> str:
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (t:{triplets.TRIPLET_LABEL} {{uuid: pair.triplet}}), '
        f'(h:{hubs.TRIPLET_HUB_LABEL} {{uuid: pair.hub}}) '
        f'MERGE (t)-[:CANONICAL]->(h)'
    )


def merge_meta_triplet_hub_support_query() -> str:
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (h:{hubs.TRIPLET_HUB_LABEL} {{uuid: pair.source_hub}}), '
        f'(m:{hubs.META_TRIPLET_HUB_LABEL} {{uuid: pair.meta_hub}}) '
        f'MERGE (h)-[:ALIGNS_TO]->(m)'
    )


def delete_triplet_hubs_query(tier: str, source: str | None = None) -> str:
    label = hubs.triplet_hub_label(tier)
    if source is None:
        return f'MATCH (h:{label}) DETACH DELETE h'
    return f'MATCH (h:{label} {{source: $source_uuid}}) DETACH DELETE h'


MERGE_HAS_SUBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{triplets.TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(e {{uuid: pair.entity}}) '
    f'WHERE e:{entities.ENTITY_LABEL} OR e:{events.EVENT_LABEL} '
    f'MERGE (t)-[r:HAS_SUBJECT]->(e) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_HAS_PROCEDURE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (s:{statements.STATEMENT_LABEL} {{uuid: pair.statement}}), '
    f'(p:{procedures.PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
    f'MERGE (s)-[r:HAS_PROCEDURE]->(p) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_HAS_OBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{triplets.TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(e {{uuid: pair.entity}}) '
    f'WHERE e:{entities.ENTITY_LABEL} OR e:{events.EVENT_LABEL} '
    f'MERGE (t)-[r:HAS_OBJECT]->(e) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_HAS_PREDICATE = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{triplets.TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(p:{predicates.PREDICATE_LABEL} {{uuid: pair.predicate}}) '
    f'MERGE (t)-[r:HAS_PREDICATE]->(p) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)


def update_hub_aliases_query(label: str) -> str:
    return (
        f'UNWIND $rows AS row '
        f'MATCH (h:{label} {{uuid: row.hub}}) '
        f'SET h.aliases = coalesce(h.aliases, []) + '
        f'[alias IN row.aliases WHERE NOT alias IN coalesce(h.aliases, [])], '
        f'h.modified_at = $now'
    )


def merge_canonical_query(component_label: str, hub_label: str) -> str:
    """Builds the MERGE statement for component→hub CANONICAL edges.

    Args:
        component_label: The label of the component vertex.
        hub_label: The label of the hub vertex.
    """
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (c:{component_label} {{uuid: pair.component}}), '
        f'(h:{hub_label} {{uuid: pair.hub}}) '
        f'MERGE (c)-[r:CANONICAL]->(h) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def merge_subsumes_query(label: str) -> str:
    """Builds the MERGE statement for hub→hub SUBSUMES edges.

    Args:
        label: The hub label shared by both endpoints.
    """
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (g:{label} {{uuid: pair.general}}), '
        f'(s:{label} {{uuid: pair.specific}}) '
        f'MERGE (g)-[r:SUBSUMES]->(s) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now'
    )


def delete_hub_alignments_query(source_hub_label: str) -> str:
    return (
        f'UNWIND $source_hubs AS source_hub '
        f'MATCH (s:{source_hub_label} {{uuid: source_hub}})'
        f'-[r:ALIGNS_TO]->() '
        f'DELETE r'
    )


def merge_alignment_query(source_hub_label: str, meta_hub_label: str) -> str:
    """Builds the MERGE for a source-hub→meta-hub alignment.

    Alignment is deliberately distinct from ``CANONICAL``: a source hub
    remains source-local evidence, while the meta hub is a cross-source
    interpretation that may be rebuilt independently.
    """
    return (
        f'UNWIND $pairs AS pair '
        f'MATCH (s:{source_hub_label} {{uuid: pair.source_hub}}), '
        f'(m:{meta_hub_label} {{uuid: pair.meta_hub}}) '
        f'MERGE (s)-[r:ALIGNS_TO]->(m) '
        f'ON CREATE SET r.created_at = $now '
        f'SET r.modified_at = $now, '
        f'r.score = pair.score, r.decision = pair.decision'
    )


def delete_invalid_meta_hubs_query(label: str) -> str:
    return (
        f'MATCH (h:{label}) '
        f'OPTIONAL MATCH (h)<-[:ALIGNS_TO]-(s) '
        f'WITH h, count(DISTINCT s.source) AS source_count '
        f'WHERE source_count < 2 DETACH DELETE h'
    )


def delete_hubs_query(label: str) -> str:
    """Builds a derived-hub cleanup query for one label.

    This is intended for the disposable meta tier. Callers must never pass
    a durable component label or a source-local hub label accidentally.
    """
    return f'MATCH (h:{label}) DETACH DELETE h'


def delete_source_hubs_query(label: str) -> str:
    """Builds a cleanup query for source-local hubs from one source."""
    return (
        f'MATCH (src:{nodes.SOURCE_LABEL} {{key: $source}}), '
        f'(h:{label} {{source: src.uuid}}) DETACH DELETE h'
    )


async def _all_components(
    session_factory: Callable,
    *,
    label: str,
    name_field: str,
    source: str,
) -> list[dict]:
    cypher = (
        f'MATCH (c:{label})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: c.source}})\n'
        f'WHERE src.key = $source\n'
        f'RETURN c.uuid AS uuid, c.{name_field} AS name, '
        f'c.node_position AS node_position, c.description AS description, '
        f'c.embedding AS embedding, src.key AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher, source=source)
        return [
            {
                'uuid': record['uuid'],
                'name': record['name'],
                'node_position': record['node_position'],
                'description': record.get('description'),
                'embedding': record.get('embedding'),
                'source': record.get('source'),
            }
            async for record in result
        ]


async def all_components(
    session_factory: Callable, kind: str, source: str
) -> list[dict]:
    if kind == 'entity':
        return await all_entity_components(session_factory, source)
    if kind == 'event':
        return await all_event_components(session_factory, source)
    if kind == 'predicate':
        return await all_predicate_components(session_factory, source)
    raise ValueError(f'unknown component kind: {kind}')


async def all_entity_components(
    session_factory: Callable, source: str
) -> list[dict]:
    return await _all_components(
        session_factory,
        label=entities.ENTITY_LABEL,
        name_field='name',
        source=source,
    )


async def all_event_components(
    session_factory: Callable, source: str
) -> list[dict]:
    return await _all_components(
        session_factory,
        label=events.EVENT_LABEL,
        name_field='name',
        source=source,
    )


async def all_predicate_components(
    session_factory: Callable, source: str
) -> list[dict]:
    return await _all_components(
        session_factory,
        label=predicates.PREDICATE_LABEL,
        name_field='predicate',
        source=source,
    )


async def all_name_occurrences(
    session_factory: Callable,
    kind: str,
    source: str,
) -> list[dict]:
    """Reads lexical occurrence nodes for one source and component kind."""
    occurrence_label = names.name_label(kind)
    cypher = (
        f'MATCH (n:{occurrence_label})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: n.source}})\n'
        f'WHERE src.key = $source\n'
        f'RETURN n.uuid AS uuid, n.text AS text, '
        f'n.normalized_text AS normalized_text, '
        f'n.component_uuid AS component_uuid, src.key AS source\n'
        f'ORDER BY n.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher, source=source)
        return [dict(record) async for record in result]


async def all_name_hubs(
    session_factory: Callable,
    kind: str,
) -> list[dict]:
    """Reads all source-local lexical hubs with source-key provenance."""
    label = names.name_hub_label(kind)
    cypher = (
        f'MATCH (h:{label})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}})\n'
        f'RETURN h.uuid AS uuid, h.canonical_form AS text, '
        f'h.normalized_form AS normalized_text, h.aliases AS aliases, '
        f'src.key AS source\n'
        f'ORDER BY h.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [dict(record) async for record in result]


async def lexical_name_hubs(
    session_factory: Callable,
    kind: str,
    text: str,
    source: str | None = None,
    tier: str = 'source',
) -> list[dict]:
    """Finds lexical hubs by exact normalized phrase."""
    hub_label = names.name_hub_label(kind, tier=tier)
    conditions = ['h.normalized_form = $normalized']
    params = {'normalized': names.normalize_text(text)}
    source_match = ''
    source_return = 'NULL AS source, '
    if source is not None:
        if tier != 'source':
            raise ValueError('meta lexical hubs do not accept source filters')
        conditions.append('src.key = $source')
        params['source'] = source
        source_match = f'\nMATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}})'
        source_return = 'src.key AS source, '
    cypher = (
        f'MATCH (h:{hub_label}){source_match}\n'
        f'WHERE {" AND ".join(conditions)}\n'
        f'RETURN h.uuid AS uuid, h.canonical_form AS canonical_form, '
        f'h.aliases AS aliases, {source_return}h.sources AS sources\n'
        f'ORDER BY h.canonical_form'
    )
    async with session_factory() as session:
        result = await session.run(cypher, **params)
        return [dict(record) async for record in result]


async def _unassigned_components(
    session_factory: Callable,
    *,
    label: str,
    name_field: str,
    hub_label: str,
    source: str,
) -> list[dict]:
    cypher = (
        f'MATCH (c:{label})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: c.source}})\n'
        f'WHERE src.key = $source AND NOT '
        f'(c)-[:CANONICAL]->(:{hub_label})\n'
        f'RETURN c.uuid AS uuid, c.{name_field} AS name, '
        f'c.description AS description, c.embedding AS embedding, src.key AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher, source=source)
        return [
            {
                'uuid': record['uuid'],
                'name': record['name'],
                'aliases': [],
                'description': record.get('description'),
                'embedding': record.get('embedding'),
                'source': record.get('source'),
            }
            async for record in result
        ]


async def unassigned_components(
    session_factory: Callable, kind: str, source: str
) -> list[dict]:
    if kind == 'entity':
        return await unassigned_entity_components(session_factory, source)
    if kind == 'event':
        raise ValueError(
            'event components do not use canonicalized assignment queries'
        )
    if kind == 'predicate':
        return await unassigned_predicate_components(session_factory, source)
    raise ValueError(f'unknown component kind: {kind}')


async def unassigned_entity_components(
    session_factory: Callable, source: str
) -> list[dict]:
    return await _unassigned_components(
        session_factory,
        label=entities.ENTITY_LABEL,
        name_field='name',
        hub_label=entity_hubs.hub_label(),
        source=source,
    )


async def unassigned_predicate_components(
    session_factory: Callable, source: str
) -> list[dict]:
    return await _unassigned_components(
        session_factory,
        label=predicates.PREDICATE_LABEL,
        name_field='predicate',
        hub_label=predicate_hubs.hub_label(),
        source=source,
    )


async def all_source_hubs(
    session_factory: Callable,
    kind: str,
    source: str | None = None,
    hub_uuids: list[str] | None = None,
) -> list[dict]:
    return await _all_source_hubs(
        session_factory,
        label=(
            entity_hubs.hub_label()
            if kind == 'entity'
            else local_event_hubs.hub_label()
            if kind == 'event'
            else predicate_hubs.hub_label()
        ),
        hub_uuids=hub_uuids,
    )


async def _all_source_hubs(
    session_factory: Callable,
    *,
    label: str,
    source: str | None = None,
    hub_uuids: list[str] | None = None,
) -> list[dict]:
    conditions: list[str] = []
    params: dict = {}
    if source is not None:
        conditions.append('h.source = $source_uuid')
        params['source_uuid'] = nodes.source_uuid(source)
    if hub_uuids is not None:
        conditions.append('h.uuid IN $hub_uuids')
        params['hub_uuids'] = hub_uuids
    where_clause = f'WHERE {" AND ".join(conditions)}\n' if conditions else ''
    cypher = (
        f'MATCH (h:{label})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}})\n'
        f'{where_clause}'
        f'RETURN h.uuid AS uuid, '
        f'h.canonical_name AS name, '
        f'h.aliases AS aliases, '
        f'h.description AS description, '
        f'h.embedding AS embedding, '
        f'src.key AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher, **params)
        return [
            {
                'uuid': record['uuid'],
                'name': record['name'],
                'aliases': record.get('aliases') or [],
                'description': record.get('description'),
                'embedding': record.get('embedding'),
                'source': record.get('source'),
            }
            async for record in result
        ]


async def all_entity_source_hubs(
    session_factory: Callable,
    source: str | None = None,
    hub_uuids: list[str] | None = None,
) -> list[dict]:
    return await _all_source_hubs(
        session_factory,
        label=entity_hubs.hub_label(),
        source=source,
        hub_uuids=hub_uuids,
    )


async def all_event_source_hubs(
    session_factory: Callable,
    source: str | None = None,
    hub_uuids: list[str] | None = None,
) -> list[dict]:
    return await _all_source_hubs(
        session_factory,
        label=local_event_hubs.hub_label(),
        source=source,
        hub_uuids=hub_uuids,
    )


async def all_predicate_source_hubs(
    session_factory: Callable,
    source: str | None = None,
    hub_uuids: list[str] | None = None,
) -> list[dict]:
    return await _all_source_hubs(
        session_factory,
        label=predicate_hubs.hub_label(),
        source=source,
        hub_uuids=hub_uuids,
    )


async def qualified_entity_meta_hub_uuids(
    session_factory: Callable,
) -> set[str]:
    return await _qualified_meta_hub_uuids(
        session_factory,
        source_hub_label=entity_hubs.hub_label(),
        meta_hub_label=entity_hubs.hub_label('meta'),
    )


async def qualified_predicate_meta_hub_uuids(
    session_factory: Callable,
) -> set[str]:
    return await _qualified_meta_hub_uuids(
        session_factory,
        source_hub_label=predicate_hubs.hub_label(),
        meta_hub_label=predicate_hubs.hub_label('meta'),
    )


async def qualified_event_meta_hub_uuids(
    session_factory: Callable,
) -> set[str]:
    return await _qualified_meta_hub_uuids(
        session_factory,
        source_hub_label=local_event_hubs.hub_label(),
        meta_hub_label=local_event_hubs.hub_label('meta'),
    )


async def triplet_hub_groups(
    session_factory: Callable,
    tier: str,
    source: str | None = None,
) -> list[dict]:
    """Returns exact three-way triplet groups for a derived hub tier.

    Each result row is grouped by the complete ordered subject, predicate,
    and object hub tuple. The triplet remains the anchor, so unrelated facts
    sharing only one or two roles cannot enter the same group.
    """
    params: dict = {}
    if tier == 'source':
        source_filter = 'WHERE src.key = $source\n' if source else ''
        params = {'source': source} if source else {}
        cypher = (
            f'MATCH (t:{triplets.TRIPLET_LABEL})\n'
            f'MATCH (t)-[:HAS_SUBJECT]->(s)-[:CANONICAL]->(sh)'
            f' WHERE sh:{hubs.ENTITY_HUB_LABEL} OR '
            f'sh:{hubs.LOCAL_EVENT_HUB_LABEL}\n'
            f'MATCH (t)-[:HAS_PREDICATE]->(p:{predicates.PREDICATE_LABEL})-[:CANONICAL]->'
            f'(ph:{hubs.PREDICATE_HUB_LABEL})\n'
            f'MATCH (t)-[:HAS_OBJECT]->(o)-[:CANONICAL]->(oh)'
            f' WHERE oh:{hubs.ENTITY_HUB_LABEL} OR '
            f'oh:{hubs.LOCAL_EVENT_HUB_LABEL}\n'
            f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: t.source}})\n'
            f'{source_filter}'
            f'WITH sh, ph, oh, src.key AS source, t, s, p, o\n'
            f'WITH sh, ph, oh, source, '
            f'collect(DISTINCT t.uuid) AS triplets, '
            f'collect(DISTINCT {{uuid: t.uuid, subject: s.name, '
            f'predicate: p.predicate, object: o.name}}) AS evidence\n'
            f'RETURN sh.uuid AS subject_hub, '
            f'sh.canonical_name AS subject_name, '
            f'sh.description AS subject_description, '
            f'ph.uuid AS predicate_hub, '
            f'ph.canonical_name AS predicate_name, '
            f'ph.description AS predicate_description, '
            f'oh.uuid AS object_hub, '
            f'oh.canonical_name AS object_name, '
            f'oh.description AS object_description, '
            f'source, triplets, evidence'
        )
    elif tier == 'meta':
        cypher = (
            f'MATCH (t:{triplets.TRIPLET_LABEL})\n'
            f'MATCH (t)-[:HAS_SUBJECT]->'
            f'(s:{entities.ENTITY_LABEL})-[:CANONICAL]->'
            f'(sh:{hubs.ENTITY_HUB_LABEL})-[:ALIGNS_TO]->'
            f'(msh:{hubs.META_ENTITY_HUB_LABEL})\n'
            f'MATCH (t)-[:HAS_PREDICATE]->'
            f'(p:{predicates.PREDICATE_LABEL})-[:CANONICAL]->'
            f'(ph:{hubs.PREDICATE_HUB_LABEL})-[:ALIGNS_TO]->'
            f'(mph:{hubs.META_PREDICATE_HUB_LABEL})\n'
            f'MATCH (t)-[:HAS_OBJECT]->'
            f'(o:{entities.ENTITY_LABEL})-[:CANONICAL]->'
            f'(oh:{hubs.ENTITY_HUB_LABEL})-[:ALIGNS_TO]->'
            f'(moh:{hubs.META_ENTITY_HUB_LABEL})\n'
            f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: t.source}})\n'
            f'WITH msh, mph, moh, sh, ph, oh, src.key AS source, '
            f't, s, p, o\n'
            f'WITH msh, mph, moh, '
            f'collect(DISTINCT t.uuid) AS triplets, '
            f'collect(DISTINCT {{uuid: t.uuid, subject: s.name, '
            f'predicate: p.predicate, object: o.name}}) AS evidence, '
            f'collect(DISTINCT {{source: source, subject_hub: sh.uuid, '
            f'predicate_hub: ph.uuid, object_hub: oh.uuid}}) '
            f'AS local_tuples, collect(DISTINCT source) AS sources\n'
            f'WHERE size(sources) >= 2\n'
            f'RETURN msh.uuid AS subject_hub, '
            f'msh.canonical_name AS subject_name, '
            f'msh.description AS subject_description, '
            f'mph.uuid AS predicate_hub, '
            f'mph.canonical_name AS predicate_name, '
            f'mph.description AS predicate_description, '
            f'moh.uuid AS object_hub, '
            f'moh.canonical_name AS object_name, '
            f'moh.description AS object_description, '
            f'local_tuples, sources, triplets, evidence'
        )
    else:
        raise ValueError(f'unknown triplet hub tier: {tier}')

    async with session_factory() as session:
        result = await session.run(cypher, **params)
        return [dict(record) async for record in result]


async def qualified_meta_hub_uuids(
    session_factory: Callable, kind: str
) -> set[str]:
    return await _qualified_meta_hub_uuids(
        session_factory,
        source_hub_label=(
            entity_hubs.hub_label()
            if kind == 'entity'
            else predicate_hubs.hub_label()
        ),
        meta_hub_label=(
            entity_hubs.hub_label('meta')
            if kind == 'entity'
            else predicate_hubs.hub_label('meta')
        ),
    )


async def _qualified_meta_hub_uuids(
    session_factory: Callable,
    *,
    source_hub_label: str,
    meta_hub_label: str,
) -> set[str]:
    cypher = (
        f'MATCH (m:{meta_hub_label})\n'
        f'OPTIONAL MATCH (m)<-[:ALIGNS_TO]-(s:{source_hub_label})\n'
        f'WITH m, count(DISTINCT s.source) AS source_count\n'
        f'WHERE source_count >= 2\n'
        f'RETURN m.uuid AS uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return {record['uuid'] async for record in result}


_VECTOR_INDEX_LABELS = {
    'node_content': nodes.NODE_LABEL,
    'statement_embedding': statements.STATEMENT_LABEL,
    'procedure_embedding': procedures.PROCEDURE_LABEL,
    'entity_embedding': entities.ENTITY_LABEL,
    'predicate_embedding': predicates.PREDICATE_LABEL,
    'local_statement_hub_embedding': statement_hubs.LOCAL_STATEMENT_HUB_LABEL,
    'global_statement_hub_embedding': statement_hubs.GLOBAL_STATEMENT_HUB_LABEL,
    'local_procedure_hub_embedding': procedure_hubs.LOCAL_PROCEDURE_HUB_LABEL,
    'global_procedure_hub_embedding': procedure_hubs.GLOBAL_PROCEDURE_HUB_LABEL,
    'triplet_hub_embedding': hubs.LOCAL_TRIPLET_HUB_LABEL,
    'meta_triplet_hub_embedding': hubs.GLOBAL_TRIPLET_HUB_LABEL,
}


async def vector_search(
    session_factory: Callable,
    *,
    index_name: str,
    query_embedding: list[float],
    top_k: int | None = None,
    source: str | None = None,
) -> list[dict]:
    """Runs a cosine-similarity vector search against a Neo4j index.

    Returns every property on each matched node, plus the similarity
    score.  Optionally filters by source when the index supports it.

    Args:
        session_factory: Async callable returning a Neo4j session.
        index_name: Name of the Neo4j vector index to query.
        query_embedding: The query vector to search with.
        top_k: Maximum number of results to return.
        source: Optional source filter on node.source.

    Returns:
        A list of dicts with all node properties plus 'score'.
    """
    if top_k is None:
        top_k = config.get_settings().stages.search.top_k
    params: dict = {
        'query_embedding': query_embedding,
        'k': top_k,
        'top_k': top_k,
        'index_name': index_name,
    }
    where_clause = ''
    if source:
        where_clause = 'WHERE node.source = $source\n'
        params['source'] = nodes.source_uuid(source)

    cypher = (
        f'CALL db.index.vector.queryNodes(\n'
        f'  $index_name, $k, $query_embedding\n'
        f') YIELD node, score\n'
        f'{where_clause}'
        f'RETURN properties(node) AS properties, '
        f'score\n'
        f'ORDER BY score DESC\n'
        f'LIMIT $top_k'
    )
    async with session_factory() as session:
        if source and index_name in _VECTOR_INDEX_LABELS:
            label = _VECTOR_INDEX_LABELS[index_name]
            count_result = await session.run(
                f'MATCH (node:{label}) '
                f'WHERE node.source = $source '
                f'RETURN count(node) AS count',
                source=nodes.source_uuid(source),
            )
            count_record = await count_result.single()
            source_count = count_record['count'] if count_record else 0
            params['k'] = max(top_k, source_count)
        result = await session.run(cypher, **params)
        return [
            {
                **record['properties'],
                'score': record['score'],
            }
            async for record in result
        ]


async def all_source_statement_hubs(
    session_factory: Callable,
) -> list[dict]:
    label = statement_hubs.hub_label()
    cypher = (
        f'MATCH (h:{label}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}}) '
        f'RETURN h.uuid AS uuid, h.description AS description, '
        f'h.embedding AS embedding, src.key AS source '
        f'ORDER BY h.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [dict(record) async for record in result]


async def all_source_procedure_hubs(
    session_factory: Callable,
) -> list[dict]:
    label = procedure_hubs.hub_label()
    cypher = (
        f'MATCH (h:{label}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}}) '
        f'RETURN h.uuid AS uuid, h.description AS description, '
        f'h.embedding AS embedding, src.key AS source '
        f'ORDER BY h.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [dict(record) async for record in result]


async def statement_hub_items(
    session_factory: Callable,
    source: str,
) -> list[dict]:
    label = statement_hubs.base_label()
    description = 'b.description'
    cypher = (
        f'MATCH (b:{label}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: b.source}}) '
        f'WHERE src.key = $source '
        f'RETURN b.uuid AS uuid, {description} AS description, '
        f'b.embedding AS embedding, src.key AS source '
        f'ORDER BY b.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher, source=source)
        return [dict(record) async for record in result]


async def procedure_hub_items(
    session_factory: Callable,
    source: str,
) -> list[dict]:
    label = procedure_hubs.base_label()
    cypher = (
        f'MATCH (b:{label}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: b.source}}) '
        f'WHERE src.key = $source AND b.kind = "source" '
        f'RETURN b.uuid AS uuid, b.procedure AS description, '
        f'b.embedding AS embedding, src.key AS source '
        f'ORDER BY b.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher, source=source)
        return [dict(record) async for record in result]


# Card and Review queries
MERGE_CARDS = (
    f'UNWIND $rows AS row '
    f'MERGE (c:{learning.CARD_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET c.created_at = $now '
    f'SET c += row, c.modified_at = $now'
)

MERGE_CARD_TARGET_EDGES = (
    f'UNWIND $pairs AS pair '
    f'MATCH (target {{uuid: pair.target}}) '
    f'MATCH (c:{learning.CARD_LABEL} {{uuid: pair.card}}) '
    f'MERGE (target)-[:HAS_CARD]->(c)'
)

MERGE_REVIEWS = (
    f'UNWIND $rows AS row '
    f'MERGE (r:{learning.REVIEW_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r += row, r.modified_at = $now'
)

MERGE_CARD_REVIEW_EDGES = (
    f'UNWIND $pairs AS pair '
    f'MATCH (c:{learning.CARD_LABEL} {{uuid: pair.card}}) '
    f'MATCH (r:{learning.REVIEW_LABEL} {{uuid: pair.review}}) '
    f'MERGE (c)-[:HAS_REVIEW]->(r)'
)


def merge_cards_query() -> str:
    return MERGE_CARDS


def merge_card_target_edges_query() -> str:
    return MERGE_CARD_TARGET_EDGES


def merge_reviews_query() -> str:
    return MERGE_REVIEWS


def _card_hub_labels(target_kind: models.CardTargetKind) -> tuple[str, str]:
    """Returns the local and global hub labels for one card target kind."""
    if target_kind == models.CardTargetKind.PROCEDURE:
        return procedure_hubs.hub_label(), procedure_hubs.hub_label(
            tier='global'
        )
    if target_kind == models.CardTargetKind.TRIPLET:
        return hubs.local_triplet_hub_label(), hubs.global_triplet_hub_label()
    return hubs.local_hub_label(target_kind), hubs.global_hub_label(target_kind)


async def card_hub_context(
    session_factory: Callable,
    *,
    target_kind: models.CardTargetKind,
    hub_uuid: str,
) -> dict:
    """Loads a selected local card hub and optional global context."""
    local_label, global_label = _card_hub_labels(target_kind)
    cypher = (
        f'MATCH (h:{local_label} {{uuid: $hub_uuid}}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}}) '
        f'OPTIONAL MATCH (h)-[:ALIGNS_TO]->(g:{global_label}) '
        f'RETURN h.uuid AS local_uuid, h.canonical_name AS local_name, '
        f'h.description AS local_description, g.uuid AS global_uuid, '
        f'g.canonical_name AS global_name, g.description AS global_description, '
        f'src.key AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher, hub_uuid=hub_uuid)
        record = await result.single()
    if record is None:
        raise ValueError(f'card hub not found: {hub_uuid}')
    return dict(record)


def _card_targets_query(target_kind: models.CardTargetKind) -> str:
    """Builds the source-local component lookup for one hub type."""
    local_label, _ = _card_hub_labels(target_kind)
    if target_kind in {
        models.CardTargetKind.ENTITY,
        models.CardTargetKind.EVENT,
    }:
        label = hubs.component_label(target_kind)
        return (
            f'MATCH (target:{label})-[:CANONICAL]->'
            f'(hub:{local_label} {{uuid: $hub_uuid}}) '
            f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: target.source}}) '
            f'RETURN target.uuid AS uuid, $kind AS kind, src.key AS source, '
            f'target.name + CASE WHEN target.description IS NULL THEN "" '
            f'ELSE ": " + target.description END AS content, {{}} AS context '
        )
    if target_kind == models.CardTargetKind.TRIPLET:
        return (
            f'MATCH (target:{triplets.TRIPLET_LABEL})-[:CANONICAL]->'
            f'(hub:{local_label} {{uuid: $hub_uuid}}) '
            f'MATCH (target)-[:HAS_SUBJECT]->(subject:{entities.ENTITY_LABEL}) '
            f'MATCH (target)-[:HAS_PREDICATE]->(predicate:{predicates.PREDICATE_LABEL}) '
            f'MATCH (target)-[:HAS_OBJECT]->(object:{entities.ENTITY_LABEL}) '
            f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: target.source}}) '
            f'RETURN target.uuid AS uuid, $kind AS kind, src.key AS source, '
            f'subject.name + " --" + predicate.predicate + "--> " + object.name AS content, '
            f'{{subject: {{uuid: subject.uuid, name: subject.name, description: subject.description}}, '
            f'predicate: {{uuid: predicate.uuid, name: predicate.predicate, description: predicate.description}}, '
            f'object: {{uuid: object.uuid, name: object.name, description: object.description}}}} AS context '
            f'ORDER BY target.uuid'
        )
    return (
        f'MATCH (target:{procedures.PROCEDURE_LABEL})-[:CANONICAL]->'
        f'(hub:{local_label} {{uuid: $hub_uuid}}) '
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: target.source}}) '
        f'OPTIONAL MATCH (statement:{statements.STATEMENT_LABEL})-[:HAS_PROCEDURE]->(target) '
        f'RETURN target.uuid AS uuid, $kind AS kind, src.key AS source, '
        f'coalesce(target.procedure, target.description) AS content, '
        f'{{statement_uuid: statement.uuid, kind: target.kind}} AS context '
        f'ORDER BY target.uuid'
    )


async def card_targets_for_hub(
    session_factory: Callable,
    *,
    target_kind: models.CardTargetKind,
    hub_uuid: str,
) -> list[dict]:
    """Loads non-empty source-local targets connected to one local hub."""
    cypher = _card_targets_query(target_kind)
    async with session_factory() as session:
        result = await session.run(cypher, hub_uuid=hub_uuid, kind=target_kind)
        rows = [dict(record) async for record in result]
    targets: list[dict] = []
    for row in rows:
        content = row.get('content')
        if not isinstance(content, str) or not content.strip():
            if target_kind == models.CardTargetKind.PROCEDURE:
                raise ValueError(
                    f'procedure target has no content: {row["uuid"]}'
                )
            continue
        targets.append(row)
    return targets


def merge_card_review_edges_query() -> str:
    return MERGE_CARD_REVIEW_EDGES


async def event_learning_candidates(session_factory: Callable) -> list[dict]:
    """Reads global event hubs with local hubs and source evidence."""
    cypher = (
        f'MATCH (g:{local_event_hubs.hub_label("meta")})'
        f'<-[:ALIGNS_TO]-(h:{local_event_hubs.hub_label()})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: h.source}})\n'
        f'OPTIONAL MATCH (e:{events.EVENT_LABEL})-[:CANONICAL]->(h)\n'
        f'WITH g, h, src, collect(DISTINCT {{uuid: e.uuid, name: e.name, '
        f'description: e.description}}) AS evidence\n'
        f'RETURN g.uuid AS global_uuid, g.canonical_name AS global_name, '
        f'g.description AS global_description, h.uuid AS local_uuid, '
        f'h.canonical_name AS local_name, h.description AS local_description, '
        f'src.key AS source, [item IN evidence WHERE item.uuid IS NOT NULL] AS evidence\n'
        f'ORDER BY g.uuid, h.uuid'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [dict(record) async for record in result]
