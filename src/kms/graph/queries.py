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

MERGE_SOURCE = (
    f'MERGE (s:{SOURCE_LABEL} {{uuid: $uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += $props, s.modified_at = $now'
)


def merge_nodes_query(label: str | None) -> str:
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

DELETE_CANONICAL_LAYER = (
    f'MATCH (:Entity)-[r:CANONICAL]->() DELETE r '
    f'WITH 1 AS done '
    f'MATCH (:Predicate)-[r:CANONICAL]->() DELETE r '
    f'WITH 1 AS done '
    f'MATCH (th:{TRIPLET_HUB_LABEL}) '
    f'OPTIONAL MATCH (th)-[r1]->() DELETE r1 '
    f'WITH th, 1 AS done '
    f'DETACH DELETE th '
    f'WITH 1 AS done '
    f'MATCH (fh:{FACT_HUB_LABEL}) '
    f'OPTIONAL MATCH (fh)-[r2]->() DELETE r2 '
    f'WITH fh, 1 AS done '
    f'DETACH DELETE fh '
    f'WITH 1 AS done '
    f'MATCH (eh:{ENTITY_HUB_LABEL}) '
    f'OPTIONAL MATCH (eh)-[r3]->() DELETE r3 '
    f'WITH eh, 1 AS done '
    f'DETACH DELETE eh '
    f'WITH 1 AS done '
    f'MATCH (ph:{PREDICATE_HUB_LABEL}) '
    f'OPTIONAL MATCH (ph)-[r4]->() DELETE r4 '
    f'WITH ph, 1 AS done '
    f'DETACH DELETE ph '
    f'WITH 1 AS done '
    f'MATCH (d:{DEFINITION_LABEL}) DETACH DELETE d'
)


async def relation_types(
    session_factory: Callable,
    *,
    source: str | None = None,
) -> list[str]:
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


async def all_entity_spokes(
    session_factory: Callable,
) -> list[dict]:
    cypher = (
        f'MATCH (e:{ENTITY_LABEL})\n'
        f'WHERE e.embedding IS NOT NULL\n'
        f'RETURN e.uuid AS uuid, e.name AS name, '
        f'e.description AS description, e.embedding AS embedding, '
        f'e.source AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [
            {
                'uuid': record['uuid'],
                'name': record['name'],
                'description': record.get('description'),
                'embedding': record['embedding'],
                'source': record.get('source'),
            }
            async for record in result
        ]


async def all_predicate_spokes(
    session_factory: Callable,
) -> list[dict]:
    cypher = (
        f'MATCH (p:{PREDICATE_LABEL})\n'
        f'WHERE p.embedding IS NOT NULL\n'
        f'RETURN p.uuid AS uuid, p.predicate AS predicate, '
        f'p.description AS description, p.embedding AS embedding, '
        f'p.source AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [
            {
                'uuid': record['uuid'],
                'predicate': record['predicate'],
                'description': record.get('description'),
                'embedding': record['embedding'],
                'source': record.get('source'),
            }
            async for record in result
        ]


async def delete_canonical_layer(session_factory: Callable) -> None:
    async with session_factory() as session:
        await session.run(DELETE_CANONICAL_LAYER)


async def all_canonical_triplets(
    session_factory: Callable,
) -> list[dict]:
    from kms.graph.entity_hubs import ENTITY_HUB_LABEL as EH
    from kms.graph.predicate_hubs import PREDICATE_HUB_LABEL as PH
    from kms.graph.triplets import TRIPLET_LABEL as TL

    cypher = (
        f'MATCH (t:{TL})\n'
        f'MATCH (t)-[:HAS_SUBJECT]->(es:Entity)\n'
        f'  -[:CANONICAL]->(sh:{EH})\n'
        f'MATCH (t)-[:HAS_PREDICATE]->(pp:Predicate)\n'
        f'  -[:CANONICAL]->(ph:{PH})\n'
        f'MATCH (t)-[:HAS_OBJECT]->(eo:Entity)\n'
        f'  -[:CANONICAL]->(oh:{EH})\n'
        f'OPTIONAL MATCH (sh)-[:HAS_DEFINITION]->(sd:Definition)\n'
        f'OPTIONAL MATCH (ph)-[:HAS_DEFINITION]->(pd:Definition)\n'
        f'OPTIONAL MATCH (oh)-[:HAS_DEFINITION]->(od:Definition)\n'
        f'RETURN t.uuid AS triplet_uuid,\n'
        f'  t.source AS source,\n'
        f'  sh.uuid AS subj_hub, sh.display_name AS subj_name,\n'
        f'  sd.text AS subj_def,\n'
        f'  ph.uuid AS pred_hub, ph.display_name AS pred_name,\n'
        f'  pd.text AS pred_def,\n'
        f'  oh.uuid AS obj_hub, oh.display_name AS obj_name,\n'
        f'  od.text AS obj_def'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [
            {
                'triplet_uuid': record['triplet_uuid'],
                'source': record.get('source'),
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


async def canonical_hub_triplets(
    session_factory: Callable,
    *,
    source: str,
) -> list[dict]:
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
        result = await session.run(cypher, source=source_uuid(source))
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
    cypher = (
        'CALL db.index.vector.queryNodes(\n'
        '  "community_summary", $k, $query_embedding\n'
        ') YIELD node, score\n'
        'WHERE node.source = $source\n'
        'RETURN node.uuid AS uuid, '
        'node.summary_text AS summary_text, '
        'node.summary_embedding AS summary_embedding, '
        'score\n'
        'ORDER BY score DESC\n'
        'LIMIT $top_k'
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
    hub_label = ENTITY_HUB_LABEL if kind == 'entity' else PREDICATE_HUB_LABEL
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
    cypher = (
        'CALL db.index.vector.queryNodes(\n'
        '  "node_content", $k, $query_embedding\n'
        ') YIELD node, score\n'
        'WHERE node.source = $source\n'
        'RETURN node.uuid AS uuid, '
        'node.content AS content, '
        'node.type AS type, '
        'node.index AS index, '
        'node.segment_index AS segment_index, '
        'score\n'
        'ORDER BY score DESC\n'
        'LIMIT $top_k'
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

