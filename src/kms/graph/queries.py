from collections.abc import Callable

from kms.graph import (
    entity_hubs,
    instructions,
    nodes,
    procedures,
    statements,
    triplets,
)

MERGE_SOURCE = (
    f'MERGE (s:{nodes.SOURCE_LABEL} {{uuid: $uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += $props, s.modified_at = $now'
)


def merge_nodes_query(label: str | None) -> str:
    query = (
        f'UNWIND $rows AS row '
        f'MERGE (n:{nodes.NODE_LABEL} {{uuid: row.uuid}}) '
        f'ON CREATE SET n.created_at = $now '
        f'SET n += row, n.modified_at = $now'
    )
    if label:
        query += f' SET n:{label}'
    return query


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

MERGE_PROCEDURES = (
    f'UNWIND $rows AS row '
    f'MERGE (p:{procedures.PROCEDURE_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET p.created_at = $now '
    f'SET p += row, p.modified_at = $now'
)

MERGE_STEPS = (
    f'UNWIND $rows AS row '
    f'MERGE (s:{procedures.STEP_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET s.created_at = $now '
    f'SET s += row, s.modified_at = $now'
)

MERGE_PROCEDURE_MEMBERS = (
    f'UNWIND $pairs AS pair '
    f'MATCH (n:{nodes.NODE_LABEL} {{uuid: pair.node}}), '
    f'(p:{procedures.PROCEDURE_LABEL} {{uuid: pair.procedure}}) '
    f'MERGE (n)-[r:MEMBER_OF]->(p) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_FIRST = (
    f'UNWIND $pairs AS pair '
    f'MATCH (p:{procedures.PROCEDURE_LABEL} {{uuid: pair.procedure}}), '
    f'(s:{procedures.STEP_LABEL} {{uuid: pair.step}}) '
    f'MERGE (p)-[r:FIRST]->(s) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)

MERGE_THEN = (
    f'UNWIND $pairs AS pair '
    f'MATCH (a:{procedures.STEP_LABEL} {{uuid: pair.from}}), '
    f'(b:{procedures.STEP_LABEL} {{uuid: pair.to}}) '
    f'MERGE (a)-[r:THEN]->(b) '
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

MERGE_ENTITY_HUBS = (
    f'UNWIND $rows AS row '
    f'MERGE (h:{entity_hubs.ENTITY_HUB_LABEL} {{uuid: row.uuid}}) '
    f'ON CREATE SET h.created_at = $now '
    f'SET h += row, h.modified_at = $now'
)

MERGE_HAS_SUBJECT = (
    f'UNWIND $pairs AS pair '
    f'MATCH (t:{triplets.TRIPLET_LABEL} {{uuid: pair.triplet}}), '
    f'(h:{entity_hubs.ENTITY_HUB_LABEL} {{uuid: pair.hub}}) '
    f'MERGE (t)-[r:HAS_SUBJECT]->(h) '
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
    f'(h:{entity_hubs.ENTITY_HUB_LABEL} {{uuid: pair.hub}}) '
    f'MERGE (t)-[r:HAS_OBJECT]->(h) '
    f'ON CREATE SET r.created_at = $now '
    f'SET r.modified_at = $now'
)


async def all_entity_spokes(
    session_factory,
) -> list[dict]:
    cypher = (
        f'MATCH (t:{triplets.TRIPLET_LABEL})\n'
        f'OPTIONAL MATCH (src:{nodes.SOURCE_LABEL} {{uuid: t.source}})\n'
        f'RETURN t.uuid AS triplet_uuid, '
        f't.subject AS subject, '
        f't.predicate AS predicate, '
        f't.object AS object, '
        f'src.key AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [
            {
                'triplet_uuid': record['triplet_uuid'],
                'subject': record['subject'],
                'predicate': record['predicate'],
                'object': record['object'],
                'source': record.get('source'),
            }
            async for record in result
        ]


async def vector_search(
    session_factory: Callable,
    *,
    index_name: str,
    query_embedding: list[float],
    top_k: int = 20,
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
        result = await session.run(cypher, **params)
        return [
            {
                **record['properties'],
                'score': record['score'],
            }
            async for record in result
        ]


async def compose_statement(
    statement_uuid: str,
    session_factory: Callable,
) -> dict:
    """Assembles a Statement's full text from its member nodes.

    Walks MEMBER_OF edges, collects node content in order, and
    returns the composited text alongside any picture references.

    Args:
        statement_uuid: UUID of the Statement node.
        session_factory: Async callable returning a Neo4j session.

    Returns:
        A dict with 'text' (str) and 'pictures' (list of node dicts
        with 'index' and 'segment_index').
    """
    cypher = (
        f'MATCH (n:{nodes.NODE_LABEL})-[:MEMBER_OF]->'
        f'(s:{statements.STATEMENT_LABEL} {{uuid: $statement_uuid}})\n'
        f'RETURN n.content AS content, n.type AS type, '
        f'n.index AS index, n.segment_index AS segment_index, '
        f'n.image_path AS image_path\n'
        f'ORDER BY n.index'
    )
    async with session_factory() as session:
        result = await session.run(cypher, statement_uuid=statement_uuid)
        nodes_list = [
            {
                'content': record['content'],
                'type': record['type'],
                'index': record['index'],
                'segment_index': record.get('segment_index'),
                'image_path': record.get('image_path'),
            }
            async for record in result
        ]

    text_parts: list[str] = []
    pictures: list[dict] = []
    for node_data in nodes_list:
        if node_data['type'] == 'image':
            pictures.append(
                {
                    'index': node_data['index'],
                    'segment_index': node_data['segment_index'],
                    'image_path': node_data['image_path'],
                }
            )
        else:
            content = node_data['content'] or ''
            if content:
                text_parts.append(content)

    return {'text': '\n\n'.join(text_parts), 'pictures': pictures}


async def orphan_statements(
    session_factory: Callable,
) -> list[dict]:
    cypher = (
        f'MATCH (s:{statements.STATEMENT_LABEL})\n'
        f'WHERE NOT (s)-[:HAS_PROCEDURE]->(:{procedures.PROCEDURE_LABEL})\n'
        f'MATCH (src:{nodes.SOURCE_LABEL} {{uuid: s.source}})\n'
        f'RETURN s.uuid AS uuid, src.key AS source'
    )
    async with session_factory() as session:
        result = await session.run(cypher)
        return [
            {'uuid': record['uuid'], 'source': record['source']}
            async for record in result
        ]
