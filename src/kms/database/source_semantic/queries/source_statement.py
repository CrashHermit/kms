"""Cypher statements for source statement."""

READ_SOURCE_STATEMENTS = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(source_block:SourceBlock)-[:MEMBER_OF]->(
          statement:SourceStatement
      )
MATCH (source)-[:FIRST_BLOCK]->(first_block:SourceBlock)
MATCH block_path = (first_block)-[:NEXT_BLOCK*0..]->(source_block)
WITH statement, source_block, length(block_path) AS block_order
ORDER BY statement.uuid, block_order
WITH statement, collect(source_block.uuid) AS member_block_uuids
RETURN statement.uuid AS uuid,
       member_block_uuids,
       statement.is_exercise AS is_exercise,
       statement.content AS content,
       statement.embedding AS embedding
ORDER BY uuid
"""


READ_SOURCE_STATEMENT_HUB_CANDIDATES = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
          query:SourceStatement
      )
WHERE query.embedding IS NOT NULL
WITH DISTINCT source, query
CALL db.index.vector.queryNodes(
    'source_statement_embedding',
    $candidate_limit,
    query.embedding
)
YIELD node AS candidate, score
WHERE candidate.uuid <> query.uuid AND score >= $minimum_similarity
  AND (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
      :SourceBlock
  )-[:MEMBER_OF]->(candidate)
WITH CASE WHEN query.uuid < candidate.uuid THEN query ELSE candidate END AS left,
     CASE WHEN query.uuid < candidate.uuid THEN candidate ELSE query END AS right,
     score
WITH left, right, max(score) AS score
RETURN left.uuid AS left_uuid,
       left.content AS left_content,
       right.uuid AS right_uuid,
       right.content AS right_content,
       score
ORDER BY left_uuid, right_uuid
"""

REPLACE_SOURCE_STATEMENT_ACCEPTED_EDGES = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        :SourceBlock
    )-[:MEMBER_OF]->(left:SourceStatement)
    MATCH (left)-[old_similarity:SIMILAR_TO]->(right:SourceStatement)
    WHERE (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        :SourceBlock
    )-[:MEMBER_OF]->(right)
    WITH DISTINCT old_similarity
    DELETE old_similarity
    RETURN count(*) AS deleted
}
CALL (source) {
    UNWIND $pairs AS pair
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        :SourceBlock
    )-[:MEMBER_OF]->(left:SourceStatement {uuid: pair.left_uuid})
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        :SourceBlock
    )-[:MEMBER_OF]->(right:SourceStatement {uuid: pair.right_uuid})
    WITH DISTINCT pair, left, right
    MERGE (left)-[similarity:SIMILAR_TO]->(right)
    SET similarity.score = pair.score
    RETURN count(*) AS materialized
}
RETURN materialized
"""

DROP_SOURCE_STATEMENT_HUB_GRAPH = """
CALL gds.graph.drop(
    $graph_name,
    false
)
YIELD graphName
RETURN graphName
"""

DETECT_SOURCE_STATEMENT_COMMUNITIES = """
MATCH (source:Source {uuid: $source_uuid})-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
          node:SourceStatement
      )
WITH DISTINCT source, node
OPTIONAL MATCH (node)-[similarity:SIMILAR_TO]->(target:SourceStatement)
WHERE (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
    :SourceBlock
)-[:MEMBER_OF]->(target)
WITH node, target, similarity
WITH gds.graph.project(
    $graph_name,
    node,
    target,
    {
        sourceNodeLabels: 'SourceStatement',
        targetNodeLabels: 'SourceStatement',
        relationshipType: 'SIMILAR_TO',
        relationshipProperties: {score: similarity.score}
    },
    {undirectedRelationshipTypes: ['*']}
) AS graph
WITH graph.graphName AS projected_graph_name
CALL gds.sllpa.stream(
    projected_graph_name,
    {
        relationshipWeightProperty: 'score',
        maxIterations: $max_iterations,
        minAssociationStrength: $min_association_strength
    }
)
YIELD nodeId, values
WITH gds.util.asNode(nodeId) AS node,
     values.communityIds AS communityIds
UNWIND communityIds AS community_id
RETURN community_id,
       node.uuid AS uuid,
       node.content AS content
"""
REPLACE_SOURCE_STATEMENT_HUBS = """
MATCH (source:Source {uuid: $source_uuid})
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            :SourceStatement
        )-[:IN_SOURCE_HUB]->(old_hub:SourceStatementHub)
    WITH collect(DISTINCT old_hub) AS old_hubs
    FOREACH (hub IN old_hubs | DETACH DELETE hub)
    RETURN count(*) AS deleted
}
UNWIND $hubs AS hub_data
CREATE (hub:SourceStatementHub {
    uuid: hub_data.uuid,
    canonical_name: hub_data.canonical_name,
    aliases: hub_data.aliases,
    description: hub_data.description,
    embedding: hub_data.embedding
})
WITH source, hub, hub_data
UNWIND hub_data.member_uuids AS member_uuid
MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
    :SourceBlock
)-[:MEMBER_OF]->(member:SourceStatement {uuid: member_uuid})
WITH DISTINCT member, hub
CREATE (member)-[:IN_SOURCE_HUB]->(hub)
RETURN count(DISTINCT hub) AS persisted
"""
