"""Cypher statements for owned knowledge storage and retrieval."""

LOCK_KNOWLEDGE_OWNER = """
MATCH (user:User {uuid: $user_uuid})
SET user._knowledge_lock = randomUUID()
REMOVE user._knowledge_lock
RETURN user.uuid AS uuid
"""

REQUIRE_OWNED_SOURCE_UNIT = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source {uuid: $source_uuid})
MATCH (source)-[:HAS_UNIT]->(:SourceUnit {uuid: $target_unit_uuid})
RETURN count(*) AS count
"""

READ_OWNED_ARGUMENTS = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_ARGUMENT]->(argument:Argument)
MATCH (argument)-[:HAS_PREMISE]->(premise:Proposition)
MATCH (argument)-[:CONCLUDES]->(conclusion:Proposition)
RETURN argument {
    .uuid, .created_at, .updated_at, .invalidated_at
} AS argument, collect(premise.uuid) AS premise_uuids,
conclusion.uuid AS conclusion_uuid
ORDER BY argument.uuid
"""


SAVE_PROPOSITIONS = """
MATCH (user:User {uuid: $user_uuid})
UNWIND $propositions AS row
CREATE (proposition:Proposition {
    uuid: row.uuid, name: row.name, statement: row.statement,
    embedding: row.embedding, embedding_model: $embedding_model,
    created_at: row.created_at,
    updated_at: row.updated_at, invalidated_at: row.invalidated_at
})
CREATE (user)-[:OWNS_PROPOSITION {
    created_at: $edge_created_at, updated_at: $edge_created_at
}]->(proposition)
RETURN count(proposition) AS count
"""

SAVE_ARGUMENTS = """
MATCH (user:User {uuid: $user_uuid})
UNWIND $arguments AS row
MATCH (user)-[:OWNS_PROPOSITION]->(conclusion:Proposition {uuid: row.conclusion_uuid})
CALL (user, row) {
    UNWIND row.premise_uuids AS premise_uuid
    MATCH (user)-[:OWNS_PROPOSITION]->(premise:Proposition {uuid: premise_uuid})
    RETURN collect(DISTINCT premise) AS premises
}
WITH user, row, conclusion, premises
WHERE size(premises) = size(row.premise_uuids)
CREATE (argument:Argument {
    uuid: row.uuid, created_at: row.created_at,
    updated_at: row.updated_at, invalidated_at: row.invalidated_at,
    origin: row.origin
})
CREATE (user)-[:OWNS_ARGUMENT {
    created_at: $edge_created_at, updated_at: $edge_created_at
}]->(argument)
FOREACH (premise IN premises |
    CREATE (argument)-[:HAS_PREMISE {
        created_at: $edge_created_at, updated_at: $edge_created_at
    }]->(premise)
)
CREATE (argument)-[:CONCLUDES {
    created_at: $edge_created_at, updated_at: $edge_created_at
}]->(conclusion)
RETURN count(argument) AS count
"""

ATTACH_SOURCE_PROVENANCE = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source {uuid: $source_uuid})
UNWIND $proposition_uuids AS uuid
MATCH (user)-[:OWNS_PROPOSITION]->(proposition:Proposition {uuid: uuid})
MERGE (proposition)-[evidence:FROM_SOURCE]->(source)
ON CREATE SET evidence.created_at = $edge_created_at,
              evidence.updated_at = $edge_created_at
RETURN count(proposition) AS count
"""

ATTACH_ARGUMENT_PROVENANCE = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source {uuid: $source_uuid})
UNWIND $argument_uuids AS uuid
MATCH (user)-[:OWNS_ARGUMENT]->(argument:Argument {uuid: uuid})
MERGE (argument)-[evidence:FROM_SOURCE]->(source)
ON CREATE SET evidence.created_at = $edge_created_at,
              evidence.updated_at = $edge_created_at
RETURN count(argument) AS count
"""

SEARCH_PROPOSITIONS = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(proposition:Proposition)
WITH collect(CASE WHEN proposition.embedding_model IS NULL
                       OR proposition.embedding_model <> $embedding_model
                  THEN proposition.uuid END) AS incompatible_uuids,
     collect(proposition) AS owned
CALL (incompatible_uuids, owned) {
    UNWIND CASE WHEN size(incompatible_uuids) = 0 THEN owned ELSE [] END AS proposition
    WITH proposition,
         vector.similarity.cosine(proposition.embedding, $embedding) AS score
    WHERE score IS NOT NULL
    RETURN collect({
        proposition: proposition {
            .uuid, .name, .statement, .created_at, .updated_at, .invalidated_at
        },
        score: score
    }) AS ranked
}
RETURN incompatible_uuids, ranked
"""

READ_MATCHED_PROPOSITIONS = """
UNWIND range(0, size($matched_uuids) - 1) AS position
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(proposition:Proposition)
WHERE proposition.uuid = $matched_uuids[position]
RETURN proposition {
    .uuid, .name, .statement, .created_at, .updated_at, .invalidated_at
} AS proposition
ORDER BY position
"""

READ_CONTEXT = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(matched:Proposition)
WHERE matched.uuid IN $matched_uuids
MATCH (:User {uuid: $user_uuid})-[:OWNS_ARGUMENT]->(argument:Argument)
MATCH (argument)-[:HAS_PREMISE|CONCLUDES]-(matched)
WITH DISTINCT argument
ORDER BY argument.uuid
LIMIT $argument_limit
MATCH (:User {uuid: $user_uuid})-[:OWNS_PROPOSITION]->(proposition:Proposition)
WHERE (argument)-[:HAS_PREMISE|CONCLUDES]-(proposition)
WITH argument, collect(DISTINCT proposition {
    .uuid, .name, .statement, .created_at, .updated_at, .invalidated_at
}) AS propositions
RETURN argument {
    .uuid, .created_at, .updated_at, .invalidated_at,
    premise_uuids: [(argument)-[:HAS_PREMISE]->(premise) | premise.uuid],
    conclusion_uuid: head([(argument)-[:CONCLUDES]->(conclusion) | conclusion.uuid])
} AS argument, propositions
ORDER BY argument.uuid
"""
