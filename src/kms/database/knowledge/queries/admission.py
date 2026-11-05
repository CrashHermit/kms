"""Cypher statements for atomic, user-owned knowledge admission."""

CREATE_KNOWLEDGE_PROPOSITION = """
MATCH (user:User {uuid: $user_uuid})
CREATE (proposition:Proposition {
    uuid: $uuid,
    content: $content,
    embedding: $embedding,
    created_at: $created_at
})
CREATE (user)-[:HAS_PROPOSITION]->(proposition)
RETURN proposition.uuid AS uuid
"""

CREATE_KNOWLEDGE_ARGUMENT = """
MATCH (user:User {uuid: $user_uuid})
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition {
    uuid: $conclusion_uuid
})
UNWIND $premise_uuids AS premise_uuid
MATCH (user)-[:HAS_PROPOSITION]->(premise:Proposition {uuid: premise_uuid})
WITH user, conclusion, collect(DISTINCT premise) AS premises
CREATE (argument:Argument {
    uuid: $uuid,
    explanation: $explanation,
    embedding: $embedding,
    created_at: $created_at
})
CREATE (user)-[:HAS_ARGUMENT]->(argument)
CREATE (argument)-[:PROVES]->(conclusion)
FOREACH (premise IN premises |
    CREATE (premise)-[:PREMISE_OF]->(argument)
)
RETURN argument.uuid AS uuid
"""

CREATE_KNOWLEDGE_ARGUMENT_WITHOUT_PREMISES = """
MATCH (user:User {uuid: $user_uuid})
MATCH (user)-[:HAS_PROPOSITION]->(conclusion:Proposition {
    uuid: $conclusion_uuid
})
CREATE (argument:Argument {
    uuid: $uuid,
    explanation: $explanation,
    embedding: $embedding,
    created_at: $created_at
})
CREATE (user)-[:HAS_ARGUMENT]->(argument)
CREATE (argument)-[:PROVES]->(conclusion)
RETURN argument.uuid AS uuid
"""

LINK_KNOWLEDGE_ARGUMENT_SUPPORT = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock {
          uuid: $block_uuid
      })
MATCH (user)-[:HAS_ARGUMENT]->(argument:Argument {uuid: $argument_uuid})
MERGE (argument)-[:SUPPORTED_BY]->(block)
RETURN block.uuid AS uuid
"""

LINK_EXISTING_KNOWLEDGE_ARGUMENT = """
MATCH (:User {uuid: $user_uuid})-[:HAS_ARGUMENT]->(argument:Argument {
    uuid: $argument_uuid
})
RETURN argument.uuid AS uuid
"""

CREATE_KNOWLEDGE_ASSERTION = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_PROPOSITION]->(
    proposition:Proposition {uuid: $proposition_uuid}
)
MATCH (user)-[:HAS_TEACHING_SESSION]->(:TeachingSession)-[:HAS_TURN]->(
    turn:TeachingTurn {uuid: $turn_uuid}
)
CREATE (assertion:KnowledgeAssertion {
    uuid: $uuid,
    created_at: $created_at
})
CREATE (turn)-[:HAS_KNOWLEDGE_ASSERTION]->(assertion)
CREATE (assertion)-[:ASSERTS]->(proposition)
RETURN assertion.uuid AS uuid
"""

LINK_KNOWLEDGE_ASSERTION_SUPPORT = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(:Source)
      -[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock {
          uuid: $block_uuid
      })
MATCH (turn:TeachingTurn {uuid: $turn_uuid})-[:HAS_KNOWLEDGE_ASSERTION]->(
    assertion:KnowledgeAssertion {uuid: $assertion_uuid}
)
MERGE (assertion)-[:SUPPORTED_BY]->(block)
RETURN block.uuid AS uuid
"""

LINK_KNOWLEDGE_ASSERTION_ARGUMENT = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_ARGUMENT]->(
    argument:Argument {uuid: $argument_uuid}
)
MATCH (turn:TeachingTurn {uuid: $turn_uuid})-[:HAS_KNOWLEDGE_ASSERTION]->(
    assertion:KnowledgeAssertion {uuid: $assertion_uuid}
)
MERGE (assertion)-[:DERIVED_BY]->(argument)
RETURN argument.uuid AS uuid
"""

INCREMENT_KNOWLEDGE_REVISION = """
MATCH (user:User {uuid: $user_uuid})
SET user.knowledge_revision = coalesce(user.knowledge_revision, 0) + 1
RETURN user.knowledge_revision AS revision
"""
