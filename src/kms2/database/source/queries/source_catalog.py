"""Cypher statements for owned and unowned source catalog queries."""

READ_SOURCES = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
RETURN source.uuid AS uuid, source.key AS key
ORDER BY source.key, source.uuid
"""

READ_UNOWNED_SOURCES = """
MATCH (source:Source)
WHERE NOT EXISTS {
    MATCH (:User)-[:OWNS_SOURCE]->(source)
}
RETURN source.uuid AS uuid, source.key AS key
ORDER BY source.key, source.uuid
"""

ADOPT_SOURCE = """
MATCH (source:Source {uuid: $source_uuid})
SET source._ownership_lock = randomUUID()
REMOVE source._ownership_lock
WITH source
WHERE NOT EXISTS {
    MATCH (:User)-[:OWNS_SOURCE]->(source)
}
MATCH (user:User {uuid: $user_uuid})
CREATE (user)-[:OWNS_SOURCE]->(source)
RETURN source.uuid AS uuid
"""
