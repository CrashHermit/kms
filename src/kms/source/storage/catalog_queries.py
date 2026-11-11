"""Cypher statements for owned and unowned source catalog queries."""

READ_SOURCES = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source)
RETURN source.uuid AS uuid,
       source.key AS key,
       source.created_at AS created_at,
       source.updated_at AS updated_at,
       source.invalidated_at AS invalidated_at,
       coalesce(source.revision, 0) AS revision
ORDER BY source.key, source.uuid
"""

READ_UNOWNED_SOURCES = """
MATCH (source:Source)
WHERE NOT EXISTS {
    MATCH (:User)-[:OWNS_SOURCE]->(source)
}
RETURN source.uuid AS uuid,
       source.key AS key,
       source.created_at AS created_at,
       source.updated_at AS updated_at,
       source.invalidated_at AS invalidated_at,
       coalesce(source.revision, 0) AS revision
ORDER BY source.key, source.uuid
"""

READ_UNOWNED_SOURCE_SCHEMA = """
MATCH (source:Source {uuid: $source_uuid})
WHERE NOT EXISTS {
    MATCH (:User)-[:OWNS_SOURCE]->(source)
}
RETURN source.identity_schema AS identity_schema
"""

ADOPT_SOURCE = """
MATCH (source:Source {uuid: $source_uuid, identity_schema: 'source-content-v3'})
SET source._ownership_lock = randomUUID()
REMOVE source._ownership_lock
WITH source
WHERE NOT EXISTS {
    MATCH (:User)-[:OWNS_SOURCE]->(source)
}
MATCH (user:User {uuid: $user_uuid})
CREATE (user)-[:OWNS_SOURCE {
    created_at: $edge_created_at, updated_at: $edge_created_at
}]->(source)
RETURN source.uuid AS uuid
"""
