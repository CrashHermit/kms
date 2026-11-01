"""Cypher statements for owned immutable source snapshots."""

from kms2.database.source.queries.source_graph import (
    CREATE_SOURCE_STRUCTURE,
)

LOCK_SNAPSHOT_OWNER = """
MATCH (user:User {uuid: $user_uuid})
SET user._source_snapshot_lock = randomUUID()
REMOVE user._source_snapshot_lock
RETURN user.uuid AS uuid
"""

READ_OWNED_SNAPSHOT = """
MATCH (:User {uuid: $user_uuid})-[:OWNS_SOURCE]->(source:Source {uuid: $source_uuid})
RETURN source.uuid AS uuid,
       source.key AS key,
       source.metadata AS metadata,
       coalesce(source.revision, 0) AS revision
"""

CREATE_OWNED_SNAPSHOT = """
MATCH (user:User {uuid: $user_uuid})
CREATE (source:Source {
    uuid: $source.uuid,
    key: $source.key,
    metadata: $source.metadata,
    revision: 0
})
CREATE (user)-[:OWNS_SOURCE]->(source)
WITH source
__CREATE_SOURCE_STRUCTURE__
RETURN source.uuid AS uuid
"""

CREATE_OWNED_SNAPSHOT = CREATE_OWNED_SNAPSHOT.replace(
    '__CREATE_SOURCE_STRUCTURE__', CREATE_SOURCE_STRUCTURE
)
