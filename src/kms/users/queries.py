"""Cypher statements for users."""

CREATE_USER = """
CREATE (user:User {
    uuid: $user_uuid,
    name: $user_name,
    created_at: $created_at,
    updated_at: $updated_at,
    invalidated_at: $invalidated_at
})
"""


READ_USER = """
MATCH (user:User {uuid: $user_uuid})
RETURN user.uuid AS uuid,
       user.name AS name,
       user.created_at AS created_at,
       user.updated_at AS updated_at,
       user.invalidated_at AS invalidated_at
"""


READ_USERS = """
MATCH (user:User)
RETURN user.uuid AS uuid,
       user.name AS name,
       user.created_at AS created_at,
       user.updated_at AS updated_at,
       user.invalidated_at AS invalidated_at
ORDER BY user.name, user.uuid
"""
