"""Cypher statements for users and their initial persistence."""

CREATE_USER = """
CREATE (user:User {uuid: $user_uuid, name: $user_name})
CREATE (deck:Deck {uuid: $deck_uuid, name: $deck_name})
CREATE (settings:UserSchedulerSettings {
    uuid: $settings_uuid,
    scheduler_json: $scheduler_json
})
CREATE (user)-[:HAS_DECK]->(deck)
CREATE (user)-[:HAS_SCHEDULER_SETTINGS]->(settings)
"""

READ_USERS = """
MATCH (user:User)
RETURN user.uuid AS uuid, user.name AS name
ORDER BY user.name, user.uuid
"""
