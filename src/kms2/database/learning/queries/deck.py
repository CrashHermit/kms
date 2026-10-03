"""Cypher statements for user decks."""

READ_DECKS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck)
RETURN deck.uuid AS uuid,
       deck.name AS name
ORDER BY deck.name, deck.uuid
"""

CREATE_DECK = """
MATCH (user:User {uuid: $user_uuid})
CREATE (user)-[:HAS_DECK]->(deck:Deck {
    uuid: $deck_uuid,
    name: $deck_name
})
RETURN deck.uuid AS uuid,
       deck.name AS name
"""
