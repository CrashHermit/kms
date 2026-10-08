"""Clear one source's generated cards and review history.

Users, decks, source evidence, and semantic records survive regeneration.
"""

CLEAR_SOURCE_LEARNING = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
OPTIONAL MATCH (card)-[:HAS_CARD_REVIEW]->(review:UserCardReview)
OPTIONAL MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
WITH source,
     collect(DISTINCT card) AS cards,
     collect(DISTINCT review) AS reviews,
     collect(DISTINCT event) AS events
FOREACH (event IN events | DETACH DELETE event)
FOREACH (review IN reviews | DETACH DELETE review)
FOREACH (card IN cards | DETACH DELETE card)
RETURN size(cards) AS deleted
"""
