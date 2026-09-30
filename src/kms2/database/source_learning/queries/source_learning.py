"""Clear one source's learning snapshot, including shared review history.

User and deck records survive; linked card-review state is deleted with the
source-owned cards.
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
WITH source
OPTIONAL MATCH (source)-[:HAS_LEARNING_FACT]->(learning_fact)
WITH collect(learning_fact) AS learning_facts
FOREACH (learning_fact IN learning_facts | DETACH DELETE learning_fact)
RETURN size(learning_facts) AS deleted
"""
