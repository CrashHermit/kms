"""Cypher statements for review scheduling and history."""

READ_DUE_DECK_CARDS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard)
MATCH (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)<-[:HAS_CARD_REVIEW]-(card)
WHERE review.due_at <= $now
RETURN card.uuid AS card_uuid,
       card.question AS question,
       card.answer AS answer,
       review.due_at AS due_at
ORDER BY review.due_at, card.uuid
"""

READ_REVIEW_SNAPSHOT = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_SCHEDULER_SETTINGS]->
      (settings:UserSchedulerSettings)
MATCH (user)-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard {uuid: $card_uuid})
MATCH (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)<-[:HAS_CARD_REVIEW]-(card)
RETURN card.uuid AS card_uuid,
       settings.scheduler_json AS scheduler_json,
       review.fsrs_card_json AS fsrs_card_json,
       review.review_count AS review_count
"""

RECORD_REVIEW = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_SCHEDULER_SETTINGS]->
      (settings:UserSchedulerSettings)
MATCH (user)-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard {uuid: $card_uuid})
MATCH (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)<-[:HAS_CARD_REVIEW]-(card)
WITH settings,
     card,
     review,
     review.review_count + 1 AS review_index
SET review.fsrs_card_json = $fsrs_card_json,
    review.due_at = $due_at,
    review.review_count = review_index
CREATE (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent {
    uuid: $event_uuid,
    review_index: review_index,
    scheduler_json: $scheduler_json,
    fsrs_card_before_json: $fsrs_card_before_json,
    fsrs_card_after_json: $fsrs_card_after_json,
    fsrs_review_log_json: $fsrs_review_log_json
})
RETURN card.uuid AS card_uuid,
       review.due_at AS due_at,
       review.review_count AS review_count,
       review_index
"""

READ_REVIEW_EVENTS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_CARD_REVIEW]->
      (review:UserCardReview)<-[:HAS_CARD_REVIEW]-
      (card:SourceFlashcard {uuid: $card_uuid})
MATCH (review)-[:HAS_REVIEW_EVENT]->(event:ReviewEvent)
RETURN event.uuid AS uuid,
       event.review_index AS review_index,
       event.scheduler_json AS scheduler_json,
       event.fsrs_card_before_json AS fsrs_card_before_json,
       event.fsrs_card_after_json AS fsrs_card_after_json,
       event.fsrs_review_log_json AS fsrs_review_log_json
ORDER BY event.review_index
"""
