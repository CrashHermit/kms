"""Cypher statements for decks, source flashcards, and user reviews."""

LIST_DECKS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck)
RETURN deck.uuid AS uuid, deck.name AS name
ORDER BY deck.name, deck.uuid
"""

CREATE_DECK = """
MATCH (user:User {uuid: $user_uuid})
CREATE (user)-[:HAS_DECK]->(deck:Deck {
    uuid: $deck_uuid, name: $deck_name
})
RETURN deck.uuid AS uuid, deck.name AS name
"""

LIST_ASSIGNABLE_CARDS = """
MATCH (user:User {uuid: $user_uuid})-[:OWNS_SOURCE]->
      (source:Source)-[:HAS_FLASHCARD]->(card:SourceFlashcard)
RETURN card.uuid AS card_uuid, card.question AS question, card.answer AS answer
ORDER BY card.uuid
"""

ADD_CARD_TO_DECK = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (user)-[:OWNS_SOURCE]->(source:Source)-[:HAS_FLASHCARD]->
      (card:SourceFlashcard {uuid: $card_uuid})
MERGE (deck)-[:HAS_CARD]->(card)
MERGE (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)<-[:HAS_CARD_REVIEW]-(card)
ON CREATE SET review.uuid = $review_uuid,
              review.fsrs_card_json = $fsrs_card_json,
              review.due_at = $due_at,
              review.review_count = $review_count
RETURN card.uuid AS card_uuid, card.question AS question, card.answer AS answer
"""

REMOVE_CARD_FROM_DECK = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[membership:HAS_CARD]->(card:SourceFlashcard {uuid: $card_uuid})
DELETE membership
"""

LIST_DECK_CARDS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard)
RETURN card.uuid AS card_uuid, card.question AS question, card.answer AS answer
ORDER BY card.uuid
"""

LIST_DUE_DECK_CARDS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard)
MATCH (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)<-[:HAS_CARD_REVIEW]-(card)
WHERE review.due_at <= $now
RETURN card.uuid AS card_uuid, card.question AS question,
       card.answer AS answer, review.due_at AS due_at
ORDER BY review.due_at, card.uuid
"""

LOAD_REVIEW_SNAPSHOT = """
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
WITH settings, card, review, review.review_count + 1 AS review_index
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

LIST_REVIEW_EVENTS = """
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


__all__ = [
    'ADD_CARD_TO_DECK',
    'CREATE_DECK',
    'LIST_ASSIGNABLE_CARDS',
    'LIST_DECK_CARDS',
    'LIST_DUE_DECK_CARDS',
    'LIST_REVIEW_EVENTS',
    'LIST_DECKS',
    'LOAD_REVIEW_SNAPSHOT',
    'RECORD_REVIEW',
    'REMOVE_CARD_FROM_DECK',
]
