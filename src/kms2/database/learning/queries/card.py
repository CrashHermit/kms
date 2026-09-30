"""Cypher statements for assigning cards to decks."""

READ_ASSIGNABLE_CARDS = """
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

READ_DECK_CARDS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard)
RETURN card.uuid AS card_uuid, card.question AS question, card.answer AS answer
ORDER BY card.uuid
"""

READ_DUE_DECK_CARDS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_DECK]->(deck:Deck {uuid: $deck_uuid})
MATCH (deck)-[:HAS_CARD]->(card:SourceFlashcard)
MATCH (user)-[:HAS_CARD_REVIEW]->(review:UserCardReview)<-[:HAS_CARD_REVIEW]-(card)
WHERE review.due_at <= $now
RETURN card.uuid AS card_uuid, card.question AS question,
       card.answer AS answer, review.due_at AS due_at
ORDER BY review.due_at, card.uuid
"""
