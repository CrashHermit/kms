"""Create one source-owned flashcard per row, linked to its selected fact."""

CREATE_SOURCE_FLASHCARDS = """
UNWIND $rows AS row
MATCH (source:Source {uuid: $source_uuid})
CREATE (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard {
    uuid: row.uuid,
    question: row.question,
    answer: row.answer
})

WITH source,
     card,
     row
MATCH (source)-[:HAS_LEARNING_FACT]->(learning_fact {uuid: row.learning_fact_uuid})
WHERE learning_fact:SourceEntityLearningFact
   OR learning_fact:SourceEventLearningFact
   OR learning_fact:SourcePredicateLearningFact
   OR learning_fact:SourceTripletLearningFact
CREATE (card)-[:DERIVED_FROM]->(learning_fact)
RETURN count(card) AS persisted
"""
