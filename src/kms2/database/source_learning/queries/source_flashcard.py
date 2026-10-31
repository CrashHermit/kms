"""Create generic source flashcards with direct semantic provenance."""

CREATE_SOURCE_FLASHCARDS = """
UNWIND $rows AS row
MATCH (source:Source {uuid: $source_uuid})
CREATE (source)-[:HAS_FLASHCARD]->(card:SourceFlashcard {
    uuid: row.uuid,
    question: row.question,
    answer: row.answer
})
WITH source, card, row
CALL (source, card, row) {
    UNWIND row.source_proposition_uuids AS proposition_uuid
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(:SourceBlock)
          -[:MEMBER_OF]->(proposition:SourceProposition {uuid: proposition_uuid})
    CREATE (card)-[:SUPPORTED_BY]->(proposition)
    RETURN count(DISTINCT proposition) AS propositions_linked
}
CALL (source, card, row) {
    UNWIND row.derived_card_uuids AS parent_uuid
    MATCH (source)-[:HAS_FLASHCARD]->(parent:SourceFlashcard {
        uuid: parent_uuid
    })
    CREATE (card)-[:DERIVED_FROM]->(parent)
    RETURN count(DISTINCT parent) AS parents_linked
}
RETURN count(DISTINCT card) AS persisted
"""
