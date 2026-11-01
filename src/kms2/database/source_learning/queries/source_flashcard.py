"""Create generic source flashcards with direct block provenance."""

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
    UNWIND row.source_block_uuids AS block_uuid
    MATCH (source)-[:HAS_PAGE]->(:SourcePage)-[:CONTAINS_BLOCK]->(
        block:SourceBlock {uuid: block_uuid}
    )
    CREATE (card)-[:SUPPORTED_BY]->(block)
    RETURN count(DISTINCT block) AS blocks_linked
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
