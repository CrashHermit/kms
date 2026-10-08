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
MATCH (hub:SourceTripletHub {
    uuid: row.hub_uuid,
    source_uuid: $source_uuid
})
CREATE (card)-[:ABOUT_HUB]->(hub)
WITH source, card, row, hub
CALL (source, card, row, hub) {
    UNWIND row.triplet_uuids AS triplet_uuid
    MATCH (triplet:SourceTriplet {uuid: triplet_uuid})
    MATCH (triplet)-[:IN_SOURCE_HUB]->(hub)
    CREATE (card)-[:SUPPORTED_BY_TRIPLET]->(triplet)
    RETURN count(DISTINCT triplet) AS triplets_linked
}
CALL (source, card, row, hub) {
    UNWIND row.source_fact_uuids AS source_fact_uuid
    MATCH (source)-[:FIRST_BLOCK]->(first:SourceBlock)
    MATCH (first)-[:NEXT_BLOCK*0..]->(source_block:SourceBlock)
    MATCH (source_block)<-[:HAS_SOURCE_BLOCK]-(:SourceFactTarget)
        <-[:HAS_TARGET]-(fact:SourceFact {uuid: source_fact_uuid})
    MATCH (fact)-[:HAS_TRIPLET]->(:SourceTriplet)-[:IN_SOURCE_HUB]->(hub)
    CREATE (card)-[:SUPPORTED_BY]->(fact)
    RETURN count(DISTINCT fact) AS facts_linked
}
CALL (source, card, row, hub) {
    UNWIND row.derived_card_uuids AS parent_uuid
    MATCH (source)-[:HAS_FLASHCARD]->(parent:SourceFlashcard {
        uuid: parent_uuid
    })
    MATCH (parent)-[:ABOUT_HUB]->(hub)
    CREATE (card)-[:DERIVED_FROM]->(parent)
    RETURN count(DISTINCT parent) AS parents_linked
}
RETURN count(DISTINCT card) AS persisted
"""
