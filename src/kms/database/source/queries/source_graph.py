"""Cypher statements for replacing the source structural graph."""

from kms.database.source.queries.source_graph_create import (
    CREATE_SOURCE_STRUCTURE,
)

REPLACE_SOURCE = """
MERGE (source:Source {uuid: $source.uuid})
SET source.key = $source.key,
    source.metadata = $source.metadata,
    source.revision = coalesce(source.revision, 0) + 1
WITH source
CALL (source) {
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
    RETURN size(cards) AS learning_artifacts_deleted
}
WITH source
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            :SourceEntity
        )-[:IN_SOURCE_HUB]->(entity_hub:SourceEntityHub)
    WITH collect(DISTINCT entity_hub) AS hubs
    FOREACH (hub IN hubs | DETACH DELETE hub)
    RETURN count(*) AS identity_hubs_deleted
}
WITH source
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            entity:SourceEntity
        )
    WITH collect(DISTINCT entity) AS entities
    FOREACH (entity IN entities | DETACH DELETE entity)
    RETURN count(*) AS _
}
WITH source
CALL (source) {
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            :SourceStatement
        )-[:IN_SOURCE_HUB]->(statement_hub:SourceStatementHub)
    OPTIONAL MATCH (source)-[:HAS_PAGE]->(:SourcePage)
        -[:CONTAINS_BLOCK]->(:SourceBlock)-[:MEMBER_OF]->(
            :SourceProcedure
        )-[:IN_SOURCE_HUB]->(procedure_hub:SourceProcedureHub)
    WITH collect(DISTINCT statement_hub)
         + collect(DISTINCT procedure_hub) AS hubs
    FOREACH (hub IN hubs | DETACH DELETE hub)
    RETURN count(*) AS _
}
WITH source
OPTIONAL MATCH (source)-[:HAS_PAGE]->(old_page:SourcePage)
OPTIONAL MATCH (old_page)-[:CONTAINS_BLOCK]->(old_block:SourceBlock)
OPTIONAL MATCH (old_block)-[:MEMBER_OF]->(old_statement:SourceStatement)
OPTIONAL MATCH (old_block)-[:MEMBER_OF]->(old_procedure:SourceProcedure)
OPTIONAL MATCH (old_block)-[:CONTAINS_VISUAL_ASSET]->(old_asset:VisualAsset)
OPTIONAL MATCH (old_block)-[:MEMBER_OF]->(old_instruction:Instruction)
WITH source,
     collect(DISTINCT old_page) AS old_pages,
     collect(DISTINCT old_block) AS old_blocks,
     collect(DISTINCT old_statement) AS old_statements,
     collect(DISTINCT old_procedure) AS old_procedures,
     collect(DISTINCT old_asset) AS old_assets,
     collect(DISTINCT old_instruction) AS old_instructions
FOREACH (page IN old_pages | DETACH DELETE page)
FOREACH (block IN old_blocks | DETACH DELETE block)
FOREACH (statement IN old_statements | DETACH DELETE statement)
FOREACH (procedure IN old_procedures | DETACH DELETE procedure)
FOREACH (asset IN old_assets | DETACH DELETE asset)
FOREACH (instruction IN old_instructions | DETACH DELETE instruction)
WITH source
__CREATE_SOURCE_STRUCTURE__
RETURN source.uuid AS uuid, source.revision AS revision
"""

REPLACE_SOURCE = REPLACE_SOURCE.replace(
    '__CREATE_SOURCE_STRUCTURE__', CREATE_SOURCE_STRUCTURE
)

LOCK_SOURCE_WRITE = """
MERGE (source:Source {uuid: $source_uuid})
SET source._source_write_lock = randomUUID()
REMOVE source._source_write_lock
RETURN source.uuid AS uuid
"""

LOCK_EXISTING_SOURCE_WRITE = """
MATCH (source:Source {uuid: $source_uuid})
SET source._source_write_lock = randomUUID()
REMOVE source._source_write_lock
RETURN source.uuid AS uuid
"""

SOURCE_EVIDENCE_REFERENCED = """
MATCH (source:Source {uuid: $source_uuid})
RETURN
    EXISTS {
        MATCH (source)-[:HAS_PAGE]->(:SourcePage)
            -[:CONTAINS_BLOCK]->(block:SourceBlock)
            <-[:CITES_BLOCK]-(:TeachingTurn)
    }
    OR EXISTS {
        MATCH (source)-[:HAS_PAGE]->(:SourcePage)
            -[:CONTAINS_BLOCK]->(:SourceBlock)
            -[:CONTAINS_VISUAL_ASSET]->(:VisualAsset)
            <-[:CITES_IMAGE]-(:TeachingTurn)
    }
    OR EXISTS {
        MATCH (source)-[:HAS_PAGE]->(:SourcePage)
            -[:CONTAINS_BLOCK]->(block:SourceBlock)
            <-[:SUPPORTED_BY]-(:KnowledgeAssertion)
    }
    OR EXISTS {
        MATCH (source)-[:HAS_PAGE]->(:SourcePage)
            -[:CONTAINS_BLOCK]->(block:SourceBlock)
            <-[:SUPPORTED_BY]-(:Argument)
    } AS referenced
"""

SOURCE_HAS_CAPTURE_DIGEST = """
MATCH (source:Source {uuid: $source_uuid})
RETURN source.metadata CONTAINS '\"snapshot_digest\"' AS captured
"""

READ_SOURCE_METADATA = """
MATCH (source:Source {uuid: $source_uuid})
RETURN source.metadata AS metadata
"""

READ_SOURCE_CAPTURE_PAGES = """
MATCH (source:Source {uuid: $source_uuid})
OPTIONAL MATCH (source)-[:HAS_PAGE]->(page:SourcePage)
WITH page ORDER BY page.index
RETURN collect(
    CASE WHEN page IS NULL THEN NULL
    ELSE {index: page.index, markdown: page.markdown} END
) AS pages
"""
