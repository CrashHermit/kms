"""Session-authorized reads for cited source and proposition evidence."""

READ_TEACHING_CITATIONS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
CALL (user, session) {
    UNWIND $references AS reference
    CALL (user, session, reference) {
        WITH user, session, reference
        WHERE reference.kind = 'block'
        MATCH (session)-[:HAS_TURN]->(:TeachingTurn)-[:CITES_BLOCK]->
              (block:SourceBlock {uuid: reference.uuid})
        MATCH (user)-[:OWNS_SOURCE]->(source:Source)-[:HAS_PAGE]->
              (page:SourcePage)-[:CONTAINS_BLOCK]->(block)
        OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->(image:VisualAsset)
        WITH block, source,
             collect(DISTINCT page.index) AS page_indexes,
             collect(DISTINCT image.uuid) AS image_uuids
        RETURN {
            kind: 'block',
            uuid: block.uuid,
            content: block.content,
            source_uuid: source.uuid,
            source_key: source.key,
            metadata: source.metadata,
            page_indexes: page_indexes,
            image_uuids: image_uuids
        } AS citation
        UNION
        WITH user, session, reference
        WHERE reference.kind = 'image'
        MATCH (session)-[:HAS_TURN]->(:TeachingTurn)-[:CITES_IMAGE]->
              (image:VisualAsset {uuid: reference.uuid})
        MATCH (user)-[:OWNS_SOURCE]->(source:Source)-[:HAS_PAGE]->
              (page:SourcePage)-[:CONTAINS_BLOCK]->(block:SourceBlock)
              -[:CONTAINS_VISUAL_ASSET]->(image)
        WITH image, block, source, collect(DISTINCT page.index) AS page_indexes
        RETURN {
            kind: 'image',
            uuid: image.uuid,
            content: block.content,
            source_uuid: source.uuid,
            source_key: source.key,
            metadata: source.metadata,
            page_indexes: page_indexes,
            image_uuids: [image.uuid]
        } AS citation
        UNION
        WITH user, session, reference
        WHERE reference.kind = 'proposition'
        MATCH (session)-[:HAS_TURN]->(:TeachingTurn)-[:CITES_PROPOSITION]->
              (proposition:Proposition {uuid: reference.uuid})
        MATCH (user)-[:HAS_PROPOSITION]->(proposition)
        RETURN {
            kind: 'proposition',
            uuid: proposition.uuid,
            content: proposition.content,
            source_uuid: null,
            source_key: null,
            url: null,
            title: null,
            retrieved_at: null,
            page_indexes: [],
            image_uuids: []
        } AS citation
    }
    RETURN collect(citation) AS citations
}
WITH user, session, citations
WHERE size(citations) = size($references)
RETURN citations
"""

READ_TEACHING_CITATION_MEDIA = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: $session_uuid})
MATCH (session)-[:HAS_TURN]->(:TeachingTurn)-[:CITES_IMAGE]->
      (image:VisualAsset {uuid: $asset_uuid})
MATCH (user)-[:OWNS_SOURCE]->(:Source)-[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(:SourceBlock)-[:CONTAINS_VISUAL_ASSET]->(image)
RETURN DISTINCT image.path AS path
"""
