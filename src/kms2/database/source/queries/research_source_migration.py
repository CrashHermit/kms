"""Cypher statements for one user's legacy research-source migration."""

READ_USER_RESEARCH_CITATIONS = """
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession)-[:HAS_TURN]->
      (turn:TeachingTurn)-[:CITES_EXTERNAL]->(evidence:ExternalEvidence)
WITH evidence, collect(DISTINCT turn.uuid) AS turn_uuids,
     collect(DISTINCT session.uuid) AS session_uuids
RETURN evidence.uuid AS uuid,
       evidence.kind AS kind,
       evidence.url AS url,
       evidence.resolved_url AS resolved_url,
       evidence.title AS title,
       evidence.retrieved_at AS retrieved_at,
       evidence.page_index AS page_index,
       evidence.line_start AS line_start,
       evidence.line_end AS line_end,
       evidence.raw_content AS raw_content,
       evidence.content AS content,
       evidence.media_path AS media_path,
       evidence.media_type AS media_type,
       evidence.original_media_path AS original_media_path,
       evidence.original_media_type AS original_media_type,
       turn_uuids,
       session_uuids
ORDER BY evidence.uuid
"""

LOCK_RESEARCH_MIGRATION_USER = """
MATCH (user:User {uuid: $user_uuid})
SET user._research_migration_lock = randomUUID()
REMOVE user._research_migration_lock
RETURN user.uuid AS uuid
"""

LOCK_LEGACY_RESEARCH_EVIDENCE = """
UNWIND $evidence_uuids AS evidence_uuid
MATCH (evidence:ExternalEvidence {uuid: evidence_uuid})
WITH evidence_uuid, evidence
ORDER BY evidence_uuid
SET evidence._research_migration_lock = randomUUID()
REMOVE evidence._research_migration_lock
RETURN collect(evidence.uuid) AS uuids
"""

LOCK_RESEARCH_MIGRATION_SOURCES = """
UNWIND $source_uuids AS source_uuid
OPTIONAL MATCH (source:Source {uuid: source_uuid})
WITH source_uuid, source
ORDER BY source_uuid
FOREACH (_ IN CASE WHEN source IS NULL THEN [] ELSE [1] END |
    SET source._source_write_lock = randomUUID()
    REMOVE source._source_write_lock
)
RETURN collect(source.uuid) AS uuids
"""

CHECK_RESEARCH_MIGRATION_SOURCE_REVISIONS = """
UNWIND $sources AS expected
MATCH (source:Source {uuid: expected.uuid})
WHERE coalesce(source.revision, 0) = expected.revision
RETURN collect(source.uuid) AS uuids
"""

LOCK_RESEARCH_MIGRATION_SESSIONS = """
UNWIND $session_uuids AS session_uuid
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (session:TeachingSession {uuid: session_uuid})
WITH session_uuid, session
ORDER BY session_uuid
SET session._teaching_write_lock = randomUUID()
REMOVE session._teaching_write_lock
RETURN collect(session.uuid) AS uuids
"""

REWRITE_RESEARCH_CITATIONS = """
UNWIND $rewrites AS rewrite
MATCH (user:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (:TeachingSession)-[:HAS_TURN]->
      (turn:TeachingTurn {uuid: rewrite.turn_uuid})-
      [citation:CITES_EXTERNAL]->
      (evidence:ExternalEvidence {uuid: rewrite.evidence_uuid})
MATCH (source:Source {uuid: rewrite.source_uuid})
      -[:HAS_PAGE]->(:SourcePage)
      -[:CONTAINS_BLOCK]->(block:SourceBlock {uuid: rewrite.block_uuid})
OPTIONAL MATCH (block)-[:CONTAINS_VISUAL_ASSET]->
               (asset:VisualAsset {uuid: rewrite.asset_uuid})
WHERE rewrite.asset_uuid IS NULL OR asset IS NOT NULL
DELETE citation
FOREACH (_ IN CASE WHEN rewrite.kind = 'image'
                        AND rewrite.asset_uuid IS NOT NULL OR
                        rewrite.kind = 'pdf_page'
                        AND rewrite.asset_uuid IS NOT NULL
                   THEN [1] ELSE [] END |
    MERGE (turn)-[:CITES_IMAGE]->(asset)
)
FOREACH (_ IN CASE WHEN rewrite.kind = 'web_passage' OR
                        rewrite.kind = 'pdf_page' OR
                        rewrite.kind = 'image'
                        AND rewrite.asset_uuid IS NULL
                   THEN [1] ELSE [] END |
    MERGE (turn)-[:CITES_BLOCK]->(block)
)
RETURN count(*) AS migrated
"""

DELETE_UNREFERENCED_LEGACY_RESEARCH_EVIDENCE = """
UNWIND $evidence_uuids AS evidence_uuid
MATCH (evidence:ExternalEvidence {uuid: evidence_uuid})
WHERE NOT EXISTS { MATCH (:TeachingTurn)-[:CITES_EXTERNAL]->(evidence) }
  AND NOT EXISTS { MATCH (evidence)--() }
DELETE evidence
RETURN count(*) AS deleted
"""

COUNT_LEGACY_RESEARCH_CITATIONS = """
MATCH ()-[citation:CITES_EXTERNAL]->()
RETURN count(citation) AS count
"""

DELETE_ORPHANED_LEGACY_RESEARCH_EVIDENCE = """
MATCH (evidence:ExternalEvidence)
WHERE NOT EXISTS { MATCH (evidence)--() }
DELETE evidence
RETURN count(*) AS deleted
"""

COUNT_LEGACY_RESEARCH_EVIDENCE = """
CALL () {
    MATCH (evidence:ExternalEvidence)
    RETURN count(evidence) AS nodes
}
CALL () {
    MATCH ()-[citation:CITES_EXTERNAL]->()
    RETURN count(citation) AS citations
}
RETURN nodes + citations AS count
"""

DROP_EXTERNAL_EVIDENCE_CONSTRAINT = """
DROP CONSTRAINT external_evidence_uuid IF EXISTS
"""

COUNT_REMAINING_USER_RESEARCH_CITATIONS = """
MATCH (:User {uuid: $user_uuid})-[:HAS_TEACHING_SESSION]->
      (:TeachingSession)-[:HAS_TURN]->
      (:TeachingTurn)-[:CITES_EXTERNAL]->(evidence:ExternalEvidence)
RETURN count(DISTINCT evidence) AS count
"""
