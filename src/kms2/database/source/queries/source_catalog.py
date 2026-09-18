"""Cypher statements for listing persisted sources."""

READ_SOURCES = """
MATCH (source:Source)
RETURN source.uuid AS uuid, source.key AS key
ORDER BY source.key, source.uuid
"""
