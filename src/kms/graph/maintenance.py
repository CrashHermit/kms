"""Maintenance operations for rebuilding persisted graph embeddings."""

from collections.abc import Callable

from kms import config
from kms.core import embeddings

# Every label below has a vector index declared by graph.schema.schema_statements.
_EMBEDDING_SPECS = (
    ('Node', 'content'),
    ('Statement', 'description'),
    ('Entity', 'description'),
    ('Event', 'description'),
    ('Predicate', 'description'),
    ('LocalEntityHub', 'description'),
    ('LocalEventHub', 'description'),
    ('LocalPredicateHub', 'description'),
    ('GlobalEntityHub', 'description'),
    ('GlobalEventHub', 'description'),
    ('GlobalPredicateHub', 'description'),
    ('LocalTripletHub', 'description'),
    ('GlobalTripletHub', 'description'),
    ('Procedure', 'procedure'),
    ('GlobalStatementHub', 'description'),
    ('GlobalProcedureHub', 'description'),
    ('LocalStatementHub', 'description'),
    ('LocalProcedureHub', 'description'),
)


async def rebuild_embeddings(
    session_factory: Callable,
    *,
    batch_size: int | None = None,
) -> dict[str, int]:
    """Re-embed all vector-indexed records without changing their UUIDs."""
    limit = batch_size or config.get_settings().embeddings.batch_size
    counts: dict[str, int] = {}
    async with session_factory() as session:
        for label, text_property in _EMBEDDING_SPECS:
            result = await session.run(
                f'MATCH (n:{label}) '
                f'RETURN n.uuid AS uuid, n.{text_property} AS text'
            )
            rows = await result.data()
            updates: list[dict] = []
            non_empty = [row for row in rows if (row.get('text') or '').strip()]
            vectors = await embeddings.embedder().embed(
                [row['text'] for row in non_empty]
            )
            for row, vector in zip(non_empty, vectors, strict=True):
                updates.append({'uuid': row['uuid'], 'embedding': vector})
            for start in range(0, len(updates), limit):
                await session.run(
                    f'UNWIND $rows AS row MATCH (n:{label} {{uuid: row.uuid}}) '
                    'SET n.embedding = row.embedding',
                    rows=updates[start : start + limit],
                )
            empty = [
                {'uuid': row['uuid']}
                for row in rows
                if not (row.get('text') or '').strip()
            ]
            for start in range(0, len(empty), limit):
                await session.run(
                    f'UNWIND $rows AS row MATCH (n:{label} {{uuid: row.uuid}}) '
                    'REMOVE n.embedding',
                    rows=empty[start : start + limit],
                )
            counts[label] = len(updates)
    return counts
