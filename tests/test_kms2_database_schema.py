import asyncio

from kms2.database.schema import (
    SCHEMA_MIGRATION_STATEMENTS,
    SCHEMA_STATEMENTS,
    VECTOR_INDEX_NAMES,
    ensure_schema,
    vector_index_statements,
)


class _RecordingSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run(self, statement: str, **parameters: object) -> None:
        self.calls.append((statement, parameters))


class _SessionContext:
    def __init__(self, session: _RecordingSession) -> None:
        self._session = session

    async def __aenter__(self) -> _RecordingSession:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_ensure_schema_runs_migration_structural_vector_and_wait_ddl():
    session = _RecordingSession()

    asyncio.run(
        ensure_schema(
            lambda: _SessionContext(session),
            embedding_dimension=4096,
        )
    )

    statements = [statement for statement, _ in session.calls]
    expected_prefix = (
        *SCHEMA_MIGRATION_STATEMENTS,
        *SCHEMA_STATEMENTS,
        *vector_index_statements(4096),
    )
    assert statements[: len(expected_prefix)] == list(expected_prefix)
    assert statements[len(expected_prefix) :] == [
        'CALL db.awaitIndex($index_name, $timeout_seconds)'
    ] * len(VECTOR_INDEX_NAMES)
    assert all(
        parameters == {}
        for _, parameters in session.calls[: len(expected_prefix)]
    )
    assert [
        parameters for _, parameters in session.calls[len(expected_prefix) :]
    ] == [
        {'index_name': index_name, 'timeout_seconds': 300}
        for index_name in VECTOR_INDEX_NAMES
    ]

    schema = '\n'.join(statements)
    assert 'SourcePage' not in schema
    assert 'SourceBlock' in schema
    assert 'VisualAsset' in schema
    assert 'Triplet' in schema
    assert 'SourceEntity' in schema
    assert 'SourceEvent' in schema
    assert 'SourcePredicate' in schema
    assert 'SourceEntityHub' in schema
    assert 'SourceEventHub' in schema
    assert 'SourcePredicateHub' in schema
    assert 'source_entity_hub_embedding' in schema
    assert 'source_event_hub_embedding' in schema
    assert 'source_predicate_hub_embedding' in schema
    assert 'HAS_PAGE' not in schema
    assert 'CONTAINS_BLOCK' not in schema
