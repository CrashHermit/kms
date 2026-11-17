import asyncio

import pytest

from kms.database.schema import (
    VECTOR_INDEX_NAMES,
    ensure_schema,
    ensure_structural_schema,
    vector_index_statements,
)


class Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    async def data(self):
        return self.rows

    async def consume(self):
        pass


class Session:
    def __init__(self, *, indexes=(), incompatible_vectors=()):
        self.indexes = indexes
        self.incompatible_vectors = incompatible_vectors
        self.statements = []
        self.vector_writes = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, statement, **parameters):
        self.statements.append((statement, parameters))
        if 'SHOW INDEXES' in statement:
            return Result(self.indexes)
        if 'size(node.embedding)' in statement:
            return Result(self.incompatible_vectors)
        if 'CREATE VECTOR INDEX' in statement:
            self.vector_writes.append(statement)
        return Result()


def test_structural_schema_does_not_destroy_existing_data():
    session = Session()

    asyncio.run(ensure_structural_schema(lambda: session))

    assert not any(
        'CALL db.create' in statement for statement, _ in session.statements
    )
    assert not any('DROP' in statement for statement, _ in session.statements)


def test_vector_schema_contains_only_block_and_unit_indexes():
    statements = vector_index_statements(1024)

    assert VECTOR_INDEX_NAMES == (
        'source_block_embedding',
        'source_unit_embedding',
    )
    assert len(statements) == len(VECTOR_INDEX_NAMES)
    assert all('1024' in statement for statement in statements)
    assert all(
        name in statement
        for name, statement in zip(VECTOR_INDEX_NAMES, statements, strict=True)
    )


def test_schema_initialization_creates_and_awaits_only_source_indexes():
    session = Session()

    asyncio.run(ensure_schema(lambda: session, embedding_dimension=1024))

    assert len(session.vector_writes) == 2
    assert all(
        name in statement
        for name, statement in zip(
            VECTOR_INDEX_NAMES, session.vector_writes, strict=True
        )
    )
    awaited = [
        parameters['index_name']
        for statement, parameters in session.statements
        if 'db.awaitIndex' in statement
    ]
    assert awaited == list(VECTOR_INDEX_NAMES)


def test_incompatible_index_requires_rebuild_without_index_writes():
    session = Session(
        indexes=[
            {
                'name': 'source_block_embedding',
                'type': 'VECTOR',
                'options': {'indexConfig': {'vector.dimensions': 4096}},
            }
        ]
    )
    with pytest.raises(ValueError, match='explicitly rebuild'):
        asyncio.run(ensure_schema(lambda: session, embedding_dimension=1024))
    assert session.vector_writes == []


def test_retained_source_vectors_reject_stale_dimension_before_index_writes():
    session = Session(
        incompatible_vectors=[
            {
                'labels': ['SourceUnit'],
                'dimension': 4096,
            }
        ]
    )
    with pytest.raises(ValueError, match='Re-embed stored material'):
        asyncio.run(ensure_schema(lambda: session, embedding_dimension=1024))
    assert session.vector_writes == []
    _, parameters = next(
        entry
        for entry in session.statements
        if 'size(node.embedding)' in entry[0]
    )
    assert parameters['labels'] == [
        'SourceBlock',
        'SourceUnit',
    ]
