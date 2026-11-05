import asyncio

import pytest

from kms.database.schema import ensure_schema


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
        self.vector_writes = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def run(self, statement, **parameters):
        if 'SHOW INDEXES' in statement:
            return Result(self.indexes)
        if 'size(node.embedding)' in statement:
            return Result(self.incompatible_vectors)
        if 'CREATE VECTOR INDEX' in statement:
            self.vector_writes.append(statement)
        return Result()


def test_incompatible_index_requires_explicit_reembedding_without_index_writes():
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


def test_old_vectors_cannot_be_mixed_into_new_dimension_without_an_index():
    session = Session(
        incompatible_vectors=[
            {
                'labels': ['SourceBlock'],
                'dimension': 4096,
            }
        ]
    )
    with pytest.raises(ValueError, match='Re-embed stored material'):
        asyncio.run(ensure_schema(lambda: session, embedding_dimension=1024))
    assert session.vector_writes == []


def test_argument_vectors_are_rejected_when_their_dimension_is_stale():
    session = Session(
        incompatible_vectors=[
            {
                'labels': ['Argument'],
                'dimension': 4096,
            }
        ]
    )
    with pytest.raises(ValueError, match='Re-embed stored material'):
        asyncio.run(ensure_schema(lambda: session, embedding_dimension=1024))
    assert session.vector_writes == []
