import asyncio
from types import SimpleNamespace

from kms2.config import ContextWindowSettings
from kms2.core.model import SourceBlock, SourceProcedure, SourceStatement
from kms2.langgraph.semantic.state import SemanticState
from kms2.node.semantic.source_procedure_description_load import (
    SourceProcedureDescriptionLoadNode,
)
from kms2.node.semantic.source_statement_description import (
    SourceStatementDescriptionNode,
)
from kms2.node.semantic.source_statement_description_load import (
    SourceStatementDescriptionLoadNode,
)
from kms2.node.semantic.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)


class _SourceRepository:
    def __init__(self, blocks):
        self.blocks = blocks

    async def load_blocks(self, source_uuid):
        return self.blocks


class _SemanticRepository:
    def __init__(self, statements, procedures):
        self.statements = statements
        self.procedures = procedures

    async def load_source_statements(self, source_uuid):
        return self.statements

    async def load_source_procedures(self, source_uuid):
        return self.procedures


class _DescriptionModule:
    async def aforward(self, *, request):
        return 'generated description'


class _EmbeddingClient:
    async def embed(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


def _blocks():
    return [
        SourceBlock(uuid='block-1', block_type='paragraph', content='one'),
        SourceBlock(uuid='block-2', block_type='paragraph', content='two'),
        SourceBlock(uuid='block-3', block_type='paragraph', content='three'),
    ]


def test_statement_loader_orders_targets_and_skips_missing_members():
    statements = [
        SourceStatement(
            uuid='statement-2',
            source_uuid='source-1',
            member_block_uuids=['block-3'],
        ),
        SourceStatement(
            uuid='statement-1',
            source_uuid='source-1',
            member_block_uuids=['missing'],
        ),
        SourceStatement(
            uuid='statement-0',
            source_uuid='source-1',
            member_block_uuids=['block-1', 'block-2'],
        ),
    ]
    repository = _SemanticRepository(statements, [])
    node = SourceStatementDescriptionLoadNode(
        _SourceRepository(_blocks()),
        repository,
        ContextWindowSettings(backward_budget=0, forward_budget=0),
    )

    result = asyncio.run(node.run(SemanticState(source_uuid='source-1')))

    assert [
        request.target.uuid
        for request in result['source_statement_description_requests']
    ] == ['statement-0', 'statement-2']
    assert result['source_statement_description_requests'][
        0
    ].model_input.target_blocks


def test_procedure_loader_projects_multi_block_targets():
    procedure = SourceProcedure(
        uuid='procedure-1',
        source_uuid='source-1',
        member_block_uuids=['block-1', 'block-2'],
    )
    node = SourceProcedureDescriptionLoadNode(
        _SourceRepository(_blocks()),
        _SemanticRepository([], [procedure]),
        ContextWindowSettings(backward_budget=0, forward_budget=0),
    )

    result = asyncio.run(node.run(SemanticState(source_uuid='source-1')))

    request = result['source_procedure_description_requests'][0]
    assert request.target.member_block_uuids == ['block-1', 'block-2']
    assert len(request.model_input.target_blocks) == 2


def test_statement_description_worker_and_embedding_preserve_identity():
    target = SourceStatement(
        uuid='statement-1',
        source_uuid='source-1',
        member_block_uuids=['block-1'],
    )
    request = SimpleNamespace(target=target, model_input=SimpleNamespace())
    description_node = SourceStatementDescriptionNode(_DescriptionModule())

    described = asyncio.run(
        description_node.worker(
            {'source_statement_description_request': request}
        )
    )
    result = described['source_statement_description_results'][0]
    embedded = asyncio.run(
        SourceStatementEmbeddingNode(_EmbeddingClient()).worker(
            {'source_statement_description_results': [result]}
        )
    )

    assert (
        embedded['source_statement_embedding_results'][0].uuid == 'statement-1'
    )
    assert embedded['source_statement_embedding_results'][0].embedding == [0.0]
