"""Deterministic Source Pedagogical reconstruction and embedding coverage."""

import asyncio
from pathlib import Path

from PIL import Image

from kms.core.embedding import EmbeddingRequest
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.media.images import load_content_images_sync
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.material import (
    Instruction,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.source import Source, SourceBlock, SourcePage
from kms.source.processing.procedure_reconstruction.embedding import (
    SourceProcedureEmbeddingNode,
)
from kms.source.processing.procedure_reconstruction.models import (
    SourceProcedureReconstructionOutput,
)
from kms.source.processing.procedure_reconstruction.node import (
    SourceProcedureReconstructionNode,
)
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.statement_reconstruction.embedding import (
    SourceStatementEmbeddingNode,
)
from kms.source.processing.statement_reconstruction.models import (
    SourceStatementReconstructionOutput,
)
from kms.source.processing.statement_reconstruction.node import (
    SourceStatementReconstructionNode,
)


class _Counter:
    def count_texts(self, texts):
        return [1 for _ in texts]


class _FakeEmbedder:
    def __init__(self):
        self.calls = []
        self.image_payloads = []

    async def embed(self, request: EmbeddingRequest):
        texts = [material.text or '' for material in request.materials]
        self.calls.append(texts)
        self.image_payloads = [
            load_content_images_sync(material.images, budget=None)
            for material in request.materials
        ]
        return [
            [float(index), float(len(text))] for index, text in enumerate(texts)
        ]


class _Reconstructor:
    def __init__(self, output_model, *, wait_for=None, signal=None):
        self.output_model = output_model
        self.wait_for = wait_for
        self.signal = signal
        self.calls = []

    async def acall(self, *, items):
        self.calls.append(list(items))
        first_text = items[0].request.target_blocks[0].material.text
        if self.wait_for is not None and first_text == self.wait_for:
            await self.signal.wait()
        elif self.signal is not None:
            self.signal.set()
        # Deliberately return valid decisions in reverse packet-local order.
        return [
            self.output_model(
                position=item.position,
                content=f'Complete reconstruction: {item.request.target_blocks[0].material.text}',
            )
            for item in reversed(items)
        ]


def _fixture(tmp_path):
    figure_path = tmp_path / 'figure-one.png'
    directive_figure_path = tmp_path / 'directive-figure.png'
    Image.new('RGB', (8, 8), (255, 0, 0)).save(figure_path, format='PNG')
    Image.new('RGB', (8, 8), (0, 0, 255)).save(
        directive_figure_path, format='PNG'
    )
    blocks = [
        SourceBlock(
            uuid='z-directive-shared',
            block_type='instruction',
            content='For Exercises 1 and 2, determine the requested value.',
        ),
        SourceBlock(
            uuid='a-directive-one',
            block_type='instruction',
            content='For Exercise 1, justify your answer.',
        ),
        SourceBlock(
            uuid='x-directive-figure',
            block_type='image',
            content='A second image illustrating the directive.',
            image_path=str(directive_figure_path),
            image_page_indexes=[0],
        ),
        SourceBlock(
            uuid='m-exercise-one',
            block_type='paragraph',
            content='1. Find x. Use the supplied hint |x| = 3.',
        ),
        SourceBlock(
            uuid='b-equation-one',
            block_type='equation',
            content=r'|x|=3',
        ),
        SourceBlock(
            uuid='y-figure-one',
            block_type='image',
            content='A line segment marked with endpoints A and B.',
            image_path=str(figure_path),
            image_page_indexes=[0],
            crop_path='/private/crop-one.png',
            crop_bbox=(1, 2, 30, 40),
        ),
        SourceBlock(
            uuid='c-exercise-two',
            block_type='paragraph',
            content='2. State whether the equation has a solution.',
        ),
        SourceBlock(
            uuid='u-unrelated',
            block_type='paragraph',
            content='Unrelated neighboring theorem must not be evidence.',
        ),
        SourceBlock(
            uuid='p-procedure-first',
            block_type='paragraph',
            content='Step 1: isolate the variable.',
        ),
        SourceBlock(
            uuid='q-procedure-equation',
            block_type='equation',
            content=r'2x=6',
        ),
        SourceBlock(
            uuid='r-procedure-last',
            block_type='paragraph',
            content='Step 2: divide by 2 and check the result.',
        ),
    ]
    pages = [
        SourcePage(index=0, blocks=blocks[:8]),
        SourcePage(index=1, blocks=blocks[8:]),
    ]
    statements = [
        StatementDraft(
            uuid='statement-two',
            member_block_uuids=['c-exercise-two'],
            is_exercise=True,
        ),
        StatementDraft(
            uuid='statement-one',
            member_block_uuids=[
                'y-figure-one',
                'b-equation-one',
                'm-exercise-one',
            ],
            is_exercise=True,
        ),
        StatementDraft(
            uuid='statement-unrelated',
            member_block_uuids=['u-unrelated'],
        ),
    ]
    procedures = [
        ProcedureDraft(
            uuid='procedure-main',
            member_block_uuids=[
                'r-procedure-last',
                'p-procedure-first',
                'q-procedure-equation',
            ],
        )
    ]
    instructions = [
        Instruction(
            uuid='instruction-shared',
            member_block_uuids=['z-directive-shared', 'x-directive-figure'],
            governed_statement_uuids=['statement-two', 'statement-one'],
        ),
        Instruction(
            uuid='instruction-one',
            member_block_uuids=['a-directive-one'],
            governed_statement_uuids=['statement-one'],
        ),
    ]
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='capture.pdf'),
        source=Source(uuid='source-1', key='capture.pdf'),
        split_pages=pages,
        statements=statements,
        procedures=procedures,
        instructions=instructions,
    )
    return state, blocks


def _node(module, batch_size=2):
    return SourceStatementReconstructionNode(
        module,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=6
        ),
        batch_size=batch_size,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )


def test_reconstruction_then_embedding_preserves_scoped_evidence_and_identity(
    tmp_path,
):
    async def exercise():
        state, blocks = _fixture(tmp_path)
        released = asyncio.Event()
        module = _Reconstructor(
            SourceStatementReconstructionOutput,
            wait_for='1. Find x. Use the supplied hint |x| = 3.',
            signal=released,
        )
        node = _node(module)
        state = state.model_copy(update=node.run(state))
        requests = state.source_statement_reconstruction_requests
        by_uuid = {request.target.uuid: request for request in requests}
        first = by_uuid['statement-one']
        assert [block.content for block in first.target_blocks] == [
            '1. Find x. Use the supplied hint |x| = 3.',
            r'|x|=3',
            'A line segment marked with endpoints A and B.',
        ]
        assert [block.content for block in first.instruction_blocks] == [
            'For Exercises 1 and 2, determine the requested value.',
            'For Exercise 1, justify your answer.',
            'A second image illustrating the directive.',
        ]
        second = by_uuid['statement-two']
        assert [block.content for block in second.instruction_blocks] == [
            'For Exercises 1 and 2, determine the requested value.',
            'A second image illustrating the directive.',
        ]
        assert [block.content for block in second.target_blocks] == [
            '2. State whether the equation has a solution.'
        ]
        assert all(
            'Unrelated neighboring theorem' not in block.content
            for request in (first, second)
            for block in [*request.target_blocks, *request.instruction_blocks]
        )
        assert by_uuid['statement-unrelated'].instruction_blocks == []

        sends = await node.dispatch(state)
        assert isinstance(sends, list) and len(sends) == 2
        # The second packet completes and releases the first, proving out-of-order
        # completion without timing assumptions. Each packet returns reversed outputs.
        workers = [asyncio.create_task(node.worker(send.arg)) for send in sends]
        second_result = await workers[1]
        first_result = await workers[0]
        merged = state.model_copy(
            update={
                'source_statement_reconstruction_results': (
                    second_result['source_statement_reconstruction_results']
                    + first_result['source_statement_reconstruction_results']
                )
            }
        )
        collected = node.collect(merged)['statements']
        assert [item.uuid for item in collected] == [
            item.uuid for item in state.statements
        ]
        assert [item.member_block_uuids for item in collected] == [
            item.member_block_uuids for item in state.statements
        ]
        assert [item.is_exercise for item in collected] == [
            item.is_exercise for item in state.statements
        ]
        assert [item.content for item in collected] == [
            'Complete reconstruction: 2. State whether the equation has a solution.',
            'Complete reconstruction: 1. Find x. Use the supplied hint |x| = 3.',
            'Complete reconstruction: Unrelated neighboring theorem must not be evidence.',
        ]

        client = _FakeEmbedder()
        embedded = await SourceStatementEmbeddingNode(client).run(
            state.model_copy(update={'statements': collected})
        )
        expected_images = {
            Path(block.image_path).read_bytes()
            for block in blocks
            if block.image_path is not None
        }
        embedded_images = {
            payload
            for material_images in client.image_payloads
            for payload in material_images
        }
        assert embedded_images == expected_images
        assert [item.uuid for item in embedded['statements']] == [
            item.uuid for item in state.statements
        ]
        assert [item.embedding for item in embedded['statements']] == [
            [0.0, float(len(collected[0].content))],
            [1.0, float(len(collected[1].content))],
            [2.0, float(len(collected[2].content))],
        ]
        assert all(
            item.member_block_uuids == original.member_block_uuids
            for item, original in zip(
                embedded['statements'], state.statements, strict=True
            )
        )
        assert all('For Exercises' not in text for text in client.calls[0])
        assert all('For Exercise 1' not in text for text in client.calls[0])

        procedure_module = _Reconstructor(SourceProcedureReconstructionOutput)
        procedure_node = SourceProcedureReconstructionNode(
            procedure_module,
            budget=TokenBudget(
                counter=MaterialTokenEstimator(_Counter()), token_limit=6
            ),
            batch_size=2,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        )
        procedure_state = state.model_copy(
            update={'source_procedure_reconstruction_requests': []}
        )
        procedure_state = procedure_state.model_copy(
            update=procedure_node.run(procedure_state)
        )
        procedure_request = (
            procedure_state.source_procedure_reconstruction_requests[0]
        )
        assert [block.content for block in procedure_request.target_blocks] == [
            'Step 1: isolate the variable.',
            r'2x=6',
            'Step 2: divide by 2 and check the result.',
        ]
        assert not hasattr(procedure_request, 'instruction_blocks')
        procedure_sends = await procedure_node.dispatch(procedure_state)
        assert isinstance(procedure_sends, list)
        procedure_result = await procedure_node.worker(procedure_sends[0].arg)
        procedure_state = procedure_state.model_copy(update=procedure_result)
        procedures = procedure_node.collect(procedure_state)['procedures']
        procedure_client = _FakeEmbedder()
        embedded_procedures = await SourceProcedureEmbeddingNode(
            procedure_client
        ).run(procedure_state.model_copy(update={'procedures': procedures}))
        assert not any(procedure_client.image_payloads)
        assert embedded_procedures['procedures'][0].uuid == 'procedure-main'
        assert (
            embedded_procedures['procedures'][0].member_block_uuids
            == procedures[0].member_block_uuids
        )
        assert embedded_procedures['procedures'][0].embedding == [
            0.0,
            float(len(procedures[0].content)),
        ]

    asyncio.run(exercise())


def test_empty_reconstruction_and_embedding_branches_bypass_dependencies():
    class UnusedCounter:
        def count_texts(self, _texts):
            raise AssertionError('empty dispatch must not count tokens')

    class UnusedModule:
        async def acall(self, **_kwargs):
            raise AssertionError('empty branch must not invoke inference')

    class UnusedEmbedder:
        async def embed(self, _request: EmbeddingRequest):
            raise AssertionError('empty branch must not invoke embedding')

    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='empty.pdf'),
        source=Source(uuid='source-empty', key='empty.pdf'),
    )
    statement_node = SourceStatementReconstructionNode(
        UnusedModule(),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(UnusedCounter()), token_limit=10
        ),
        batch_size=2,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )
    procedure_node = SourceProcedureReconstructionNode(
        UnusedModule(),
        budget=TokenBudget(
            counter=MaterialTokenEstimator(UnusedCounter()), token_limit=10
        ),
        batch_size=2,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )
    statement_state = state.model_copy(update=statement_node.run(state))
    procedure_state = state.model_copy(update=procedure_node.run(state))
    assert (
        asyncio.run(statement_node.dispatch(statement_state))
        == 'source_statement_reconstruction_collect'
    )
    assert (
        asyncio.run(procedure_node.dispatch(procedure_state))
        == 'source_procedure_reconstruction_collect'
    )
    assert statement_node.collect(statement_state)['statements'] == []
    assert procedure_node.collect(procedure_state)['procedures'] == []
    assert asyncio.run(
        SourceStatementEmbeddingNode(UnusedEmbedder()).run(state)
    ) == {'statements': []}
    assert asyncio.run(
        SourceProcedureEmbeddingNode(UnusedEmbedder()).run(state)
    ) == {'procedures': []}
