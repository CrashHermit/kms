"""Deterministic Source Pedagogical reconstruction and embedding coverage."""

import asyncio
import json

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source_pedagogical.instruction import Instruction
from kms2.core.model.source_pedagogical.pedagogical import (
    ProcedureDraft,
    StatementDraft,
)
from kms2.core.model.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionOutput,
)
from kms2.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionOutput,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.node.source_pedagogical.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)
from kms2.node.source_pedagogical.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)


class _Counter:
    def count_texts(self, texts):
        return [1 for _ in texts]


class _EmbeddingClient:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
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
        first_text = items[0].request.target_blocks[0].content
        if self.wait_for is not None and first_text == self.wait_for:
            await self.signal.wait()
        elif self.signal is not None:
            self.signal.set()
        # Deliberately return valid decisions in reverse packet-local order.
        return [
            self.output_model(
                position=item.position,
                content=f'Complete reconstruction: {item.request.target_blocks[0].content}',
            )
            for item in reversed(items)
        ]


def _fixture():
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
            assets=[{'uuid': 'asset-1', 'path': '/private/figure-one.png'}],
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
        SourcePage(index=0, blocks=blocks[:7]),
        SourcePage(index=1, blocks=blocks[7:]),
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
            member_block_uuids=['z-directive-shared'],
            governed_statement_uuids=['statement-two', 'statement-one'],
        ),
        Instruction(
            uuid='instruction-one',
            member_block_uuids=['a-directive-one'],
            governed_statement_uuids=['statement-one'],
        ),
    ]
    state = SourcePedagogicalState(
        source_uuid='source-1',
        split_pages=pages,
        statements=statements,
        procedures=procedures,
        instructions=instructions,
    )
    return state, blocks


def _node(module, batch_size=2):
    return SourceStatementReconstructionNode(
        module,
        budget=TokenBudget(counter=_Counter(), token_limit=6),
        batch_size=batch_size,
    )


def test_reconstruction_then_embedding_preserves_scoped_evidence_and_identity():
    async def exercise():
        state, blocks = _fixture()
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
        first = by_uuid['statement-one'].model_input
        assert [block.content for block in first.target_blocks] == [
            '1. Find x. Use the supplied hint |x| = 3.',
            r'|x|=3',
            'A line segment marked with endpoints A and B.',
        ]
        assert [block.content for block in first.instruction_blocks] == [
            'For Exercises 1 and 2, determine the requested value.',
            'For Exercise 1, justify your answer.',
        ]
        second = by_uuid['statement-two'].model_input
        assert [block.content for block in second.instruction_blocks] == [
            'For Exercises 1 and 2, determine the requested value.'
        ]
        assert [block.content for block in second.target_blocks] == [
            '2. State whether the equation has a solution.'
        ]
        assert all(
            'Unrelated neighboring theorem' not in block.content
            for request in (first, second)
            for block in [*request.target_blocks, *request.instruction_blocks]
        )
        assert (
            by_uuid['statement-unrelated'].model_input.instruction_blocks == []
        )
        serialized = json.dumps(
            [
                request.model_input.model_dump(mode='json')
                for request in requests
            ]
        )
        assert all(block.uuid not in serialized for block in blocks)
        assert '/private/' not in serialized

        sends = node.dispatch(state)
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

        client = _EmbeddingClient()
        embedded = await SourceStatementEmbeddingNode(client).run(
            state.model_copy(update={'statements': collected})
        )
        assert client.calls == [[item.content for item in collected]]
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
            budget=TokenBudget(counter=_Counter(), token_limit=6),
            batch_size=2,
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
        assert [
            block.content
            for block in procedure_request.model_input.target_blocks
        ] == [
            'Step 1: isolate the variable.',
            r'2x=6',
            'Step 2: divide by 2 and check the result.',
        ]
        assert not hasattr(procedure_request.model_input, 'instruction_blocks')
        procedure_sends = procedure_node.dispatch(procedure_state)
        assert isinstance(procedure_sends, list)
        procedure_result = await procedure_node.worker(procedure_sends[0].arg)
        procedure_state = procedure_state.model_copy(update=procedure_result)
        procedures = procedure_node.collect(procedure_state)['procedures']
        procedure_client = _EmbeddingClient()
        embedded_procedures = await SourceProcedureEmbeddingNode(
            procedure_client
        ).run(procedure_state.model_copy(update={'procedures': procedures}))
        assert procedure_client.calls == [[procedures[0].content]]
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

    class UnusedEmbeddingClient:
        async def embed(self, _texts):
            raise AssertionError('empty branch must not invoke embedding')

    state = SourcePedagogicalState(source_uuid='source-empty')
    statement_node = SourceStatementReconstructionNode(
        UnusedModule(),
        budget=TokenBudget(counter=UnusedCounter(), token_limit=10),
        batch_size=2,
    )
    procedure_node = SourceProcedureReconstructionNode(
        UnusedModule(),
        budget=TokenBudget(counter=UnusedCounter(), token_limit=10),
        batch_size=2,
    )
    statement_state = state.model_copy(update=statement_node.run(state))
    procedure_state = state.model_copy(update=procedure_node.run(state))
    assert (
        statement_node.dispatch(statement_state)
        == 'source_statement_reconstruction_collect'
    )
    assert (
        procedure_node.dispatch(procedure_state)
        == 'source_procedure_reconstruction_collect'
    )
    assert statement_node.collect(statement_state)['statements'] == []
    assert procedure_node.collect(procedure_state)['procedures'] == []
    assert asyncio.run(
        SourceStatementEmbeddingNode(UnusedEmbeddingClient()).run(state)
    ) == {'statements': []}
    assert asyncio.run(
        SourceProcedureEmbeddingNode(UnusedEmbeddingClient()).run(state)
    ) == {'procedures': []}
