import asyncio

import pytest
from langgraph.graph import END, START, StateGraph
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.config.source_pedagogical import (
    ExerciseFinderSettings,
    InstructionFinderSettings,
    InstructionGovernanceSettings,
    PedagogicalFinderSettings,
    StatementProcedureSettings,
)
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.context import PositionedSourceBlock, SourceBlockContext
from kms.core.model.page import SourcePage
from kms.core.model.source_pedagogical.exercise_splitter import (
    SplitRouteDecision,
)
from kms.core.model.source_pedagogical.instruction import (
    Instruction,
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionGovernanceDecision,
    InstructionGovernanceInput,
    InstructionStartDecision,
)
from kms.core.model.source_pedagogical.pedagogical import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseStartDecision,
    PedagogicalBoundaryDecision,
    PedagogicalClassificationInput,
    PedagogicalStartDecision,
    ProcedureDraft,
    StatementDraft,
)
from kms.core.model.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionOutput,
)
from kms.core.model.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionOutput,
)
from kms.core.model.visual_asset import VisualAsset
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.source_pedagogical.graph import SourcePedagogicalGraph
from kms.langgraph.source_pedagogical.source_procedure_reconstruction import (
    add_source_procedure_reconstruction_phase,
)
from kms.langgraph.source_pedagogical.source_statement_reconstruction import (
    add_source_statement_reconstruction_phase,
)
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.node.source_pedagogical.embedding import (
    SourcePedagogicalEmbeddingNode,
)
from kms.node.source_pedagogical.exercise_finder import ExerciseFinderNode
from kms.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode
from kms.node.source_pedagogical.instruction_finder import (
    InstructionFinderNode,
)
from kms.node.source_pedagogical.instruction_governance import (
    InstructionGovernanceNode,
)
from kms.node.source_pedagogical.pedagogical_finder import (
    PedagogicalFinderNode,
)
from kms.node.source_pedagogical.persistence import (
    SourcePedagogicalPersistenceNode,
)
from kms.node.source_pedagogical.source_load import (
    SourcePedagogicalSourceLoadNode,
)
from kms.node.source_pedagogical.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)
from kms.node.source_pedagogical.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)
from kms.node.source_pedagogical.statement_procedure import (
    StatementProcedureNode,
)


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Repository:
    def __init__(self, pages: list[SourcePage]) -> None:
        self.pages = pages
        self.persisted = None

    async def load_pages(self, source_uuid: str) -> list[SourcePage]:
        assert source_uuid == 'source-1'
        return self.pages

    async def replace_pedagogical_material(
        self,
        source_uuid,
        pages,
        instructions,
        statements,
        procedures,
    ) -> None:
        self.persisted = (
            source_uuid,
            pages,
            instructions,
            statements,
            procedures,
        )


class _NoSplitRouter:
    async def acall(
        self,
        *,
        context_before: list[SourceBlockContext],
        targets: list[PositionedSourceBlock],
        context_after: list[SourceBlockContext],
        content_images: list,
    ) -> list[SplitRouteDecision]:
        return [
            SplitRouteDecision(
                position=target.position,
                contains_multiple_exercises=False,
            )
            for target in targets
        ]


class _UncalledSplitter:
    async def acall(self, **kwargs: object) -> object:
        raise AssertionError('no blocks were routed for splitting')


class _InstructionStart:
    async def acall(
        self,
        *,
        items: list[InstructionClassificationInput],
        content_images: list,
    ) -> list[InstructionStartDecision]:
        return [
            InstructionStartDecision(
                position=item.position,
                is_instruction_start=item.target_block.material.text.startswith(
                    'Directions'
                ),
            )
            for item in reversed(items)
        ]


class _InstructionBoundary:
    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[InstructionClassificationInput],
        content_images: list,
    ) -> list[InstructionBoundaryDecision]:
        return [
            InstructionBoundaryDecision(
                position=item.position,
                is_instruction_boundary=item.target_block.material.text.startswith(
                    'Exercise'
                ),
            )
            for item in reversed(items)
        ]


class _ExerciseStart:
    async def acall(
        self,
        *,
        items: list[ExerciseClassificationInput],
        content_images: list,
    ) -> list[ExerciseStartDecision]:
        return [
            ExerciseStartDecision(
                position=item.position,
                is_exercise_start=item.target_block.material.text.startswith(
                    'Exercise'
                ),
            )
            for item in reversed(items)
        ]


class _ExerciseBoundary:
    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[ExerciseClassificationInput],
        content_images: list,
    ) -> list[ExerciseBoundaryDecision]:
        return [
            ExerciseBoundaryDecision(
                position=item.position,
                is_exercise_boundary=False,
            )
            for item in reversed(items)
        ]


class _PedagogicalStart:
    async def acall(
        self,
        *,
        items: list[PedagogicalClassificationInput],
        content_images: list,
    ) -> list[PedagogicalStartDecision]:
        return [
            PedagogicalStartDecision(
                position=item.position,
                is_pedagogical_start=False,
            )
            for item in reversed(items)
        ]


class _PedagogicalBoundary:
    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[PedagogicalClassificationInput],
        content_images: list,
    ) -> list[PedagogicalBoundaryDecision]:
        return [
            PedagogicalBoundaryDecision(
                position=item.position,
                is_pedagogical_boundary=False,
            )
            for item in reversed(items)
        ]


class _GovernanceJudge:
    async def acall(
        self,
        *,
        instruction_blocks: list[SourceBlockContext],
        items: list[InstructionGovernanceInput],
        content_images: list,
    ) -> list[InstructionGovernanceDecision]:
        return [
            InstructionGovernanceDecision(
                position=item.position,
                governs=True,
            )
            for item in reversed(items)
        ]


class _FakeEmbedder:
    async def embed(
        self, materials, *, content_images, input_type
    ) -> list[list[float]]:
        assert input_type == 'document'
        return [[float(position)] for position, _ in enumerate(materials)]


class _StatementReconstructor:
    async def acall(self, *, items, content_images):
        return [
            SourceStatementReconstructionOutput(
                position=item.position,
                content=' '.join(
                    [
                        *(
                            block.material.text
                            for block in item.request.target_blocks
                        ),
                        *(
                            block.material.text
                            for block in item.request.instruction_blocks
                        ),
                    ]
                ),
            )
            for item in reversed(items)
        ]


class _UncalledReconstructor:
    async def acall(self, **_kwargs):
        raise AssertionError('empty procedure path must not invoke inference')


def test_source_pedagogical_graph_loads_classifies_embeds_and_persists():
    blocks = [
        SourceBlock(
            uuid='block-0', block_type='paragraph', content='Directions'
        ),
        SourceBlock(
            uuid='block-1', block_type='paragraph', content='Exercise 1'
        ),
        SourceBlock(
            uuid='block-2',
            block_type='paragraph',
            content='Compute the derivative',
        ),
    ]
    pages = [SourcePage(index=0, markdown='Exercise section', blocks=blocks)]
    repository = _Repository(pages)
    counter = _Counter()
    budget = TokenBudget(MaterialTokenEstimator(counter), 100_000)
    token_counters = (MaterialTokenEstimator(counter),)
    context_window = ContextWindowSettings(backward_budget=0, forward_budget=0)
    embedder = _FakeEmbedder()

    graph = SourcePedagogicalGraph(
        source_load=SourcePedagogicalSourceLoadNode(repository),
        exercise_splitter=ExerciseSplitterNode(
            _NoSplitRouter(),
            _UncalledSplitter(),
            context_window,
            batch_size=16,
            token_counters=token_counters,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        instruction_finder=InstructionFinderNode(
            _InstructionStart(),
            _InstructionBoundary(),
            InstructionFinderSettings(
                start_context_window=context_window,
                boundary_context_window=context_window,
            ),
            start_budget=budget,
            boundary_budget=budget,
            start_token_counters=token_counters,
            boundary_token_counters=token_counters,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        exercise_finder=ExerciseFinderNode(
            _ExerciseStart(),
            _ExerciseBoundary(),
            ExerciseFinderSettings(
                start_context_window=context_window,
                boundary_context_window=context_window,
            ),
            start_budget=budget,
            boundary_budget=budget,
            start_token_counters=token_counters,
            boundary_token_counters=token_counters,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        pedagogical_finder=PedagogicalFinderNode(
            _PedagogicalStart(),
            _PedagogicalBoundary(),
            PedagogicalFinderSettings(
                start_context_window=context_window,
                boundary_context_window=context_window,
            ),
            start_budget=budget,
            boundary_budget=budget,
            start_token_counters=token_counters,
            boundary_token_counters=token_counters,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        statement_procedure=StatementProcedureNode(
            None,
            None,
            None,
            StatementProcedureSettings(),
            role_budget=budget,
            statement_budget=budget,
            procedure_budget=budget,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        instruction_governance=InstructionGovernanceNode(
            _GovernanceJudge(),
            InstructionGovernanceSettings(context_window=context_window),
            budget=budget,
            token_counters=token_counters,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        source_statement_reconstruction=SourceStatementReconstructionNode(
            _StatementReconstructor(),
            budget=budget,
            batch_size=16,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        source_procedure_reconstruction=SourceProcedureReconstructionNode(
            _UncalledReconstructor(),
            budget=budget,
            batch_size=16,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        source_statement_embedding=SourceStatementEmbeddingNode(embedder),
        source_procedure_embedding=SourceProcedureEmbeddingNode(embedder),
        embedding=SourcePedagogicalEmbeddingNode(embedder),
        persistence=SourcePedagogicalPersistenceNode(repository),
    )

    result = asyncio.run(
        graph.run(SourcePedagogicalState(source_uuid='source-1'))
    )

    assert [item.member_block_uuids for item in result.instructions] == [
        ['block-0']
    ]
    assert [item.member_block_uuids for item in result.exercise_components] == [
        ['block-1', 'block-2']
    ]
    assert [item.member_block_uuids for item in result.statements] == [
        ['block-1', 'block-2']
    ]
    assert result.instructions[0].governed_statement_uuids == [
        result.statements[0].uuid
    ]
    assert result.reembedded_block_count == 3
    statement = result.statements[0]
    assert statement.content == ('Exercise 1 Compute the derivative Directions')
    assert statement.embedding == [0.0]
    assert repository.persisted[3] == result.statements
    assert [
        block.embedding
        for page in result.embedded_pages
        for block in page.blocks
    ] == [[0.0], [1.0], [2.0]]
    assert repository.persisted == (
        'source-1',
        result.embedded_pages,
        result.instructions,
        result.statements,
        result.procedures,
    )


class _ControlledReconstructor:
    def __init__(
        self,
        output_model,
        kind,
        slow_text,
        fast_text,
        fast_completed,
        slow_started,
        release_slow,
        slow_completed,
        order,
    ):
        self.output_model = output_model
        self.kind = kind
        self.slow_text = slow_text
        self.fast_text = fast_text
        self.fast_completed = fast_completed
        self.slow_started = slow_started
        self.release_slow = release_slow
        self.slow_completed = slow_completed
        self.order = order

    async def acall(self, *, items, content_images):
        assert len(items) == 1
        item = items[0]
        text = item.request.target_blocks[0].material.text
        if text == self.slow_text:
            self.slow_started.set()
            await self.release_slow.wait()
            self.order.append(f'{self.kind}-slow')
            self.slow_completed.set()
        else:
            assert text == self.fast_text
            await self.slow_started.wait()
            self.order.append(f'{self.kind}-fast')
            self.fast_completed.set()
        return [
            self.output_model(
                position=item.position,
                content=f'complete:{self.kind}:{text}',
            )
        ]


class _BarrierEmbedder:
    def __init__(self, block_started, block_finished, order):
        self.block_started = block_started
        self.block_finished = block_finished
        self.order = order

    async def embed(self, materials, *, content_images, input_type):
        texts = [material.text or '' for material in materials]
        if texts[0].startswith('complete:'):
            return [[float(len(text))] for text in texts]
        assert texts[0] == 'Shared directive'
        self.block_started.set()
        self.order.append('block-embedding')
        self.block_finished.set()
        return [[float(len(text))] for text in texts]


class _PersistRecorder:
    def __init__(self):
        self.calls = []
        self.persisted = None
        self.order = None

    async def replace_pedagogical_material(
        self,
        source_uuid,
        pages,
        instructions,
        statements,
        procedures,
    ):
        self.calls.append(
            (source_uuid, pages, instructions, statements, procedures)
        )
        self.persisted = self.calls[-1]
        if self.order is not None:
            self.order.append('persistence')


def _reconstruction_test_graph(
    statement_module,
    procedure_module,
    embedder,
    repository,
):
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100_000)
    graph = StateGraph(SourcePedagogicalState)
    graph.add_node('instruction_governance', lambda _state: {})
    graph.add_edge(START, 'instruction_governance')
    add_source_statement_reconstruction_phase(
        graph,
        SourceStatementReconstructionNode(
            statement_module,
            budget=budget,
            batch_size=1,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
    )
    add_source_procedure_reconstruction_phase(
        graph,
        SourceProcedureReconstructionNode(
            procedure_module,
            budget=budget,
            batch_size=1,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
    )
    graph.add_node(
        'source_statement_embedding',
        SourceStatementEmbeddingNode(embedder).run,
    )
    graph.add_node(
        'source_procedure_embedding',
        SourceProcedureEmbeddingNode(embedder).run,
    )
    graph.add_node(
        'source_pedagogical_embedding',
        SourcePedagogicalEmbeddingNode(embedder).run,
    )
    graph.add_node(
        'source_pedagogical_persistence',
        SourcePedagogicalPersistenceNode(repository).run,
    )
    graph.add_edge('instruction_governance', 'source_pedagogical_embedding')
    graph.add_edge(
        [
            'source_statement_embedding',
            'source_procedure_embedding',
            'source_pedagogical_embedding',
        ],
        'source_pedagogical_persistence',
    )
    graph.add_edge('source_pedagogical_persistence', END)
    return graph.compile()


def test_material_fan_in_waits_for_all_concurrent_reconstructions_and_vectors(
    tmp_path,
):
    async def exercise():
        figure_path = tmp_path / 'figure.png'
        Image.new('RGB', (8, 8), (0, 0, 255)).save(figure_path, format='PNG')
        order = []
        statement_fast = asyncio.Event()
        statement_slow_started = asyncio.Event()
        statement_release_slow = asyncio.Event()
        statement_slow_completed = asyncio.Event()
        procedure_fast = asyncio.Event()
        procedure_slow_started = asyncio.Event()
        procedure_release_slow = asyncio.Event()
        procedure_slow_completed = asyncio.Event()
        block_started = asyncio.Event()
        block_finished = asyncio.Event()
        statement_module = _ControlledReconstructor(
            SourceStatementReconstructionOutput,
            'statement',
            'Delayed statement',
            'Fast statement',
            statement_fast,
            statement_slow_started,
            statement_release_slow,
            statement_slow_completed,
            order,
        )
        procedure_module = _ControlledReconstructor(
            SourceProcedureReconstructionOutput,
            'procedure',
            'Delayed procedure',
            'Fast procedure',
            procedure_fast,
            procedure_slow_started,
            procedure_release_slow,
            procedure_slow_completed,
            order,
        )
        embedder = _BarrierEmbedder(block_started, block_finished, order)
        repository = _PersistRecorder()
        repository.order = order
        pages = [
            SourcePage(
                index=0,
                blocks=[
                    SourceBlock(
                        uuid='block-directive',
                        block_type='instruction',
                        content='Shared directive',
                    ),
                    SourceBlock(
                        uuid='block-statement-slow',
                        block_type='paragraph',
                        content='Delayed statement',
                    ),
                    SourceBlock(
                        uuid='block-statement-fast',
                        block_type='paragraph',
                        content='Fast statement',
                    ),
                ],
            ),
            SourcePage(
                index=1,
                blocks=[
                    SourceBlock(
                        uuid='block-procedure-slow',
                        block_type='paragraph',
                        content='Delayed procedure',
                    ),
                    SourceBlock(
                        uuid='block-procedure-fast',
                        block_type='paragraph',
                        content='Fast procedure',
                        assets=[
                            VisualAsset(
                                uuid='asset-preserved', path=str(figure_path)
                            )
                        ],
                    ),
                ],
            ),
        ]
        graph = _reconstruction_test_graph(
            statement_module,
            procedure_module,
            embedder,
            repository,
        )
        initial_state = {
            'source_uuid': 'source-1',
            'split_pages': pages,
            'instructions': [
                Instruction(
                    uuid='instruction-1',
                    member_block_uuids=['block-directive'],
                    governed_statement_uuids=[
                        'statement-slow',
                        'statement-fast',
                    ],
                )
            ],
            'statements': [
                StatementDraft(
                    uuid='statement-fast',
                    member_block_uuids=['block-statement-fast'],
                    is_exercise=True,
                ),
                StatementDraft(
                    uuid='statement-slow',
                    member_block_uuids=['block-statement-slow'],
                    is_exercise=True,
                ),
            ],
            'procedures': [
                ProcedureDraft(
                    uuid='procedure-fast',
                    member_block_uuids=['block-procedure-fast'],
                ),
                ProcedureDraft(
                    uuid='procedure-slow',
                    member_block_uuids=['block-procedure-slow'],
                ),
            ],
        }
        graph_task = asyncio.create_task(
            graph.ainvoke(initial_state, config={'max_concurrency': 16})
        )

        async def wait_for_event(event, name):
            waiter = asyncio.create_task(event.wait())
            done, _ = await asyncio.wait(
                {waiter, graph_task},
                timeout=10,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise AssertionError(f'graph did not reach {name}')
            if graph_task in done:
                waiter.cancel()
                await graph_task
                raise AssertionError('graph completed before the event')
            await waiter

        await wait_for_event(statement_slow_started, 'statement slow start')
        await wait_for_event(statement_fast, 'statement fast completion')
        await wait_for_event(procedure_slow_started, 'procedure slow start')
        await wait_for_event(procedure_fast, 'procedure fast completion')
        await wait_for_event(block_started, 'block embed start')

        assert repository.calls == []
        await wait_for_event(block_finished, 'block embedding complete')
        statement_release_slow.set()
        await wait_for_event(
            statement_slow_completed, 'statement slow completion'
        )
        assert repository.calls == []
        procedure_release_slow.set()
        result = await graph_task
        assert order.index('block-embedding') < order.index('statement-slow')
        assert order.index('block-embedding') < order.index('procedure-slow')
        assert order[-1] == 'persistence'
        assert len(repository.calls) == 1
        assert [item.uuid for item in result['statements']] == [
            'statement-fast',
            'statement-slow',
        ]
        assert [item.content for item in result['statements']] == [
            'complete:statement:Fast statement',
            'complete:statement:Delayed statement',
        ]
        assert [item.embedding for item in result['statements']] == [
            [float(len(item.content))] for item in result['statements']
        ]
        assert [item.uuid for item in result['procedures']] == [
            'procedure-fast',
            'procedure-slow',
        ]
        assert [item.content for item in result['procedures']] == [
            'complete:procedure:Fast procedure',
            'complete:procedure:Delayed procedure',
        ]
        assert [item.embedding for item in result['procedures']] == [
            [float(len(item.content))] for item in result['procedures']
        ]
        stored_pages = repository.persisted[1]
        assert [page.index for page in stored_pages] == [0, 1]
        assert [
            (block.uuid, block.content, block.assets)
            for page in stored_pages
            for block in page.blocks
        ] == [
            (block.uuid, block.content, block.assets)
            for page in pages
            for block in page.blocks
        ]
        assert repository.persisted[3] == result['statements']
        assert repository.persisted[4] == result['procedures']

    asyncio.run(exercise())


@pytest.mark.parametrize('failure', ['reconstruction', 'embedding'])
def test_material_failure_does_not_replace_previous_annotations(failure):
    class FailingReconstructor:
        async def acall(self, **_kwargs):
            raise RuntimeError('provider failure')

    class ValidReconstructor:
        async def acall(self, *, items, content_images):
            item = items[0]
            return [
                SourceStatementReconstructionOutput(
                    position=item.position,
                    content=f'complete:{item.request.target_blocks[0].material.text}',
                )
            ]

    class FailingEmbedder:
        async def embed(self, materials, *, content_images, input_type):
            text = materials[0].text or ''
            if failure == 'embedding' and text.startswith('complete:'):
                raise RuntimeError('embedding failure')
            return [[1.0] for _ in materials]

    async def exercise():
        repository = _PersistRecorder()
        repository.persisted = 'previous annotations'
        graph = _reconstruction_test_graph(
            FailingReconstructor()
            if failure == 'reconstruction'
            else ValidReconstructor(),
            _UncalledReconstructor(),
            FailingEmbedder(),
            repository,
        )
        state = {
            'source_uuid': 'source-1',
            'split_pages': [
                SourcePage(
                    index=0,
                    blocks=[
                        SourceBlock(
                            uuid='block-1',
                            block_type='paragraph',
                            content='Statement evidence',
                        )
                    ],
                )
            ],
            'statements': [
                StatementDraft(
                    uuid='statement-1',
                    member_block_uuids=['block-1'],
                    is_exercise=True,
                )
            ],
        }
        with pytest.raises(RuntimeError):
            await graph.ainvoke(state, config={'max_concurrency': 8})
        assert repository.persisted == 'previous annotations'
        assert repository.calls == []

    asyncio.run(exercise())
