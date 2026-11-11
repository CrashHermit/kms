import asyncio
import hashlib
from io import BytesIO
from pathlib import Path

import pytest
from langgraph.graph import END, START, StateGraph
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.embedding import EmbeddingRequest
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.context import SourceBlockContext
from kms.source.models.material import (
    Instruction,
    ProcedureDraft,
    StatementDraft,
)
from kms.source.models.source import BlockType, Source, SourceBlock, SourcePage
from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.correction.models import ContentCorrectionOutput
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.exercise_finder.models import (
    ExerciseBoundaryDecision,
    ExerciseClassificationInput,
    ExerciseStartDecision,
)
from kms.source.processing.exercise_finder.node import ExerciseFinderNode
from kms.source.processing.exercise_finder.settings import (
    ExerciseFinderSettings,
)
from kms.source.processing.exercise_splitter.models import (
    SplitDecision,
    SplitPiece,
    SplitRouteDecision,
)
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode
from kms.source.processing.formatting.models import FormattingOutput
from kms.source.processing.formatting.node import FormattingNode
from kms.source.processing.graph import SourceProcessingGraph
from kms.source.processing.image_filter.models import ImageFilterDecision
from kms.source.processing.image_filter.node import ImageFilterNode
from kms.source.processing.image_seam.models import ImageSeamDecision
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.instruction_finder.models import (
    InstructionBoundaryDecision,
    InstructionClassificationInput,
    InstructionStartDecision,
)
from kms.source.processing.instruction_finder.node import InstructionFinderNode
from kms.source.processing.instruction_finder.settings import (
    InstructionFinderSettings,
)
from kms.source.processing.instruction_governance.models import (
    InstructionGovernanceDecision,
    InstructionGovernanceInput,
)
from kms.source.processing.instruction_governance.node import (
    InstructionGovernanceNode,
)
from kms.source.processing.instruction_governance.settings import (
    InstructionGovernanceSettings,
)
from kms.source.processing.ocr import OCRNode
from kms.source.processing.ocr_models import OCRArtifact, OCRPageArtifact
from kms.source.processing.pedagogical_finder.models import (
    PedagogicalBoundaryDecision,
    PedagogicalClassificationInput,
    PedagogicalStartDecision,
)
from kms.source.processing.pedagogical_finder.node import PedagogicalFinderNode
from kms.source.processing.pedagogical_finder.settings import (
    PedagogicalFinderSettings,
)
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
from kms.source.processing.statement_procedure.models import (
    PedagogicalRoleDecision,
    ProcedurePartitionDecision,
    StatementPartitionDecision,
)
from kms.source.processing.statement_procedure.node import (
    StatementProcedurePartitionNode,
)
from kms.source.processing.statement_procedure.settings import (
    StatementProcedureSettings,
)
from kms.source.processing.statement_reconstruction.embedding import (
    SourceStatementEmbeddingNode,
)
from kms.source.processing.statement_reconstruction.models import (
    SourceStatementReconstructionOutput,
)
from kms.source.processing.statement_reconstruction.node import (
    SourceStatementReconstructionNode,
)
from kms.source.processing.text_seam.models import TextSeamDecision
from kms.source.processing.text_seam.node import TextSeamNode


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _ContentCorrector:
    def __init__(self):
        self.calls = 0

    async def acall(self, *, blocks):
        self.calls += 1
        return [
            ContentCorrectionOutput(
                position=block.position,
                keep=True,
                corrected_content=block.material.text,
                cleanup_instructions=[],
            )
            for block in blocks
        ]


class _Formatter:
    def __init__(self):
        self.calls = 0

    async def acall(self, *, blocks):
        self.calls += 1
        return [
            FormattingOutput(
                position=block.position,
                formatted_content=block.content,
            )
            for block in blocks
        ]


class _ImageFilter:
    def __init__(self):
        self.calls = 0

    async def acall(self, *, items):
        self.calls += 1
        return [
            ImageFilterDecision(position=item.position, keep=True)
            for item in items
        ]


class _SeamJudges:
    def __init__(self):
        self.text_calls = 0
        self.image_calls = 0

    async def acall(self, *, items, **_kwargs):
        if hasattr(items[0], 'tail'):
            self.text_calls += 1
            return [
                TextSeamDecision(position=item.position, is_split=False)
                for item in items
            ]
        self.image_calls += 1
        return [
            ImageSeamDecision(position=item.position, is_continuation=False)
            for item in items
        ]


class _PackedRouter:
    async def acall(self, *, targets, **_kwargs):
        return [
            SplitRouteDecision(
                position=target.position,
                contains_multiple_exercises=(
                    target.source_block.block_type != BlockType.IMAGE
                    and '1.' in target.source_block.material.text
                    and '2.' in target.source_block.material.text
                ),
            )
            for target in targets
        ]


class _PackedSplitter:
    async def acall(self, *, candidates, **_kwargs):
        return [
            SplitDecision(
                position=candidate.position,
                pieces=[
                    SplitPiece(content='1. $f(x)=x^2$.'),
                    SplitPiece(content='2. $g(x)=x^3$.'),
                ],
            )
            for candidate in candidates
        ]


class _InstructionStart:
    async def acall(
        self,
        *,
        items: list[InstructionClassificationInput],
    ) -> list[InstructionStartDecision]:
        return [
            InstructionStartDecision(
                position=item.position,
                is_instruction_start=(
                    (item.target_block.material.text or '').startswith(
                        'For Exercises'
                    )
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
    ) -> list[InstructionBoundaryDecision]:
        return [
            InstructionBoundaryDecision(
                position=item.position,
                is_instruction_boundary=(
                    (item.target_block.material.text or '').startswith('1.')
                ),
            )
            for item in reversed(items)
        ]


class _ExerciseStart:
    async def acall(
        self,
        *,
        items: list[ExerciseClassificationInput],
    ) -> list[ExerciseStartDecision]:
        return [
            ExerciseStartDecision(
                position=item.position,
                is_exercise_start=(
                    (item.target_block.material.text or '').startswith('1.')
                    or (item.target_block.material.text or '').startswith('2.')
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
    ) -> list[ExerciseBoundaryDecision]:
        return [
            ExerciseBoundaryDecision(
                position=item.position,
                is_exercise_boundary=(
                    (item.target_block.material.text or '').startswith('2.')
                    or (item.target_block.material.text or '').startswith(
                        'Example.'
                    )
                ),
            )
            for item in reversed(items)
        ]


class _PedagogicalStart:
    async def acall(
        self,
        *,
        items: list[PedagogicalClassificationInput],
    ) -> list[PedagogicalStartDecision]:
        return [
            PedagogicalStartDecision(
                position=item.position,
                is_pedagogical_start=(
                    (item.target_block.material.text or '').startswith(
                        'Example.'
                    )
                ),
            )
            for item in reversed(items)
        ]


class _PedagogicalBoundary:
    async def acall(
        self,
        *,
        start_block: SourceBlockContext,
        items: list[PedagogicalClassificationInput],
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
    ) -> list[InstructionGovernanceDecision]:
        return [
            InstructionGovernanceDecision(
                position=item.position,
                governs=' '.join(
                    block.material.text or '' for block in item.statement_blocks
                ).startswith(('1.', '2.')),
            )
            for item in reversed(items)
        ]


class _FakeEmbedder:
    def __init__(self):
        self.image_counts = []
        self.image_paths = []
        self.image_digests = []

    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        assert request.input_type == 'document'
        image_materials = [
            material for material in request.materials if material.images
        ]
        self.image_counts.append(
            sum(len(material.images) for material in request.materials)
        )
        self.image_paths.append(
            [
                asset.payload
                for material in image_materials
                for asset in material.images
            ]
        )
        self.image_digests.extend(
            hashlib.sha256(asset.payload.read_bytes()).hexdigest()
            for material in image_materials
            for asset in material.images
        )
        return [
            [float(position)] for position, _ in enumerate(request.materials)
        ]


class _StatementReconstructor:
    async def acall(self, *, items):
        return [
            SourceStatementReconstructionOutput(
                position=item.position,
                content=' '.join(
                    [
                        *(
                            block.material.text
                            for block in item.request.target_blocks
                            if block.material.text is not None
                        ),
                        *(
                            block.material.text
                            for block in item.request.instruction_blocks
                            if block.material.text is not None
                        ),
                    ]
                ),
            )
            for item in reversed(items)
        ]


class _RoleTyper:
    async def acall(self, *, items):
        return [
            PedagogicalRoleDecision(
                position=item.position,
                has_statement=True,
                has_procedure=True,
            )
            for item in items
        ]


class _StatementPartitioner:
    async def acall(self, *, items):
        return [
            StatementPartitionDecision(
                position=item.position,
                statement_positions=[0],
            )
            for item in items
        ]


class _ProcedurePartitioner:
    async def acall(self, *, items):
        return [
            ProcedurePartitionDecision(
                position=item.position,
                procedure_positions=[1, 2],
            )
            for item in items
        ]


class _ProcedureReconstructor:
    def __init__(self):
        self.image_digests = []

    async def acall(self, *, items):
        image_materials = [
            block.material
            for item in items
            for block in item.request.target_blocks
            if block.material.images
        ]
        for material in image_materials:
            for asset in material.images:
                image_bytes = asset.payload.read_bytes()
                self.image_digests.append(
                    hashlib.sha256(image_bytes).hexdigest()
                )
                with Image.open(BytesIO(image_bytes)) as image:
                    assert image.getpixel((0, 0)) == (255, 0, 0)

        return [
            SourceProcedureReconstructionOutput(
                position=item.position,
                content=' '.join(
                    [
                        *(
                            block.material.text
                            for block in item.request.target_blocks
                            if block.material.text is not None
                        ),
                        *(
                            'The source figure accompanies the worked calculation.'
                            for block in item.request.target_blocks
                            if block.material.images
                        ),
                    ]
                ),
            )
            for item in reversed(items)
        ]


class _UncalledReconstructor:
    async def acall(self, **_kwargs):
        raise AssertionError('empty procedure path must not invoke inference')


def test_merged_source_graph_processes_and_embeds_complete_multimodal_snapshot(
    tmp_path,
):
    image_path = tmp_path / 'source-image.png'
    Image.new('RGB', (2, 2), 'red').save(image_path, format='PNG')
    image_digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    pages = [
        SourcePage(
            index=2,
            markdown='Exercises',
            blocks=[
                SourceBlock(
                    uuid='instruction-block',
                    block_type='instruction',
                    content='For Exercises 1 and 2, differentiate with respect to x.',
                ),
                SourceBlock(
                    uuid='packed-block',
                    block_type='paragraph',
                    content='1. $f(x)=x^2$.\n2. $g(x)=x^3$.\n',
                ),
            ],
        ),
        SourcePage(
            index=0,
            markdown='Worked example',
            blocks=[
                SourceBlock(
                    uuid='example-block',
                    block_type='paragraph',
                    content='Example. Differentiate $h(x)=x^2$.',
                ),
                SourceBlock(
                    uuid='procedure-block',
                    block_type='paragraph',
                    content="$h'(x)=2x$.",
                ),
                SourceBlock(
                    uuid='image-block',
                    block_type='image',
                    content=None,
                    image_path=str(image_path),
                    image_page_indexes=[0],
                ),
            ],
        ),
    ]
    artifacts = [
        OCRPageArtifact(
            page_index=page.index,
            page_image_path=str(image_path),
            markdown=page.markdown,
            blocks=[
                OCRArtifact(
                    page_index=page.index,
                    block_index=position,
                    block_type=block.block_type,
                    content=block.content,
                    crop_path=block.image_path or str(image_path),
                    crop_bbox=block.crop_bbox or (0, 0, 2, 2),
                )
                for position, block in enumerate(page.blocks)
            ],
        )
        for page in pages
    ]

    class _OCRProvider:
        expected_pages = [2, 0]

        def extract(self, _pdf_path, *, source_uuid, pages=None):
            assert source_uuid == '12345678-1234-5678-1234-567812345678'
            assert pages == self.expected_pages
            return self.artifacts

    provider = _OCRProvider()
    provider.artifacts = artifacts
    ocr = OCRNode(provider)
    counter = _Counter()
    budget = TokenBudget(MaterialTokenEstimator(counter), 100_000)
    token_counters = (MaterialTokenEstimator(counter),)
    context_window = ContextWindowSettings(backward_budget=0, forward_budget=0)
    embedder = _FakeEmbedder()

    image_budget = ImagePayloadBudget(4, 26_214_400, 40_000_000)
    corrector = _ContentCorrector()
    formatter = _Formatter()
    image_filter = _ImageFilter()
    seam_models = _SeamJudges()
    procedure_reconstructor = _ProcedureReconstructor()
    graph = SourceProcessingGraph(
        ocr=ocr,
        content_correction=ContentCorrectionNode(
            corrector,
            budget=budget,
            image_budget=image_budget,
            batch_size=16,
        ),
        formatter=FormattingNode(formatter, budget=budget, batch_size=16),
        text_seam=TextSeamNode(
            seam_models,
            seam_models,
            judge_budget=budget,
            rewrite_budget=budget,
            judge_batch_size=16,
            rewrite_batch_size=16,
        ),
        image_seam=ImageSeamNode(
            seam_models,
            budget=budget,
            image_budget=image_budget,
            batch_size=16,
        ),
        image_filter=ImageFilterNode(
            image_filter,
            context_window,
            budget=budget,
            image_budget=image_budget,
            batch_size=16,
            token_counters=token_counters,
        ),
        exercise_splitter=ExerciseSplitterNode(
            _PackedRouter(),
            _PackedSplitter(),
            context_window,
            batch_size=16,
            token_counters=token_counters,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
            router_budget=budget,
            splitter_budget=budget,
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
        statement_procedure=StatementProcedurePartitionNode(
            _RoleTyper(),
            _StatementPartitioner(),
            _ProcedurePartitioner(),
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
            procedure_reconstructor,
            budget=budget,
            batch_size=16,
            image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
        ),
        source_statement_embedding=SourceStatementEmbeddingNode(embedder),
        source_procedure_embedding=SourceProcedureEmbeddingNode(embedder),
        embedding=SourceBlockEmbeddingNode(embedder, model='test', dimension=1),
    )

    result = asyncio.run(
        graph.run(
            SourceProcessingState(
                input={
                    'kind': 'pdf',
                    'pdf_path': 'capture.pdf',
                    'pages': [2, 0],
                },
                source=Source(
                    uuid='12345678-1234-5678-1234-567812345678',
                    key='capture.pdf',
                ),
            )
        )
    )
    assert corrector.calls > 0
    assert formatter.calls > 0
    assert image_filter.calls > 0

    final_pages = result.split_pages
    assert [page.index for page in final_pages] == [2, 0]
    assert [block.content for block in final_pages[0].blocks] == [
        'For Exercises 1 and 2, differentiate with respect to x.',
        '1. $f(x)=x^2$.',
        '2. $g(x)=x^3$.',
    ]
    assert [block.content for block in final_pages[1].blocks] == [
        'Example. Differentiate $h(x)=x^2$.',
        "$h'(x)=2x$.",
        None,
    ]
    assert len(result.statements) == 3
    exercise_statements = [
        statement for statement in result.statements if statement.is_exercise
    ]
    assert len(exercise_statements) == 2
    assert len(result.instructions) == 1
    assert result.instructions[0].governed_statement_uuids == [
        statement.uuid for statement in exercise_statements
    ]
    assert len(result.procedures) == 1
    procedure_members = result.procedures[0].member_block_uuids
    image_block = final_pages[1].blocks[2]
    assert image_block.content is None
    assert image_block.image_path == str(image_path)
    assert image_block.image_page_indexes == [0]
    assert (
        hashlib.sha256(Path(image_block.image_path).read_bytes()).hexdigest()
        == image_digest
    )
    assert image_block.uuid in procedure_members
    assert procedure_reconstructor.image_digests == [image_digest]
    assert embedder.image_digests.count(image_digest) == 2
    assert [
        block.content
        for page in final_pages
        for block in page.blocks
        if block.uuid in procedure_members
    ] == ["$h'(x)=2x$.", None]
    assert result.procedures[0].content == (
        "$h'(x)=2x$. The source figure accompanies the worked calculation."
    )
    assert embedder.image_paths.count([image_path]) == 2
    assert sum(len(page.blocks) for page in final_pages) == 6
    assert all(
        block.embedding is not None
        for page in final_pages
        for block in page.blocks
    )
    assert seam_models.text_calls == 0
    assert seam_models.image_calls == 0

    adjacent_pages = [
        SourcePage(
            index=2,
            markdown='Exercises',
            blocks=[
                SourceBlock(
                    uuid='adjacent-instruction',
                    block_type='instruction',
                    content='For Exercises 1 and 2, differentiate with respect to x.',
                ),
                SourceBlock(
                    uuid='adjacent-packed',
                    block_type='paragraph',
                    content='1. $f(x)=x^2$.\n2. $g(x)=x^3$.\n',
                ),
                SourceBlock(
                    uuid='image-2',
                    block_type='image',
                    content='',
                    image_path=str(image_path),
                    image_page_indexes=[2],
                ),
            ],
        ),
        SourcePage(
            index=3,
            markdown='Worked example',
            blocks=[
                SourceBlock(
                    uuid='image-3-first',
                    block_type='image',
                    content='',
                    image_path=str(image_path),
                    image_page_indexes=[3],
                ),
                SourceBlock(
                    uuid='adjacent-example',
                    block_type='paragraph',
                    content='Example. Differentiate $h(x)=x^2$.',
                ),
                SourceBlock(
                    uuid='adjacent-procedure',
                    block_type='paragraph',
                    content="$h'(x)=2x$.",
                ),
                SourceBlock(
                    uuid='image-3-last',
                    block_type='image',
                    content='',
                    image_path=str(image_path),
                    image_page_indexes=[3],
                ),
            ],
        ),
        SourcePage(
            index=4,
            markdown='Conclusion',
            blocks=[
                SourceBlock(
                    uuid='image-4',
                    block_type='image',
                    content='',
                    image_path=str(image_path),
                    image_page_indexes=[4],
                ),
                SourceBlock(
                    uuid='adjacent-conclusion',
                    block_type='paragraph',
                    content='A concluding note.',
                ),
            ],
        ),
    ]
    provider.expected_pages = [2, 3, 4]
    provider.artifacts = [
        OCRPageArtifact(
            page_index=page.index,
            page_image_path=str(image_path),
            markdown=page.markdown,
            blocks=[
                OCRArtifact(
                    page_index=page.index,
                    block_index=position,
                    block_type=block.block_type,
                    content=block.content,
                    crop_path=block.image_path or str(image_path),
                    crop_bbox=(0, 0, 2, 2),
                )
                for position, block in enumerate(page.blocks)
            ],
        )
        for page in adjacent_pages
    ]
    adjacent_result = asyncio.run(
        graph.run(
            SourceProcessingState(
                input={
                    'kind': 'pdf',
                    'pdf_path': 'capture.pdf',
                    'pages': [2, 3, 4],
                },
                source=Source(
                    uuid='12345678-1234-5678-1234-567812345678',
                    key='capture.pdf',
                ),
            )
        )
    )
    assert [page.index for page in adjacent_result.split_pages] == [2, 3, 4]
    assert seam_models.text_calls == 2
    assert seam_models.image_calls == 1
    assert adjacent_result.statements
    assert adjacent_result.procedures


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

    async def acall(self, *, items):
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

    async def embed(self, request: EmbeddingRequest):
        texts = [material.text or '' for material in request.materials]
        if texts[0].startswith('complete:'):
            return [[float(len(text))] for text in texts]
        assert texts[0] == 'Shared directive'
        self.block_started.set()
        self.order.append('block-embedding')
        self.block_finished.set()
        return [[float(len(text))] for text in texts]


def _reconstruction_test_graph(statement_module, procedure_module, embedder):
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100_000)
    graph = StateGraph(SourceProcessingState)
    graph.add_node('instruction_governance', lambda _state: {})
    graph.add_edge(START, 'instruction_governance')
    statement_node = SourceStatementReconstructionNode(
        statement_module,
        budget=budget,
        batch_size=1,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )
    procedure_node = SourceProcedureReconstructionNode(
        procedure_module,
        budget=budget,
        batch_size=1,
        image_budget=ImagePayloadBudget(4, 26_214_400, 40_000_000),
    )
    for stage, node, embedding_stage in (
        (
            'source_statement_reconstruction',
            statement_node,
            'source_statement_embedding',
        ),
        (
            'source_procedure_reconstruction',
            procedure_node,
            'source_procedure_embedding',
        ),
    ):
        worker = f'{stage}_worker'
        collect = f'{stage}_collect'
        graph.add_node(stage, node.run)
        graph.add_node(worker, node.worker)
        graph.add_node(collect, node.collect)
        graph.add_edge('instruction_governance', stage)
        graph.add_conditional_edges(
            stage,
            node.dispatch,
            [worker, collect],
        )
        graph.add_edge(worker, collect)
        graph.add_edge(collect, embedding_stage)
    graph.add_node(
        'source_statement_embedding',
        SourceStatementEmbeddingNode(embedder).run,
    )
    graph.add_node(
        'source_procedure_embedding',
        SourceProcedureEmbeddingNode(embedder).run,
    )
    graph.add_node(
        'embedding',
        SourceBlockEmbeddingNode(embedder, model='test', dimension=1).run,
    )
    graph.add_edge(
        ['source_statement_embedding', 'source_procedure_embedding'],
        'embedding',
    )
    graph.add_edge('embedding', END)
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
                        block_type='image',
                        content='Fast procedure',
                        image_path=str(figure_path),
                        image_page_indexes=[1],
                    ),
                ],
            ),
        ]
        graph = _reconstruction_test_graph(
            statement_module,
            procedure_module,
            embedder,
        )
        initial_state = SourceProcessingState(
            input={'kind': 'pdf', 'pdf_path': 'capture.pdf'},
            source=Source(uuid='source-1', key='capture.pdf'),
            split_pages=pages,
            instructions=[
                Instruction(
                    uuid='instruction-1',
                    member_block_uuids=['block-directive'],
                    governed_statement_uuids=[
                        'statement-slow',
                        'statement-fast',
                    ],
                )
            ],
            statements=[
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
            procedures=[
                ProcedureDraft(
                    uuid='procedure-fast',
                    member_block_uuids=['block-procedure-fast'],
                ),
                ProcedureDraft(
                    uuid='procedure-slow',
                    member_block_uuids=['block-procedure-slow'],
                ),
            ],
        )
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
        assert not block_started.is_set()
        statement_release_slow.set()
        procedure_release_slow.set()
        await wait_for_event(
            statement_slow_completed, 'statement slow completion'
        )
        await wait_for_event(
            procedure_slow_completed, 'procedure slow completion'
        )
        await wait_for_event(block_started, 'block embedding start')
        await wait_for_event(block_finished, 'block embedding complete')
        result = await graph_task
        assert order.index('statement-slow') < order.index('block-embedding')
        assert order.index('procedure-slow') < order.index('block-embedding')
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
        embedded_pages = result['split_pages']
        assert [page.index for page in embedded_pages] == [0, 1]
        embedded_blocks = [
            block for page in embedded_pages for block in page.blocks
        ]
        assert [
            (block.uuid, block.content, block.image_path)
            for block in embedded_blocks
        ] == [
            (block.uuid, block.content, block.image_path)
            for page in pages
            for block in page.blocks
        ]

    asyncio.run(exercise())


@pytest.mark.parametrize('failure', ['reconstruction', 'embedding'])
def test_material_failures_propagate_without_mutating_drafts(failure):
    class FailingReconstructor:
        async def acall(self, **_kwargs):
            raise RuntimeError('provider failure')

    class ValidReconstructor:
        async def acall(self, *, items):
            item = items[0]
            return [
                SourceStatementReconstructionOutput(
                    position=item.position,
                    content=f'complete:{item.request.target_blocks[0].material.text}',
                )
            ]

    class FailingEmbedder:
        async def embed(self, request: EmbeddingRequest):
            text = request.materials[0].text or ''
            if failure == 'embedding' and text.startswith('complete:'):
                raise RuntimeError('embedding failure')
            return [[1.0] for _ in request.materials]

    async def exercise():
        graph = _reconstruction_test_graph(
            FailingReconstructor()
            if failure == 'reconstruction'
            else ValidReconstructor(),
            _UncalledReconstructor(),
            FailingEmbedder(),
        )
        state = SourceProcessingState(
            input={'kind': 'pdf', 'pdf_path': 'capture.pdf'},
            source=Source(uuid='source-1', key='capture.pdf'),
            split_pages=[
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
            statements=[
                StatementDraft(
                    uuid='statement-1',
                    member_block_uuids=['block-1'],
                    is_exercise=True,
                )
            ],
        )
        with pytest.raises(RuntimeError):
            await graph.ainvoke(state, config={'max_concurrency': 8})
        assert state.statements[0].content == ''

    asyncio.run(exercise())
