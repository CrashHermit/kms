from __future__ import annotations

import asyncio

from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.embedding import EmbeddingRequest
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.inputs import PDFSourceInput
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.exercise_finder.node import ExerciseFinderNode
from kms.source.processing.exercise_finder.settings import (
    ExerciseFinderSettings,
)
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode
from kms.source.processing.formatting.node import FormattingNode
from kms.source.processing.graph import SourceProcessingGraph
from kms.source.processing.image_filter.models import ImageFilterDecision
from kms.source.processing.image_filter.node import ImageFilterNode
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.instruction_redistribution.node import (
    InstructionRedistributionNode,
)
from kms.source.processing.instruction_redistribution.settings import (
    InstructionRedistributionSettings,
)
from kms.source.processing.ocr import OCRNode
from kms.source.processing.ocr_models import OCRArtifact, OCRExtraction
from kms.source.processing.procedure_finder.node import ProcedureFinderNode
from kms.source.processing.procedure_finder.settings import (
    ProcedureFinderSettings,
)
from kms.source.processing.procedure_splitter.node import ProcedureSplitterNode
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.text_seam.node import TextSeamNode
from kms.source.processing.unit_assembly import SourceUnitAssemblyNode
from kms.source.processing.unit_embedding import SourceUnitEmbeddingNode


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Provider:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path

    def extract(self, pdf_path, *, source_uuid, pages=None):
        return OCRExtraction(
            blocks=[
                OCRArtifact(
                    page_index=0,
                    block_index=0,
                    block_type=BlockType.IMAGE,
                    crop_path=self.image_path,
                    crop_bbox=(0, 0, 8, 8),
                )
            ],
            page_indexes=[0],
            page_images={0: self.image_path},
        )


class _EmptyClassifier:
    async def acall(self, **kwargs):
        return []


class _KeepImage:
    async def acall(self, *, items):
        return [
            ImageFilterDecision(position=item.position, keep=True)
            for item in items
        ]


class _Embedder:
    async def embed(self, request: EmbeddingRequest) -> list[list[float]]:
        return [[1.0, 2.0] for _ in request.materials]


def test_real_processing_graph_publishes_one_image_only_content_unit(tmp_path):
    page_path = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), 'blue').save(page_path)
    counter = MaterialTokenEstimator(_Counter())
    budget = TokenBudget(counter=counter, token_limit=100_000)
    image_budget = ImagePayloadBudget(8, 26_214_400, 40_000_000)
    context = ContextWindowSettings(
        backward_budget=100, forward_budget=100, target_budget=0
    )
    classifier = _EmptyClassifier()
    embedder = _Embedder()

    graph = SourceProcessingGraph(
        OCRNode(_Provider(str(page_path))),
        ContentCorrectionNode(
            object(), budget=budget, image_budget=image_budget, batch_size=8
        ),
        FormattingNode(object(), budget=budget, batch_size=8),
        TextSeamNode(
            classifier,
            classifier,
            judge_budget=budget,
            rewrite_budget=budget,
            judge_batch_size=8,
            rewrite_batch_size=8,
        ),
        ImageSeamNode(
            classifier, budget=budget, image_budget=image_budget, batch_size=8
        ),
        ImageFilterNode(
            _KeepImage(),
            context,
            budget=budget,
            image_budget=image_budget,
            batch_size=8,
            token_counters=(counter,),
        ),
        ExerciseSplitterNode(
            classifier,
            classifier,
            context,
            batch_size=8,
            token_counters=(counter,),
            image_budget=image_budget,
            router_budget=budget,
            splitter_budget=budget,
        ),
        ProcedureSplitterNode(
            classifier,
            classifier,
            context,
            batch_size=8,
            token_counters=(counter,),
            image_budget=image_budget,
            router_budget=budget,
            splitter_budget=budget,
        ),
        InstructionRedistributionNode(
            classifier,
            classifier,
            classifier,
            InstructionRedistributionSettings(),
            start_budget=budget,
            boundary_budget=budget,
            applicability_budget=budget,
            start_token_counters=(counter,),
            boundary_token_counters=(counter,),
            applicability_token_counters=(counter,),
            image_budget=image_budget,
            applicability_image_budget=image_budget,
        ),
        ExerciseFinderNode(
            classifier,
            classifier,
            ExerciseFinderSettings(),
            start_budget=budget,
            boundary_budget=budget,
            start_token_counters=(counter,),
            boundary_token_counters=(counter,),
            image_budget=image_budget,
        ),
        ProcedureFinderNode(
            classifier,
            classifier,
            ProcedureFinderSettings(),
            start_budget=budget,
            boundary_budget=budget,
            start_token_counters=(counter,),
            boundary_token_counters=(counter,),
            image_budget=image_budget,
        ),
        SourceUnitAssemblyNode(),
        SourceBlockEmbeddingNode(embedder, model='image-test', dimension=2),
        SourceUnitEmbeddingNode(embedder),
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(
            uuid='12345678-1234-5678-1234-567812345678',
            key='source.pdf',
        ),
    )

    result = asyncio.run(graph.run(state))

    assert result.selected_page_indexes == [0]
    assert len(result.units) == 1
    assert result.units[0].kind is SourceUnitKind.CONTENT
    assert len(result.units[0].blocks) == 1
    block = result.units[0].blocks[0]
    assert block.block_type is BlockType.IMAGE
    assert block.embedding == [1.0, 2.0]
    assert result.units[0].embedding == block.embedding


def test_assembly_orders_units_by_source_anchor_and_adds_singleton_content():

    blocks = [
        SourceBlock(
            uuid=f'b{index}',
            block_type=BlockType.PARAGRAPH,
            content=f'block {index}',
        )
        for index in range(5)
    ]
    exercise = SourceUnit(kind=SourceUnitKind.EXERCISE, blocks=[blocks[2]])
    procedure = SourceUnit(
        kind=SourceUnitKind.PROCEDURE, blocks=[blocks[3], blocks[4]]
    )
    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='assembly', key='source.pdf'),
        procedure_split_blocks=blocks,
        exercise_units=[exercise],
        procedure_units=[procedure],
    )

    units = asyncio.run(SourceUnitAssemblyNode().run(state))['units']

    assert [
        (unit.kind, [block.uuid for block in unit.blocks]) for unit in units
    ] == [
        (SourceUnitKind.CONTENT, ['b0']),
        (SourceUnitKind.CONTENT, ['b1']),
        (SourceUnitKind.EXERCISE, ['b2']),
        (SourceUnitKind.PROCEDURE, ['b3', 'b4']),
    ]


def test_empty_stream_has_no_content_unit():

    state = SourceProcessingState(
        input=PDFSourceInput(pdf_path='source.pdf'),
        source=Source(uuid='empty', key='source.pdf'),
    )

    assert asyncio.run(SourceUnitAssemblyNode().run(state))['units'] == []
