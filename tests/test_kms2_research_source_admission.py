"""Deterministic tests for durable researched-source preparation."""

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from PIL import Image

from kms2.core.model.block import SourceBlock
from kms2.core.model.block_types import BlockType
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.admission import (
    ResearchImageMaterial,
    ResearchPDFMaterial,
    ResearchPDFPageMaterial,
    ResearchWebMaterial,
    SourceSnapshotPreparation,
    StagedSourceSnapshot,
)
from kms2.core.model.source_processing.input import (
    OCRSourceInput,
    SourceProcessingInput,
    TextSourceInput,
)
from kms2.core.model.source_processing.ocr import (
    OCRArtifact,
    OCRImageArtifact,
    OCRPageArtifact,
)
from kms2.core.source_snapshot import (
    SourceSnapshotConflict,
    assert_matching_snapshot,
)
from kms2.langgraph.source_admission.graph import SourceAdmissionGraph
from kms2.node.source_admission.finalization import (
    SourceAdmissionFinalizationNode,
)
from kms2.node.source_admission.preparation import (
    SourceAdmissionPreparationNode,
)
from kms2.node.source_admission.processing import SourceAdmissionProcessingNode
from kms2.service.source_admission import SourceAdmissionService


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]


class _Repository:
    def __init__(self) -> None:
        self.load_calls = 0
        self.create_calls = 0

    async def load_owned_snapshot(
        self, user_uuid: str, source_uuid: str
    ) -> None:
        self.load_calls += 1
        return None

    async def create_owned_snapshot(self, user_uuid, source, pages) -> None:
        self.create_calls += 1
        raise AssertionError('source admission must not publish snapshots')


def _png(path: Path, color: str) -> bytes:
    image = Image.new('RGB', (4, 4), color)
    image.save(path, format='PNG')
    image.close()
    return path.read_bytes()


def test_web_capture_preserves_markdown_and_places_relative_figures(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    normalized_png = _png(tmp_path / 'normalized.png', 'red')
    raw_png = _png(tmp_path / 'raw.png', 'blue')
    later_png = _png(tmp_path / 'later.png', 'green')
    material = ResearchWebMaterial(
        url='https://example.test/notes/page',
        resolved_url='https://example.test/notes/page?view=1',
        title='Notes',
        retrieved_at=now,
        raw_markdown=(
            '# Notes\n\nA figure: ![later](figures/z.png) and '
            '![diagram](figures/a.png)\n\nMore text.'
        ),
        images=[
            ResearchImageMaterial(
                url='figures/a.png',
                retrieved_at=now,
                media_bytes=normalized_png,
                original_media_bytes=raw_png,
                original_media_type='image/png',
            ),
            ResearchImageMaterial(
                url='figures/z.png',
                retrieved_at=now,
                media_bytes=later_png,
            ),
        ],
    )
    service = SourceAdmissionService(
        _Repository(), _Counter(), tmp_path / 'store', 8
    )
    preparation = asyncio.run(service.prepare('user-1', material))

    assert isinstance(preparation.input, TextSourceInput)
    page = preparation.input.pages[0]
    assert page.markdown == material.raw_markdown
    assert page.blocks[0].block_type == BlockType.MARKDOWN
    assert '![diagram](figures/a.png)' in page.blocks[0].content
    image_blocks = [
        block for block in page.blocks if block.block_type == BlockType.IMAGE
    ]
    assert len(image_blocks) == 2
    assert [
        Path(block.assets[0].path).read_bytes() for block in image_blocks
    ] == [later_png, normalized_png]
    assert preparation.source.metadata['raw_markdown_path']
    assert preparation.source.metadata['captures']['images'][0]['original_path']
    repeated = asyncio.run(service.prepare('user-1', material))
    assert repeated.source.uuid == preparation.source.uuid

    renamed = asyncio.run(
        service.prepare(
            'user-1', material.model_copy(update={'title': 'Updated'})
        )
    )
    assert renamed.source.uuid == preparation.source.uuid
    assert renamed.source.key == 'Updated'


def test_pdf_capture_keeps_selected_pages_and_real_ocr_artifacts(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    render_path = tmp_path / 'page.png'
    crop_path = tmp_path / 'crop.png'
    figure_path = tmp_path / 'figure.png'
    rendered = _png(render_path, 'white')
    crop = _png(crop_path, 'green')
    figure = _png(figure_path, 'yellow')
    page = ResearchPDFPageMaterial(
        page_index=2,
        markdown='# Selected page\n\nActual OCR text.',
        rendered_page=rendered,
        artifacts=OCRPageArtifact(
            page_index=2,
            markdown='# Selected page\n\nActual OCR text.',
            blocks=[
                OCRArtifact(
                    page_index=2,
                    block_index=0,
                    block_type=BlockType.PARAGRAPH,
                    content='Actual OCR text.',
                    crop_path=str(crop_path),
                    crop_bbox=(0, 0, 2, 2),
                    images=[OCRImageArtifact(path=str(figure_path))],
                )
            ],
        ),
    )
    material = ResearchPDFMaterial(
        url='https://example.test/book.pdf',
        title='Book',
        retrieved_at=now,
        pages=[page],
    )
    service = SourceAdmissionService(
        _Repository(), _Counter(), tmp_path / 'store', 8
    )
    preparation = asyncio.run(service.prepare('user-1', material))

    assert isinstance(preparation.input, OCRSourceInput)
    assert [page.index for page in preparation.input.pages] == [2]
    processed_page = preparation.input.pages[0]
    assert processed_page.markdown == page.markdown
    assert len(processed_page.blocks) == 1
    block = processed_page.blocks[0]
    assert block.content == 'Actual OCR text.'
    assert Path(block.crop_path).read_bytes() == crop
    assert Path(block.assets[0].path).read_bytes() == figure
    assert preparation.source.metadata['selected_page_indexes'] == [2]
    capture = preparation.source.metadata['captures']['pages'][0]
    assert Path(capture['rendered_path']).read_bytes() == rendered
    assert capture['page_index'] == 2


def test_finalized_snapshot_has_stable_child_ids_and_detects_changes(tmp_path):
    source_uuid = str(uuid4())
    asset_path = tmp_path / 'asset.png'
    crop_path = tmp_path / 'crop.png'
    _png(asset_path, 'red')
    _png(crop_path, 'blue')
    source = Source(uuid=source_uuid, key='Original', metadata={'title': 'one'})
    preparation = SourceSnapshotPreparation(
        source=source,
        input=TextSourceInput(pages=[]),
    )
    pages = [
        SourcePage(
            index=7,
            markdown='page',
            blocks=[
                SourceBlock(
                    block_type=BlockType.PARAGRAPH,
                    content='processed text',
                    crop_path=str(crop_path),
                    assets=[{'path': str(asset_path)}],
                )
            ],
        )
    ]
    service = SourceAdmissionService(
        _Repository(), _Counter(), tmp_path / 'store', 8
    )

    first = asyncio.run(service.finalize(preparation, pages))
    second = asyncio.run(
        service.finalize(
            preparation,
            [
                page.model_copy(
                    update={
                        'blocks': [
                            page.blocks[0].model_copy(
                                update={
                                    'uuid': str(uuid4()),
                                    'assets': [
                                        page.blocks[0]
                                        .assets[0]
                                        .model_copy(
                                            update={'uuid': str(uuid4())}
                                        )
                                    ],
                                }
                            )
                        ]
                    }
                )
                for page in pages
            ],
        )
    )

    assert first.pages[0].blocks[0].uuid == second.pages[0].blocks[0].uuid
    assert (
        first.pages[0].blocks[0].assets[0].uuid
        == second.pages[0].blocks[0].assets[0].uuid
    )
    assert first.source.metadata['processed_digest']
    assert (
        Path(first.pages[0].blocks[0].crop_path).parent.parent
        == tmp_path / 'store'
    )
    assert (
        Path(first.pages[0].blocks[0].assets[0].path).parent.parent
        == tmp_path / 'store'
    )
    assert_matching_snapshot(first, second)

    metadata_only = first.model_copy(
        update={
            'source': first.source.model_copy(
                update={'key': 'Renamed', 'metadata': {'retrieved_at': 'later'}}
            )
        }
    )
    assert_matching_snapshot(first, metadata_only)
    changed = asyncio.run(
        service.finalize(
            preparation,
            [
                pages[0].model_copy(
                    update={
                        'blocks': [
                            pages[0]
                            .blocks[0]
                            .model_copy(
                                update={'content': 'changed processed text'}
                            )
                        ]
                    }
                )
            ],
        )
    )
    with pytest.raises(
        SourceSnapshotConflict,
        match='Processed source snapshot changed before publication',
    ):
        assert_matching_snapshot(first, changed)


def test_source_admission_graph_returns_staged_snapshot_without_publication(
    tmp_path,
):
    now = datetime(2026, 1, 1, tzinfo=UTC)
    material = ResearchWebMaterial(
        url='https://example.test/notes',
        title='Notes',
        retrieved_at=now,
        raw_markdown='# Notes\n\nProcessed content.',
    )
    repository = _Repository()
    service = SourceAdmissionService(
        repository, _Counter(), tmp_path / 'store', 8
    )
    pages = [
        SourcePage(
            index=0,
            markdown=material.raw_markdown,
            blocks=[
                SourceBlock(
                    block_type=BlockType.MARKDOWN,
                    content='Processed content.',
                )
            ],
        )
    ]

    class _ProcessingState(TypedDict, total=False):
        source: Source
        input: SourceProcessingInput
        passes: int
        embedded_pages: list[SourcePage]

    processing = StateGraph(_ProcessingState)

    async def advance_processing(
        state: _ProcessingState,
    ) -> dict[str, int | list[SourcePage]]:
        passes = state.get('passes', 0) + 1
        if passes == 8:
            return {'passes': passes, 'embedded_pages': pages}
        return {'passes': passes}

    step_names = [f'processing_{index}' for index in range(8)]
    for step_name in step_names:
        processing.add_node(step_name, advance_processing)
    processing.add_edge(START, step_names[0])
    for index in range(len(step_names) - 1):
        processing.add_edge(step_names[index], step_names[index + 1])
    processing.add_edge(step_names[-1], END)

    graph = SourceAdmissionGraph(
        SourceAdmissionPreparationNode(service),
        SourceAdmissionProcessingNode(processing.compile()),
        SourceAdmissionFinalizationNode(service),
    )

    class _OuterState(TypedDict, total=False):
        admission_result: StagedSourceSnapshot

    outer = StateGraph(_OuterState)

    async def admit(_: _OuterState) -> dict[str, StagedSourceSnapshot]:
        result = await graph.run('user-1', material)
        return {'admission_result': result}

    outer.add_node('source_admission', admit)
    outer.add_edge(START, 'source_admission')
    outer.add_edge('source_admission', END)
    result = asyncio.run(
        outer.compile().ainvoke(
            {},
            config={'recursion_limit': 4},
        )
    )['admission_result']
    assert isinstance(result, StagedSourceSnapshot)
    assert result.pages[0].blocks[0].uuid
    assert repository.load_calls == 1
    assert repository.create_calls == 0
