import asyncio
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from PIL import Image

from kms.config.inference import ContextWindowSettings
from kms.core.content import ContentImage
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.content import MultimodalContent
from kms.core.model.page import SourcePage
from kms.core.model.source import Source
from kms.core.model.source_processing.content_correction import (
    ContentCorrectionOutput,
    ContentCorrectionResult,
)
from kms.core.model.source_processing.formatting import (
    FormattingInput,
    FormattingOutput,
)
from kms.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionOutput,
)
from kms.core.model.source_processing.image_filter import ImageFilterDecision
from kms.core.model.source_processing.image_seam import (
    ImageSeamDecision,
    ImageSeamInput,
)
from kms.core.model.source_processing.ocr import (
    OCRArtifact,
    OCRImageArtifact,
    OCRPageArtifact,
)
from kms.core.model.source_processing.text_seam import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRewriteOutput,
)
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database.source.source_graph_records import source_graph_parameters
from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.langgraph.source_processing.graph import SourceProcessingGraph
from kms.langgraph.source_processing.state import SourceProcessingState
from kms.node.source_processing.content_correction import ContentCorrectionNode
from kms.node.source_processing.embedding import EmbeddingNode
from kms.node.source_processing.formatting import FormattingNode
from kms.node.source_processing.image_description import ImageDescriptionNode
from kms.node.source_processing.image_filter import ImageFilterNode
from kms.node.source_processing.image_seam import ImageSeamNode
from kms.node.source_processing.ocr import OCRNode
from kms.node.source_processing.persistence import SourcePersistenceNode
from kms.node.source_processing.text_seam import TextSeamNode
from kms.ocr.provider import OCRProvider
from kms.service.source_artifacts import SourceArtifactStore


def test_source_graph_cleans_filters_persists_and_reloads_survivors(tmp_path):
    crop = tmp_path / 'crop.png'
    Image.new('RGB', (2, 2), 'red').save(crop)
    rejected_image = tmp_path / 'rejected.png'
    Image.new('RGB', (3, 3), 'blue').save(rejected_image)
    retained_image = tmp_path / 'retained.png'
    Image.new('RGB', (4, 4), 'green').save(retained_image)
    original_page = tmp_path / 'page.png'
    Image.new('RGB', (8, 8), 'white').save(original_page)
    page = OCRPageArtifact(
        page_index=7,
        page_image_path=str(original_page),
        markdown='captured page markdown',
        blocks=[
            OCRArtifact(
                page_index=7,
                block_index=0,
                block_type='paragraph',
                content='Image control',
                crop_path=str(crop),
                crop_bbox=(0, 0, 2, 2),
            ),
            OCRArtifact(
                page_index=7,
                block_index=1,
                block_type='paragraph',
                content='x^2 Advertisement footer https://example.com/keep',
                crop_path=str(crop),
                crop_bbox=(2, 0, 4, 2),
            ),
            OCRArtifact(
                page_index=7,
                block_index=2,
                block_type='image',
                images=[OCRImageArtifact(path=str(rejected_image))],
                crop_bbox=(4, 0, 7, 3),
            ),
            OCRArtifact(
                page_index=7,
                block_index=3,
                block_type='image',
                images=[OCRImageArtifact(path=str(retained_image))],
                crop_bbox=(0, 3, 4, 7),
            ),
        ],
    )

    class FakeProvider:
        def extract(self, pdf_path, *, source_uuid, pages=None):
            return [page]

    class Corrector:
        async def acall(self, *, blocks, content_images):
            return [
                ContentCorrectionOutput(
                    position=block.position,
                    keep=block.material.text != 'Image control',
                    corrected_content=(
                        None
                        if block.material.text == 'Image control'
                        else block.material.text
                    ),
                    cleanup_instructions=(
                        ["Remove the 'Advertisement footer'."]
                        if 'Advertisement footer' in block.material.text
                        else []
                    ),
                )
                for block in blocks
            ]

    formatted = []

    class Formatter:
        async def acall(self, *, blocks):
            assert blocks[0].cleanup_instructions == [
                "Remove the 'Advertisement footer'."
            ]
            formatted.extend(blocks)
            return [
                FormattingOutput(
                    position=block.position,
                    formatted_content=block.content.replace(
                        'Advertisement footer ', ''
                    ),
                )
                for block in blocks
            ]

    class TextJudge:
        async def acall(self, *, items):
            return [
                TextSeamDecision(position=item.position, is_split=True)
                for item in items
            ]

    class TextRewriter:
        async def acall(self, *, items):
            raise AssertionError('no text seam candidates')

    class ImageJudge:
        async def acall(self, *, items, content_images):
            raise AssertionError('no image seam candidates')

    class Filter:
        async def acall(self, *, items, content_images):
            return [
                ImageFilterDecision(
                    position=item.position, keep=item.position == 1
                )
                for item in items
            ]

    class Describer:
        async def acall(self, *, items, content_images):
            return [
                ImageDescriptionOutput(
                    position=item.position, description='retained figure'
                )
                for item in items
            ]

    class MemoryRepository:
        """In-memory adapter around the production graph serialize/hydrate contract."""

        def __init__(self):
            self.pages = []

        async def replace_source(self, source, pages):
            parameters = source_graph_parameters(source, pages)
            assets = {item['uuid']: item for item in parameters['assets']}
            blocks = {item['uuid']: item for item in parameters['blocks']}
            assets_by_block = {}
            for pair in parameters['block_asset_pairs']:
                assets_by_block.setdefault(pair['block_uuid'], []).append(
                    assets[pair['asset_uuid']]
                )
            blocks_by_page = {}
            for pair in parameters['page_block_pairs']:
                block = blocks[pair['block_uuid']]
                blocks_by_page.setdefault(pair['page_index'], []).append(
                    {**block, 'assets': assets_by_block.get(block['uuid'], [])}
                )
            self.pages = [
                {
                    **page,
                    'blocks': blocks_by_page.get(page['index'], []),
                }
                for page in parameters['pages']
            ]
            return source

        async def load_pages(self, source_uuid):
            class Result:
                def __init__(self, pages):
                    self.pages = pages

                async def single(self, *, strict=False):
                    assert strict
                    return {'pages': self.pages}

            class Session:
                async def run(self, query, **parameters):
                    return Result(repository.pages)

            class SessionContext:
                async def __aenter__(self):
                    return Session()

                async def __aexit__(self, *args):
                    return None

            # Load through the production repository read/hydration path.
            return await SourceGraphRepository(
                lambda: SessionContext()
            ).load_pages(source_uuid)

    repository = MemoryRepository()
    persistence = SourcePersistenceNode(
        repository, SourceArtifactStore(tmp_path / 'durable')
    )
    graph = SourceProcessingGraph(
        OCRNode(FakeProvider()),
        _content_correction_node(Corrector()),
        _formatting_node(Formatter()),
        _text_seam_node(TextJudge(), TextRewriter()),
        _image_seam_node(ImageJudge()),
        _image_filter_node(Filter()),
        ImageDescriptionNode(
            Describer(),
            ContextWindowSettings(backward_budget=100, forward_budget=100),
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 10_000),
            image_budget=ImagePayloadBudget(
                max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
            ),
            batch_size=4,
            token_counters=(MaterialTokenEstimator(_Counter()),),
        ),
        _embedding_node(),
        persistence,
    )

    async def run_and_reload():
        result = await graph.run(
            SourceProcessingState(
                input={'kind': 'pdf', 'pdf_path': 'capture.pdf'},
                source=Source(uuid='source-1', key='capture.pdf'),
            )
        )
        return result, await repository.load_pages('source-1')

    result, reloaded = asyncio.run(run_and_reload())

    assert [result_item.keep for result_item in result.correction_results] == [
        False,
        True,
    ]
    assert [block.content for block in result.corrected_pages[0].blocks] == [
        'x^2 Advertisement footer https://example.com/keep',
        None,
        None,
    ]
    assert formatted[0].position == 0
    assert result.corrected_pages[0].blocks[0].uuid == (
        result.formatted_pages[0].blocks[0].uuid
    )
    assert [block.uuid for block in reloaded[0].blocks] == [
        block.uuid for block in result.embedded_pages[0].blocks
    ]
    assert [block.content for block in reloaded[0].blocks] == [
        'x^2 https://example.com/keep',
        'retained figure',
    ]
    assert [asset.path for asset in reloaded[0].blocks[1].assets] == [
        result.embedded_pages[0].blocks[1].assets[0].path
    ]
    assert Path(reloaded[0].blocks[1].assets[0].path).read_bytes() == (
        retained_image.read_bytes()
    )
    assert (
        Path(reloaded[0].blocks[0].crop_path).read_bytes() == crop.read_bytes()
    )
    assert reloaded[0].markdown == 'captured page markdown'
    assert rejected_image.exists()


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]


def _content_correction_node(corrector: object) -> ContentCorrectionNode:
    return ContentCorrectionNode(
        corrector,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=10_000
        ),
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=32,
    )


def _formatting_node(formatter: object) -> FormattingNode:

    return FormattingNode(
        formatter,
        budget=TokenBudget(
            counter=MaterialTokenEstimator(_Counter()), token_limit=10_000
        ),
        batch_size=32,
    )


class _UnexpectedDescriber:
    async def acall(self, **kwargs: object) -> list[ImageDescriptionOutput]:
        raise AssertionError('describer must not run without image blocks')


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.materials: list[MultimodalContent] | None = None
        self.content_images: list[ContentImage] | None = None

    async def embed(
        self,
        materials: list[MultimodalContent],
        *,
        content_images: list[ContentImage],
        input_type: Literal['query', 'document'],
    ) -> list[list[float]]:
        self.materials = materials
        self.content_images = content_images
        return [[float(index)] for index in range(len(materials))]


class _RecordingPersistence:
    def __init__(self) -> None:
        self.pages: list[SourcePage] | None = None

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        self.pages = state.embedded_pages
        return {}


def _embedding_node() -> EmbeddingNode:
    return EmbeddingNode(
        _RecordingEmbedder(),
        model='test-embedding',
        dimension=1,
    )


def _no_describer() -> ImageDescriptionNode:
    counter = _Counter()
    return ImageDescriptionNode(
        _UnexpectedDescriber(),
        ContextWindowSettings(backward_budget=100, forward_budget=100),
        budget=TokenBudget(MaterialTokenEstimator(counter), 10_000),
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=4,
        token_counters=(MaterialTokenEstimator(counter),),
    )


class _UnexpectedImageFilter:
    async def acall(self, **kwargs: object) -> list[ImageFilterDecision]:
        raise AssertionError('image filter must not run without image targets')


class _KeepAllImageFilter:
    async def acall(self, *, items, content_images):
        return [
            ImageFilterDecision(position=item.position, keep=True)
            for item in items
        ]


def _image_filter_node(module: object | None = None) -> ImageFilterNode:
    counter = MaterialTokenEstimator(_Counter())
    return ImageFilterNode(
        module or _UnexpectedImageFilter(),
        ContextWindowSettings(backward_budget=100, forward_budget=100),
        budget=TokenBudget(counter, 10_000),
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        batch_size=4,
        token_counters=(counter,),
    )


def _text_seam_node(judge: object, rewriter: object) -> TextSeamNode:
    return TextSeamNode(
        judge,
        rewriter,
        judge_budget=TokenBudget(MaterialTokenEstimator(_Counter()), 10_000),
        rewrite_budget=TokenBudget(MaterialTokenEstimator(_Counter()), 10_000),
        judge_batch_size=16,
        rewrite_batch_size=16,
    )


def _image_seam_node(judge: object) -> ImageSeamNode:
    return ImageSeamNode(
        judge,
        image_budget=ImagePayloadBudget(
            max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
        ),
        budget=TokenBudget(MaterialTokenEstimator(_Counter()), 10_000),
        batch_size=4,
    )


def test_ocr_node_delegates_once_to_injected_provider(tmp_path):
    calls = []
    page_image = tmp_path / 'page.png'
    Image.new('RGB', (4, 4), 'white').save(page_image)
    artifacts = [
        OCRPageArtifact(
            page_index=2,
            page_image_path=str(page_image),
            markdown='# Page 2',
            blocks=[
                OCRArtifact(
                    page_index=2,
                    block_index=1,
                    block_type='paragraph',
                    content='second',
                    crop_bbox=(1, 1, 2, 2),
                ),
                OCRArtifact(
                    page_index=2,
                    block_index=0,
                    block_type='paragraph',
                    content='first',
                    crop_bbox=(0, 0, 1, 1),
                ),
            ],
        )
    ]

    class FakeProvider:
        def extract(
            self,
            pdf_path: str,
            *,
            source_uuid: str,
            pages: list[int] | None = None,
        ) -> list[OCRPageArtifact]:
            calls.append((pdf_path, source_uuid, pages))
            return artifacts

    provider: OCRProvider = FakeProvider()
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf', 'pages': [2]},
        source=Source(uuid='source-1', key='book.pdf'),
    )
    result = OCRNode(provider).run(state)

    assert calls == [('book.pdf', 'source-1', [2])]
    assert set(result) == {'ocr_pages', 'ocr_page_images'}
    assert result['ocr_pages'][0].index == 2
    assert result['ocr_page_images'] == {2: str(page_image)}
    assert [block.content for block in result['ocr_pages'][0].blocks] == [
        'first',
        'second',
    ]
    assert 'ocr_artifacts' not in result


def test_content_correction_dispatch_routes_empty_ocr_to_collect():
    node = _content_correction_node(object())
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        source=Source(uuid='source-1', key='book.pdf'),
    )

    assert asyncio.run(node.dispatch(state)) == 'content_correction_collect'


def test_content_correction_collects_results_by_page_and_block_position():
    first = SourceBlock(uuid='first', block_type='paragraph', content='first')
    second = SourceBlock(
        uuid='second', block_type='paragraph', content='second'
    )
    state = SourceProcessingState(
        input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
        ocr_pages=[SourcePage(index=4, blocks=[first, second])],
        source=Source(uuid='source-1', key='book.pdf'),
    )
    state.correction_results = [
        ContentCorrectionResult(
            page_index=4,
            block_position=1,
            source_block=second.model_copy(
                update={'content': 'corrected second'}
            ),
            keep=True,
            cleanup_instructions=[],
        ),
        ContentCorrectionResult(
            page_index=4,
            block_position=0,
            source_block=first.model_copy(
                update={'content': 'corrected first'}
            ),
            keep=True,
            cleanup_instructions=[],
        ),
    ]

    result = _content_correction_node(object()).collect(state)

    assert [block.content for block in result['corrected_pages'][0].blocks] == [
        'corrected first',
        'corrected second',
    ]


def test_source_graph_runs_correction_formatting_and_text_seams(tmp_path):
    crop = tmp_path / 'crop.png'
    page_image = tmp_path / 'page.png'
    Image.new('RGB', (2, 2), 'red').save(crop)
    Image.new('RGB', (8, 8), 'white').save(page_image)
    artifacts = [
        OCRPageArtifact(
            page_index=page_index,
            page_image_path=str(page_image),
            markdown=f'# Page {page_index}',
            blocks=[
                OCRArtifact(
                    page_index=page_index,
                    block_index=0,
                    block_type='paragraph',
                    content=content,
                    crop_path=str(crop),
                    crop_bbox=(0, 0, 2, 2),
                )
            ],
        )
        for page_index, content in enumerate(['first', 'second'])
    ]

    class FakeProvider:
        def extract(
            self,
            pdf_path: str,
            *,
            source_uuid: str,
            pages: list[int] | None = None,
        ) -> list[OCRPageArtifact]:
            return artifacts

    events: list[str] = []

    class FakeCorrector:
        async def acall(
            self,
            *,
            blocks: list[object],
            content_images: list[object],
        ) -> list[ContentCorrectionOutput]:
            return [
                ContentCorrectionOutput(
                    keep=True,
                    cleanup_instructions=[],
                    position=block.position,
                    corrected_content=f'corrected {block.material.text}',
                )
                for block in blocks
            ]

    class FakeFormatter:
        async def acall(
            self, *, blocks: list[FormattingInput]
        ) -> list[FormattingOutput]:
            events.extend(f'format {block.content}' for block in blocks)
            return [
                FormattingOutput(
                    position=block.position,
                    formatted_content=f'formatted {block.content}',
                )
                for block in blocks
            ]

    class FakeJudge:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamDecision]:
            assert events == [
                'format corrected first',
                'format corrected second',
            ]
            return [
                TextSeamDecision(position=item.position, is_split=True)
                for item in items
            ]

    class FakeRewriter:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamRewriteOutput]:
            events.append('rewrite')
            return [
                TextSeamRewriteOutput(
                    position=item.position, merged='merged tail and head'
                )
                for item in items
            ]

    class FakeImageJudge:
        async def acall(
            self, *, items: list[ImageSeamInput], content_images: list[object]
        ) -> list[ImageSeamDecision]:
            raise AssertionError('no eligible image seams')

    persistence = _RecordingPersistence()
    source_graph = SourceProcessingGraph(
        OCRNode(FakeProvider()),
        _content_correction_node(FakeCorrector()),
        _formatting_node(FakeFormatter()),
        _text_seam_node(FakeJudge(), FakeRewriter()),
        _image_seam_node(FakeImageJudge()),
        _image_filter_node(),
        _no_describer(),
        _embedding_node(),
        persistence,
    )

    async def run_nested_with_lower_parent_limit():
        parent = StateGraph(dict)

        async def process(state):
            result = await source_graph.run(state['source_state'])
            return {'result': result}

        parent.add_node('process', process)
        parent.add_edge(START, 'process')
        parent.add_edge('process', END)
        compiled_parent = parent.compile()
        return await compiled_parent.ainvoke(
            {
                'source_state': SourceProcessingState(
                    input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
                    source=Source(uuid='source-1', key='book.pdf'),
                )
            },
            config={'recursion_limit': 16},
        )

    result = asyncio.run(run_nested_with_lower_parent_limit())['result']

    assert [
        content.content
        for page in result.corrected_pages
        for content in page.blocks
    ] == ['corrected first', 'corrected second']
    assert [
        content.content
        for page in result.formatted_pages
        for content in page.blocks
    ] == ['formatted corrected first', 'formatted corrected second']
    assert [
        content.content
        for page in result.text_seam_pages
        for content in page.blocks
    ] == ['merged tail and head']
    assert [page.index for page in result.text_seam_pages] == [0, 1]
    assert [page.markdown for page in result.image_described_pages] == [
        '# Page 0',
        '# Page 1',
    ]
    assert persistence.pages == result.embedded_pages


def test_source_graph_merges_adjacent_image_artifacts_after_text_seams(
    tmp_path,
):
    crop = tmp_path / 'crop.png'
    top_image = tmp_path / 'top.png'
    bottom_image = tmp_path / 'bottom.png'
    Image.new('RGB', (2, 2), 'red').save(crop)
    Image.new('RGB', (2, 2), 'blue').save(top_image)
    Image.new('RGB', (2, 2), 'green').save(bottom_image)
    artifacts = [
        OCRPageArtifact(
            page_index=0,
            page_image_path=str(crop),
            markdown='',
            blocks=[
                OCRArtifact(
                    page_index=0,
                    block_index=0,
                    block_type='image',
                    images=[OCRImageArtifact(path=str(top_image))],
                    crop_path=str(crop),
                    crop_bbox=(0, 0, 2, 2),
                )
            ],
        ),
        OCRPageArtifact(
            page_index=1,
            page_image_path=str(crop),
            markdown='',
            blocks=[
                OCRArtifact(
                    page_index=1,
                    block_index=0,
                    block_type='image',
                    images=[OCRImageArtifact(path=str(bottom_image))],
                    crop_path=str(crop),
                    crop_bbox=(0, 0, 2, 2),
                )
            ],
        ),
    ]

    class FakeProvider:
        def extract(
            self,
            pdf_path: str,
            *,
            source_uuid: str,
            pages: list[int] | None = None,
        ) -> list[OCRPageArtifact]:
            return artifacts

    class FakeCorrector:
        async def acall(
            self,
            *,
            blocks: list[object],
            content_images: list[object],
        ) -> list[ContentCorrectionOutput]:
            return [
                ContentCorrectionOutput(
                    keep=True,
                    cleanup_instructions=[],
                    position=block.position,
                    corrected_content=block.material.text,
                )
                for block in blocks
            ]

    class FakeFormatter:
        async def acall(
            self, *, blocks: list[FormattingInput]
        ) -> list[FormattingOutput]:
            raise AssertionError('formatter must not run for image-only pages')

    class FakeTextJudge:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamDecision]:
            raise AssertionError('text judge must not run for image-only pages')

    class FakeTextRewriter:
        async def acall(
            self, *, items: list[TextSeamInput]
        ) -> list[TextSeamRewriteOutput]:
            raise AssertionError(
                'text rewriter must not run for image-only pages'
            )

    class FakeImageJudge:
        async def acall(
            self, *, items: list[ImageSeamInput], content_images: list[object]
        ) -> list[ImageSeamDecision]:
            return [
                ImageSeamDecision(position=item.position, is_continuation=True)
                for item in items
            ]

    class FakeDescriber:
        async def acall(
            self,
            *,
            items: list[ImageDescriptionInput],
            content_images: list[object],
        ) -> list[ImageDescriptionOutput]:
            assert len(items) == 1
            assert items[0].target.image_positions == [0, 1]
            assert len(content_images) == 2
            return [
                ImageDescriptionOutput(
                    position=items[0].position,
                    description='one merged visual description',
                )
            ]

    persistence = _RecordingPersistence()
    source_graph = SourceProcessingGraph(
        OCRNode(FakeProvider()),
        _content_correction_node(FakeCorrector()),
        _formatting_node(FakeFormatter()),
        _text_seam_node(FakeTextJudge(), FakeTextRewriter()),
        _image_seam_node(FakeImageJudge()),
        _image_filter_node(_KeepAllImageFilter()),
        ImageDescriptionNode(
            FakeDescriber(),
            ContextWindowSettings(backward_budget=100, forward_budget=100),
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 10_000),
            image_budget=ImagePayloadBudget(
                max_images=4, max_bytes=25_000_000, max_pixels=40_000_000
            ),
            batch_size=4,
            token_counters=(MaterialTokenEstimator(_Counter()),),
        ),
        _embedding_node(),
        persistence,
    )

    result = asyncio.run(
        source_graph.run(
            SourceProcessingState(
                input={'kind': 'pdf', 'pdf_path': 'book.pdf'},
                source=Source(uuid='source-1', key='book.pdf'),
            )
        )
    )

    assert [
        [asset.path for asset in page.blocks[0].assets]
        for page in result.text_seam_pages
    ] == [[str(top_image)], [str(bottom_image)]]
    assert [
        [asset.path for asset in page.blocks[0].assets] if page.blocks else []
        for page in result.image_seam_pages
    ] == [[str(top_image), str(bottom_image)], []]
    assert [
        page.blocks[0].content if page.blocks else None
        for page in result.image_described_pages
    ] == ['one merged visual description', None]
    assert persistence.pages == result.embedded_pages
