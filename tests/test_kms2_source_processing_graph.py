import asyncio

from PIL import Image

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.content_correction import (
    ContentCorrectionOutput,
    ContentCorrectionResult,
)
from kms2.core.model.source_processing.formatting import (
    FormattingInput,
    FormattingOutput,
)
from kms2.core.model.source_processing.image_description import (
    ImageDescriptionInput,
    ImageDescriptionOutput,
)
from kms2.core.model.source_processing.image_seam import (
    ImageSeamDecision,
    ImageSeamInput,
)
from kms2.core.model.source_processing.ocr import (
    OCRArtifact,
    OCRImageArtifact,
    OCRPageArtifact,
)
from kms2.core.model.source_processing.text_seam import (
    TextSeamDecision,
    TextSeamInput,
    TextSeamRewriteOutput,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_processing.graph import SourceProcessingGraph
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.node.source_processing.content_correction import ContentCorrectionNode
from kms2.node.source_processing.embedding import EmbeddingNode
from kms2.node.source_processing.formatting import FormattingNode
from kms2.node.source_processing.image_description import ImageDescriptionNode
from kms2.node.source_processing.image_seam import ImageSeamNode
from kms2.node.source_processing.input import SourceInputNode
from kms2.node.source_processing.ocr import OCRNode
from kms2.node.source_processing.text_seam import TextSeamNode
from kms2.ocr.provider import OCRProvider


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [1 for _ in texts]


def _content_correction_node(corrector: object) -> ContentCorrectionNode:
    return ContentCorrectionNode(
        corrector,
        budget=TokenBudget(counter=_Counter(), token_limit=10_000),
        batch_size=32,
    )


def _formatting_node(formatter: object) -> FormattingNode:

    return FormattingNode(
        formatter,
        budget=TokenBudget(counter=_Counter(), token_limit=10_000),
        batch_size=32,
    )


class _UnexpectedDescriber:
    async def acall(self, **kwargs: object) -> list[ImageDescriptionOutput]:
        raise AssertionError('describer must not run without image blocks')


class _RecordingEmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[float(index)] for index in range(len(texts))]


class _RecordingPersistence:
    def __init__(self) -> None:
        self.pages: list[SourcePage] | None = None

    async def run(self, state: SourceProcessingState) -> dict[str, object]:
        self.pages = state.embedded_pages
        return {}


def _embedding_node() -> EmbeddingNode:
    return EmbeddingNode(_RecordingEmbeddingClient())


def _no_describer() -> ImageDescriptionNode:
    counter = _Counter()
    return ImageDescriptionNode(
        _UnexpectedDescriber(),
        ContextWindowSettings(backward_budget=100, forward_budget=100),
        budget=TokenBudget(counter, 10_000),
        batch_size=4,
        token_counters=(counter,),
    )


def _text_seam_node(judge: object, rewriter: object) -> TextSeamNode:
    return TextSeamNode(
        judge,
        rewriter,
        judge_budget=TokenBudget(_Counter(), 10_000),
        rewrite_budget=TokenBudget(_Counter(), 10_000),
        judge_batch_size=16,
        rewrite_batch_size=16,
    )


def _image_seam_node(judge: object) -> ImageSeamNode:
    return ImageSeamNode(
        judge,
        budget=TokenBudget(_Counter(), 10_000),
        batch_size=4,
    )


def test_ocr_node_delegates_once_to_injected_provider():
    calls = []
    artifacts = [
        OCRPageArtifact(
            page_index=2,
            markdown='# Page 2',
            blocks=[
                OCRArtifact(
                    page_index=2,
                    block_index=1,
                    block_type='paragraph',
                    content='second',
                ),
                OCRArtifact(
                    page_index=2,
                    block_index=0,
                    block_type='paragraph',
                    content='first',
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
    assert list(result) == ['ocr_pages']
    assert result['ocr_pages'][0].index == 2
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

    assert node.dispatch(state) == 'content_correction_collect'


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
        ),
        ContentCorrectionResult(
            page_index=4,
            block_position=0,
            source_block=first.model_copy(
                update={'content': 'corrected first'}
            ),
        ),
    ]

    result = _content_correction_node(object()).collect(state)

    assert [block.content for block in result['corrected_pages'][0].blocks] == [
        'corrected first',
        'corrected second',
    ]


def test_source_graph_runs_correction_formatting_and_text_seams(tmp_path):
    crop = tmp_path / 'crop.png'
    Image.new('RGB', (2, 2), 'red').save(crop)
    artifacts = [
        OCRPageArtifact(
            page_index=page_index,
            markdown=f'# Page {page_index}',
            blocks=[
                OCRArtifact(
                    page_index=page_index,
                    block_index=0,
                    block_type='paragraph',
                    content=content,
                    crop_path=str(crop),
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
            block_crops: list[object],
        ) -> list[ContentCorrectionOutput]:
            return [
                ContentCorrectionOutput(
                    position=block.position,
                    corrected_content=f'corrected {block.content}',
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
            self, *, items: list[ImageSeamInput], seam_images: list[object]
        ) -> list[ImageSeamDecision]:
            raise AssertionError('no eligible image seams')

    persistence = _RecordingPersistence()
    source_graph = SourceProcessingGraph(
        SourceInputNode(),
        OCRNode(FakeProvider()),
        _content_correction_node(FakeCorrector()),
        _formatting_node(FakeFormatter()),
        _text_seam_node(FakeJudge(), FakeRewriter()),
        _image_seam_node(FakeImageJudge()),
        _no_describer(),
        _embedding_node(),
        persistence,
    ).build_graph()

    result = asyncio.run(
        source_graph.ainvoke(
            {
                'input': {'kind': 'pdf', 'pdf_path': 'book.pdf'},
                'source': Source(uuid='source-1', key='book.pdf'),
            }
        )
    )

    assert [
        content.content
        for page in result['corrected_pages']
        for content in page.blocks
    ] == ['corrected first', 'corrected second']
    assert [
        content.content
        for page in result['formatted_pages']
        for content in page.blocks
    ] == ['formatted corrected first', 'formatted corrected second']
    assert [
        content.content
        for page in result['text_seam_pages']
        for content in page.blocks
    ] == ['merged tail and head']
    assert [page.index for page in result['text_seam_pages']] == [0, 1]
    assert [page.markdown for page in result['image_described_pages']] == [
        '# Page 0',
        '# Page 1',
    ]
    assert persistence.pages == result['embedded_pages']


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
            markdown='',
            blocks=[
                OCRArtifact(
                    page_index=0,
                    block_index=0,
                    block_type='image',
                    images=[OCRImageArtifact(path=str(top_image))],
                    crop_path=str(crop),
                )
            ],
        ),
        OCRPageArtifact(
            page_index=1,
            markdown='',
            blocks=[
                OCRArtifact(
                    page_index=1,
                    block_index=0,
                    block_type='image',
                    images=[OCRImageArtifact(path=str(bottom_image))],
                    crop_path=str(crop),
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
            block_crops: list[object],
        ) -> list[ContentCorrectionOutput]:
            return [
                ContentCorrectionOutput(
                    position=block.position,
                    corrected_content=block.content,
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
            self, *, items: list[ImageSeamInput], seam_images: list[object]
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
            target_images: list[object],
        ) -> list[ImageDescriptionOutput]:
            assert len(items) == 1
            assert items[0].image_positions == [0, 1]
            assert len(target_images) == 2
            return [
                ImageDescriptionOutput(
                    position=items[0].position,
                    description='one merged visual description',
                )
            ]

    persistence = _RecordingPersistence()
    source_graph = SourceProcessingGraph(
        SourceInputNode(),
        OCRNode(FakeProvider()),
        _content_correction_node(FakeCorrector()),
        _formatting_node(FakeFormatter()),
        _text_seam_node(FakeTextJudge(), FakeTextRewriter()),
        _image_seam_node(FakeImageJudge()),
        ImageDescriptionNode(
            FakeDescriber(),
            ContextWindowSettings(backward_budget=100, forward_budget=100),
            budget=TokenBudget(_Counter(), 10_000),
            batch_size=4,
            token_counters=(_Counter(),),
        ),
        _embedding_node(),
        persistence,
    ).build_graph()

    result = asyncio.run(
        source_graph.ainvoke(
            {
                'input': {'kind': 'pdf', 'pdf_path': 'book.pdf'},
                'source': Source(uuid='source-1', key='book.pdf'),
            }
        )
    )

    assert [
        [asset.path for asset in page.blocks[0].assets]
        for page in result['text_seam_pages']
    ] == [[str(top_image)], [str(bottom_image)]]
    assert [
        [asset.path for asset in page.blocks[0].assets] if page.blocks else []
        for page in result['image_seam_pages']
    ] == [[str(top_image), str(bottom_image)], []]
    assert [
        page.blocks[0].content if page.blocks else None
        for page in result['image_described_pages']
    ] == ['one merged visual description', None]
    assert persistence.pages == result['embedded_pages']
