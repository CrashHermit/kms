import asyncio

from kms2.config import ContextWindowSettings
from kms2.core.model import Source, SourceBlock, SourcePage
from kms2.core.model.content_correction import ContentCorrectionResult
from kms2.core.model.ocr import OCRArtifact, OCRImageArtifact
from kms2.langgraph.source import OCRNode, SourceState
from kms2.langgraph.source.graph import SourceGraph
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_enrichment import ImageEnrichmentNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.splitter import SplitterNode
from kms2.node.source.text_seam import TextSeamNode
from kms2.ocr.provider import OCRProvider


class _FalseRouter:
    async def aforward(self, **kwargs: object) -> bool:
        return False


class _UnexpectedSplitter:
    async def aforward(self, **kwargs: object) -> object:
        raise AssertionError('splitter must not run without routed candidates')


class _UnexpectedEnricher:
    async def aforward(self, **kwargs: object) -> str:
        raise AssertionError('enricher must not run without image blocks')


class _RecordingPersistence:
    def __init__(self) -> None:
        self.pages: list[SourcePage] | None = None

    async def run(self, state: SourceState) -> dict[str, object]:
        self.pages = state.split_pages
        return {}


def _no_splitter() -> SplitterNode:
    return SplitterNode(
        _FalseRouter(),
        _UnexpectedSplitter(),
        ContextWindowSettings(
            backward_budget=100,
            forward_budget=100,
        ),
    )


def _no_enricher() -> ImageEnrichmentNode:
    return ImageEnrichmentNode(
        _UnexpectedEnricher(),
        ContextWindowSettings(backward_budget=100, forward_budget=100),
    )


def test_source_state_carries_source_pipeline_data():
    source = Source(uuid='source-1', key='document.pdf')
    artifact = SourceBlock(
        uuid='content-1',
        block_type='text',
        content='original',
        crop_path='output/blocks/block-0.png',
    )
    ocr_page = SourcePage(index=0, blocks=[artifact])
    corrected_content = SourceBlock(
        uuid='content-1',
        block_type='text',
        content='corrected source content',
    )
    corrected_page = SourcePage(index=0, blocks=[corrected_content])
    state = SourceState(
        pdf_path='document.pdf',
        pages=[1, 2],
        source=source,
        ocr_pages=[ocr_page],
        corrected_pages=[corrected_page],
    )

    assert state.source is source
    assert state.pages == [1, 2]
    assert state.ocr_pages == [ocr_page]
    assert state.correction_results == []
    assert state.corrected_pages == [corrected_page]
    assert state.formatting_results == []
    assert state.formatted_pages == []
    assert state.text_seam_even_results == []
    assert state.text_seam_even_pages == []
    assert state.text_seam_odd_results == []
    assert state.text_seam_pages == []
    assert state.image_seam_even_results == []
    assert state.image_seam_even_pages == []
    assert state.image_seam_odd_results == []
    assert state.image_seam_pages == []
    assert state.image_enrichment_results == []
    assert state.image_enriched_pages == []
    assert state.split_pages == []
    assert state.split_results == []


def test_source_state_defaults_optional_pipeline_fields():
    source = Source(uuid='source-1', key='document.pdf')

    state = SourceState(pdf_path='document.pdf', source=source)

    assert state.pages is None
    assert state.ocr_pages == []
    assert state.correction_results == []
    assert state.corrected_pages == []
    assert state.formatting_results == []
    assert state.formatted_pages == []
    assert state.text_seam_even_results == []
    assert state.text_seam_even_pages == []
    assert state.text_seam_odd_results == []
    assert state.text_seam_pages == []
    assert state.image_seam_even_results == []
    assert state.image_seam_even_pages == []
    assert state.image_seam_odd_results == []
    assert state.image_seam_pages == []
    assert state.image_enrichment_results == []
    assert state.image_enriched_pages == []
    assert state.split_pages == []
    assert state.split_results == []


def test_ocr_node_delegates_once_to_injected_provider():
    calls = []
    artifacts = [
        OCRArtifact(
            page_index=2,
            block_index=1,
            block_type='text',
            content='second',
        ),
        OCRArtifact(
            page_index=2,
            block_index=0,
            block_type='text',
            content='first',
        ),
    ]

    class FakeProvider:
        def extract(
            self, pdf_path: str, *, pages: list[int] | None = None
        ) -> list[OCRArtifact]:
            calls.append((pdf_path, pages))
            return artifacts

    provider: OCRProvider = FakeProvider()
    state = SourceState(
        pdf_path='book.pdf',
        pages=[2],
        source=Source(uuid='source-1', key='book.pdf'),
    )
    result = OCRNode(provider).run(state)

    assert calls == [('book.pdf', [2])]
    assert list(result) == ['ocr_pages']
    assert result['ocr_pages'][0].index == 2
    assert [block.content for block in result['ocr_pages'][0].blocks] == [
        'first',
        'second',
    ]
    assert 'ocr_artifacts' not in result


def test_content_correction_dispatch_routes_empty_ocr_to_collect():
    node = ContentCorrectionNode(object())
    state = SourceState(
        pdf_path='book.pdf',
        source=Source(uuid='source-1', key='book.pdf'),
    )

    assert node.dispatch(state) == 'content_correction_collect'


def test_content_correction_collects_results_by_page_and_block_position():
    first = SourceBlock(uuid='first', block_type='text', content='first')
    second = SourceBlock(uuid='second', block_type='text', content='second')
    state = SourceState(
        pdf_path='book.pdf',
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

    result = ContentCorrectionNode(object()).collect(state)

    assert [block.content for block in result['corrected_pages'][0].blocks] == [
        'corrected first',
        'corrected second',
    ]


def test_source_graph_runs_correction_formatting_and_text_seams():
    artifacts = [
        OCRArtifact(
            page_index=page_index,
            block_index=0,
            block_type='text',
            content=content,
            crop_path='data:image/png;base64,AA==',
        )
        for page_index, content in enumerate(['first', 'second'])
    ]

    class FakeProvider:
        def extract(
            self, pdf_path: str, *, pages: list[int] | None = None
        ) -> list[OCRArtifact]:
            return artifacts

    events: list[str] = []

    class FakeCorrector:
        async def acall(
            self,
            *,
            block_crop: object,
            block_type: str,
            content: str,
        ) -> str:
            return f'corrected {content}'

    class FakeFormatter:
        async def acall(self, *, content: str) -> str:
            events.append(f'format {content}')
            return f'formatted {content}'

    class FakeJudge:
        async def aforward(self, **kwargs: object) -> bool:
            assert events == [
                'format corrected first',
                'format corrected second',
            ]
            return True

    class FakeRewriter:
        async def aforward(self, **kwargs: object) -> str:
            events.append('rewrite')
            return 'merged tail and head'

    class FakeImageJudge:
        async def aforward(self, **kwargs: object) -> bool:
            raise AssertionError('image judge must not run for text-only pages')

    persistence = _RecordingPersistence()
    source_graph = SourceGraph(
        OCRNode(FakeProvider()),
        ContentCorrectionNode(FakeCorrector()),
        FormattingNode(FakeFormatter()),
        TextSeamNode(FakeJudge(), FakeRewriter()),
        ImageSeamNode(FakeImageJudge()),
        _no_enricher(),
        _no_splitter(),
        persistence,
    ).build_graph()

    result = asyncio.run(
        source_graph.ainvoke(
            {
                'pdf_path': 'book.pdf',
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
    assert result['split_pages'] == result['image_enriched_pages']
    assert persistence.pages == result['split_pages']


def test_source_graph_merges_adjacent_image_artifacts_after_text_seams():
    artifacts = [
        OCRArtifact(
            page_index=0,
            block_index=0,
            block_type='image',
            images=[OCRImageArtifact(path='top.png')],
            crop_path='data:image/png;base64,AA==',
        ),
        OCRArtifact(
            page_index=1,
            block_index=0,
            block_type='image',
            images=[OCRImageArtifact(path='bottom.png')],
            crop_path='data:image/png;base64,AA==',
        ),
    ]

    class FakeProvider:
        def extract(
            self, pdf_path: str, *, pages: list[int] | None = None
        ) -> list[OCRArtifact]:
            return artifacts

    class FakeCorrector:
        async def acall(
            self,
            *,
            block_crop: object,
            block_type: str,
            content: str | None,
        ) -> str | None:
            return content

    class FakeFormatter:
        async def acall(self, *, content: str) -> str:
            raise AssertionError('formatter must not run for image-only pages')

    class FakeTextJudge:
        async def aforward(self, **kwargs: object) -> bool:
            raise AssertionError('text judge must not run for image-only pages')

    class FakeTextRewriter:
        async def aforward(self, **kwargs: object) -> str:
            raise AssertionError(
                'text rewriter must not run for image-only pages'
            )

    class FakeImageJudge:
        async def aforward(self, **kwargs: object) -> bool:
            return True

    class FakeEnricher:
        async def aforward(
            self,
            *,
            source_block: SourceBlock,
            context_before: object,
            context_after: object,
        ) -> str:
            assert [asset.path for asset in source_block.assets] == [
                'top.png',
                'bottom.png',
            ]
            return 'one merged visual description'

    persistence = _RecordingPersistence()
    source_graph = SourceGraph(
        OCRNode(FakeProvider()),
        ContentCorrectionNode(FakeCorrector()),
        FormattingNode(FakeFormatter()),
        TextSeamNode(FakeTextJudge(), FakeTextRewriter()),
        ImageSeamNode(FakeImageJudge()),
        ImageEnrichmentNode(
            FakeEnricher(),
            ContextWindowSettings(backward_budget=100, forward_budget=100),
        ),
        _no_splitter(),
        persistence,
    ).build_graph()

    result = asyncio.run(
        source_graph.ainvoke(
            {
                'pdf_path': 'book.pdf',
                'source': Source(uuid='source-1', key='book.pdf'),
            }
        )
    )

    assert [
        [asset.path for asset in page.blocks[0].assets]
        for page in result['text_seam_pages']
    ] == [['top.png'], ['bottom.png']]
    assert [
        [asset.path for asset in page.blocks[0].assets] if page.blocks else []
        for page in result['image_seam_pages']
    ] == [['top.png', 'bottom.png'], []]
    assert [
        page.blocks[0].content if page.blocks else None
        for page in result['image_enriched_pages']
    ] == ['one merged visual description', None]
    assert result['split_pages'] == result['image_enriched_pages']
    assert persistence.pages == result['split_pages']
