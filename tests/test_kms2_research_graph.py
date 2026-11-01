import asyncio
import base64
from datetime import UTC, datetime
from io import BytesIO

import pytest
from PIL import Image

from kms2.core.media_budget import ImagePayloadBudget
from kms2.core.model.research.inference import (
    FetchWebAction,
    FinishResearchAction,
    ListEvidenceAction,
    OpenPdfAction,
    ReadEvidenceAction,
    ReadPdfPagesAction,
    ResearchTask,
    SearchPdfAction,
    WebSearchAction,
)
from kms2.core.model.research.retrieval import (
    PdfSearchMatch,
    PdfSearchResult,
    ResearchDocument,
    ResearchEvidenceEntry,
    WebExtraction,
    WebExtractionBatch,
    WebExtractionFailure,
)
from kms2.core.model.source_processing.ocr import OCRPageArtifact
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.research.graph import ResearchGraph
from kms2.langgraph.research.state import ResearchState
from kms2.node.research.researcher import ResearcherNode
from kms2.node.research.tools import ResearchToolsNode


def _png(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new('RGB', (1, 1), color).save(output, format='PNG')
    return output.getvalue()


def _pixel(image) -> tuple[int, int, int]:
    encoded = image.url.split(',', 1)[1]
    decoded = Image.open(BytesIO(base64.b64decode(encoded)))
    return decoded.convert('RGB').getpixel((0, 0))


class _Counter:
    def __init__(self, *, limit: int | None = None) -> None:
        self.limit = limit

    def count_texts(self, texts: list[str]) -> list[int]:
        return [
            self.limit if self.limit is not None else len(text)
            for text in texts
        ]


class _Predictor:
    def __init__(self, actions) -> None:
        self.actions = list(actions)
        self.contexts = []
        self.images = []

    async def aforward(self, *, context, evidence_images):
        self.contexts.append(context)
        self.images.append(evidence_images)
        return self.actions.pop(0)


class _FailingPredictor(_Predictor):
    async def aforward(self, *, context, evidence_images):
        raise RuntimeError('research inference failed')


class _WebClient:
    async def search(self, query: str):
        return []


class _FailingWebClient:
    async def search(self, query: str):
        raise PermissionError('Tavily authentication failed')


class _EvidenceService:
    def __init__(self) -> None:
        self.pdf_reads: list[list[int]] = []

    async def fetch_web(self, urls: list[str]) -> WebExtractionBatch:
        now = datetime(2026, 10, 31, tzinfo=UTC)
        return WebExtractionBatch(
            results=[
                WebExtraction(
                    url=urls[0],
                    retrieved_at=now,
                    raw_markdown='The source says Av = λv.',
                    image_urls=[
                        'https://example.test/diagram.png',
                        'https://example.test/unread.png',
                    ],
                )
            ]
        )

    async def discover_web(
        self,
        document: ResearchDocument,
        document_position: int,
    ) -> list[ResearchEvidenceEntry]:
        return [
            ResearchEvidenceEntry(
                document_position=document_position,
                kind='web_passage',
                url=document.url,
                retrieved_at=document.retrieved_at,
                line_start=0,
                line_end=1,
                token_count=7,
                preview='The source says Av = λv.',
                raw_content='The source says Av = λv.',
            ),
            ResearchEvidenceEntry(
                document_position=document_position,
                kind='web_passage',
                url=document.url,
                retrieved_at=document.retrieved_at,
                line_start=2,
                line_end=4,
                token_count=3,
                preview='An unrelated sidebar.',
                raw_content='An unrelated sidebar.',
            ),
            ResearchEvidenceEntry(
                document_position=document_position,
                kind='image',
                url='https://example.test/diagram.png',
                retrieved_at=document.retrieved_at,
                preview='https://example.test/diagram.png',
            ),
            ResearchEvidenceEntry(
                document_position=document_position,
                kind='image',
                url='https://example.test/unread.png',
                retrieved_at=document.retrieved_at,
                preview='https://example.test/unread.png',
            ),
        ]

    async def read_evidence(self, entries: list[ResearchEvidenceEntry]):
        return [
            entry.model_copy(
                update={
                    'content': entry.raw_content or None,
                    'media_type': 'image/png'
                    if entry.kind == 'image'
                    else None,
                    'media_bytes': _png((220, 30, 30))
                    if entry.kind == 'image'
                    else None,
                }
            )
            for entry in entries
        ]

    async def open_pdf(self, url: str) -> ResearchDocument:
        return ResearchDocument(
            url=url,
            title='Course notes',
            retrieved_at=datetime(2026, 10, 31, tzinfo=UTC),
            media_type='application/pdf',
            page_count=2,
        )

    async def search_pdf(self, document: ResearchDocument, query: str):
        return PdfSearchResult(
            matches=[PdfSearchMatch(page_index=1, snippet='projection')]
        )

    async def prepare_pdf_pages(
        self,
        document: ResearchDocument,
        page_indexes: list[int],
    ):
        self.pdf_reads.append(page_indexes)
        return [
            ResearchEvidenceEntry(
                document_position=0,
                kind='pdf_page',
                url=document.url,
                title=document.title,
                retrieved_at=document.retrieved_at,
                page_index=page_index,
                line_start=0,
                line_end=1,
                raw_content='The selected page describes a projection.',
                content='The selected page describes a projection.',
                media_type='image/png',
                ocr_artifacts=OCRPageArtifact(
                    page_index=page_index,
                    markdown='The selected page describes a projection.',
                    blocks=[],
                ),
                rendered_page_bytes=_png((20, 30, 220)),
                media_bytes=_png((20, 30, 220)),
            )
            for page_index in page_indexes
        ]


class _PartlyFailingService(_EvidenceService):
    async def fetch_web(self, urls: list[str]) -> WebExtractionBatch:
        now = datetime(2026, 10, 31, tzinfo=UTC)
        return WebExtractionBatch(
            results=[
                WebExtraction(
                    url=urls[0],
                    retrieved_at=now,
                    raw_markdown='One source was extracted.',
                )
            ],
            failures=[
                WebExtractionFailure(
                    url=urls[1],
                    error='The second URL was unavailable.',
                )
            ],
        )


class _FailingOcrService(_EvidenceService):
    async def prepare_pdf_pages(
        self,
        document: ResearchDocument,
        page_indexes: list[int],
    ):
        raise PermissionError('OCR credentials are unavailable')


def _graph(
    actions,
    *,
    max_tool_calls: int = 8,
    counter_limit: int | None = None,
    service=None,
    web_client=None,
):
    predictor = _Predictor(actions)
    evidence_service = service or _EvidenceService()
    budget = TokenBudget(
        counter=_Counter(limit=counter_limit), token_limit=100_000
    )
    graph = ResearchGraph(
        ResearcherNode(
            predictor,
            budget,
            image_budget=ImagePayloadBudget(4, 26214400, 40000000),
        ),
        ResearchToolsNode(evidence_service, web_client or _WebClient()),
        max_tool_calls,
        12,
    )
    return graph, predictor, evidence_service


def test_research_graph_selects_only_inspected_evidence_and_maps_pdf_visuals() -> (
    None
):
    actions = [
        WebSearchAction(kind='web_search', query='eigenvector university'),
        FetchWebAction(kind='fetch_web', urls=['https://example.test/notes']),
        ListEvidenceAction(
            kind='list_evidence', document_position=0, start_position=0
        ),
        ReadEvidenceAction(kind='read_evidence', evidence_positions=[0, 1, 2]),
        OpenPdfAction(kind='open_pdf', url='https://example.test/course.pdf'),
        SearchPdfAction(
            kind='search_pdf', document_position=1, query='projection'
        ),
        ReadPdfPagesAction(
            kind='read_pdf_pages', document_position=1, page_indexes=[1]
        ),
        ReadEvidenceAction(kind='read_evidence', evidence_positions=[4]),
        FinishResearchAction(
            kind='finish',
            brief=(
                'A nonzero eigenvector is scaled by the transformation; '
                'the selected page illustrates projection.'
            ),
            evidence_positions=[0, 4],
        ),
    ]
    graph, predictor, service = _graph(actions)
    state = ResearchState(
        task=ResearchTask(
            question='Explain eigenvectors.',
            context='Focus on projection.',
        ),
        max_tool_calls=8,
        catalog_page_size=12,
    )

    result = asyncio.run(graph.ainvoke(state))

    assert result.executed_tool_calls == 8
    assert service.pdf_reads == [[1]]
    assert [
        item.position for item in result.final_result.selected_evidence
    ] == [0, 4]
    pdf_evidence = result.final_result.selected_evidence[1]
    assert pdf_evidence.page_index == 1
    assert pdf_evidence.media_bytes == _png((20, 30, 220))
    projected_ordinals = {
        evidence.position: evidence.native_image_position
        for evidence in predictor.contexts[-1].inspected_evidence
    }
    assert projected_ordinals[2] == 0
    assert projected_ordinals[4] == 1
    assert result.entry(0).inspected is True
    assert result.entry(1).inspected is True
    assert result.entry(2).inspected is True
    assert result.entry(3).inspected is False
    assert result.entry(4).inspected is True

    web_material = result.final_result.source_materials[0]
    assert [image.url for image in web_material.images] == [
        'https://example.test/diagram.png'
    ]
    assert [_pixel(image) for image in predictor.images[-1]] == [
        (220, 30, 30),
        (20, 30, 220),
    ]


def test_finish_is_allowed_at_zero_tools_and_budget_failure_precedes_prediction() -> (
    None
):
    finish = FinishResearchAction(
        kind='finish',
        brief='No source could be inspected.',
        evidence_positions=[],
    )
    graph, predictor, _ = _graph([finish], max_tool_calls=0)
    state = ResearchState(
        task=ResearchTask(question='Find evidence.', context=''),
        max_tool_calls=0,
        catalog_page_size=4,
    )

    result = asyncio.run(graph.ainvoke(state))

    assert result.final_result.brief == finish.brief
    assert result.executed_tool_calls == 0
    assert len(predictor.contexts) == 1

    too_small_graph, too_small_predictor, _ = _graph(
        [finish], max_tool_calls=0, counter_limit=100_001
    )
    too_small_state = ResearchState(
        task=ResearchTask(question='Find evidence.', context=''),
        max_tool_calls=0,
        catalog_page_size=4,
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(too_small_graph.ainvoke(too_small_state))
    assert too_small_predictor.contexts == []


def test_research_graph_rejects_tool_action_after_cap() -> None:
    graph, predictor, _ = _graph(
        [WebSearchAction(kind='web_search', query='topic')],
        max_tool_calls=0,
    )

    with pytest.raises(RuntimeError, match='tool-call limit'):
        asyncio.run(
            graph.run(ResearchTask(question='Find evidence.', context=''))
        )
    assert len(predictor.contexts) == 1


@pytest.mark.parametrize(
    ('actions', 'service', 'web_client', 'error'),
    [
        (
            [WebSearchAction(kind='web_search', query='topic')],
            None,
            _FailingWebClient(),
            PermissionError,
        ),
        (
            [
                OpenPdfAction(
                    kind='open_pdf', url='https://example.test/notes.pdf'
                ),
                ReadPdfPagesAction(
                    kind='read_pdf_pages', document_position=0, page_indexes=[0]
                ),
            ],
            _FailingOcrService(),
            None,
            PermissionError,
        ),
    ],
)
def test_research_graph_propagates_auth_and_ocr_failures(
    actions,
    service,
    web_client,
    error,
) -> None:
    graph, _, _ = _graph(
        actions,
        max_tool_calls=2,
        service=service,
        web_client=web_client,
    )

    with pytest.raises(error):
        asyncio.run(
            graph.run(ResearchTask(question='Verify this claim.', context=''))
        )


def test_research_graph_retains_partial_failures_and_empty_search_results() -> (
    None
):
    actions = [
        WebSearchAction(kind='web_search', query='topic'),
        FetchWebAction(
            kind='fetch_web',
            urls=[
                'https://example.test/available',
                'https://example.test/unavailable',
            ],
        ),
        FinishResearchAction(
            kind='finish',
            brief='Only one requested page was available; no claim was inspected.',
            evidence_positions=[],
        ),
    ]
    graph, predictor, _ = _graph(
        actions,
        max_tool_calls=2,
        service=_PartlyFailingService(),
    )

    result = asyncio.run(
        graph.run(ResearchTask(question='Check two sources.', context=''))
    )

    assert len(result.failures) == 1
    assert result.failures[0].url == 'https://example.test/unavailable'
    assert result.selected_evidence == []
    assert predictor.contexts[1].observations[0].summary.endswith('No results.')


def test_research_graph_propagates_inference_failure_without_a_brief() -> None:
    budget = TokenBudget(counter=_Counter(), token_limit=100_000)
    graph = ResearchGraph(
        ResearcherNode(
            _FailingPredictor([]),
            budget,
            image_budget=ImagePayloadBudget(4, 26214400, 40000000),
        ),
        ResearchToolsNode(_EvidenceService(), _WebClient()),
        max_tool_calls=1,
        catalog_page_size=4,
    )

    with pytest.raises(RuntimeError, match='research inference failed'):
        asyncio.run(
            graph.run(ResearchTask(question='Find evidence.', context=''))
        )


def test_researcher_rejects_image_payload_before_prediction() -> None:
    image_bytes = _png((120, 80, 40))
    predictor = _Predictor(
        [
            FinishResearchAction(
                kind='finish', brief='Brief.', evidence_positions=[0]
            )
        ]
    )
    node = ResearcherNode(
        predictor,
        TokenBudget(counter=_Counter(), token_limit=100_000),
        image_budget=ImagePayloadBudget(0, 26214400, 40000000),
    )
    state = ResearchState(
        task=ResearchTask(question='Inspect the diagram.', context=''),
        max_tool_calls=1,
        catalog_page_size=4,
        evidence=[
            ResearchEvidenceEntry(
                document_position=0,
                kind='image',
                url='https://example.test/diagram.png',
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                media_bytes=image_bytes,
            )
        ],
        visible_evidence_positions=[0],
        pending_positions=[0],
    )

    with pytest.raises(InputBudgetExceeded, match='Active image evidence'):
        asyncio.run(node.run(state))

    assert predictor.contexts == []
