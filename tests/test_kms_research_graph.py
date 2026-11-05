"""Research searches expose text-only catalogs and submit explicit selections."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from kms.core.model.research.inference import (
    ResearchSearchResult,
    ResearchTask,
    SubmitSourcesAction,
    WebSearchAction,
)
from kms.core.model.research.retrieval import WebSearchHit
from kms.core.model.source import Source
from kms.core.model.source_processing.admission import CapturedPDFSource
from kms.core.model.teaching.workflow import TeachingTurnRequest
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.research.tools import ResearchToolsNode
from kms.retrieval import ExternalRetrievalError


@pytest.fixture
def anyio_backend():
    return 'asyncio'


class _Web:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    async def search(self, query):
        self.queries.append(query)
        return self.hits


class _Capture:
    def __init__(self):
        self.urls = []

    async def capture_source(self, url, *, title=None):
        self.urls.append(url)
        if url.endswith('/bad'):
            raise ExternalRetrievalError('capture failed')
        return CapturedPDFSource(
            url=url,
            title=title,
            retrieved_at=datetime.now(UTC),
            data=b'pdf',
        )


class _Admission:
    def __init__(self):
        self.states = []

    async def admit(self, user_uuid, capture):
        self.states.append((user_uuid, capture))
        source = Source(uuid=str(uuid4()), key=capture.title or capture.url)
        return SimpleNamespace(source=source)


def _state() -> TeachingTurnState:
    return TeachingTurnState(
        request=TeachingTurnRequest.model_construct(
            user_uuid='user',
            session_uuid='session',
            expected_revision=0,
            snapshot=None,
            source_uuids=None,
        ),
        research_task=ResearchTask(question='Q', context='C'),
        research_max_tool_calls=8,
    )


def _tools(hits):
    web = _Web(hits)
    capture = _Capture()
    admission = _Admission()
    return ResearchToolsNode(capture, web, admission), web, capture, admission


@pytest.mark.anyio
async def test_search_accumulates_ordered_deduplicated_text_catalog():
    first = WebSearchHit(
        title='First', url='https://example.test/a', snippet='relevant text'
    )
    duplicate = WebSearchHit(
        title='Changed', url=first.url, snippet='duplicate'
    )
    second = WebSearchHit(
        title=None, url='https://example.test/b', snippet='more text'
    )
    node, web, _, _ = _tools([first, duplicate, second])
    state = _state()
    state.latest_action = WebSearchAction(kind='web_search', query='topic')

    await node.run(state)

    assert web.queries == ['topic']
    assert state.search_results == [
        ResearchSearchResult(
            position=0, title='First', url=first.url, snippet='relevant text'
        ),
        ResearchSearchResult(
            position=1, title=None, url=second.url, snippet='more text'
        ),
    ]


def test_research_contract_is_text_only_and_task_has_no_supplied_urls():
    assert ResearchTask.model_fields.keys() == {'question', 'context'}
    assert 'content_images' not in ResearchSearchResult.model_fields
    with pytest.raises(ValidationError):
        ResearchTask(question='Q', context='C', urls=['https://example.test'])


@pytest.mark.anyio
async def test_unselected_results_are_never_captured():
    node, _, capture, admission = _tools(
        [WebSearchHit(title='A', url='https://example.test/a', snippet='A')]
    )
    state = _state()
    state.search_results = [
        ResearchSearchResult(
            position=0, title='A', url='https://example.test/a', snippet='A'
        )
    ]
    state.latest_action = SubmitSourcesAction(
        kind='submit_sources', result_positions=[]
    )

    await node.run(state)

    assert capture.urls == []
    assert admission.states == []


@pytest.mark.anyio
async def test_invalid_positions_reject_whole_selection_before_capture():
    node, _, capture, admission = _tools([])
    state = _state()
    state.search_results = [
        ResearchSearchResult(
            position=0, title='A', url='https://example.test/a', snippet='A'
        )
    ]
    state.latest_action = SubmitSourcesAction(
        kind='submit_sources', result_positions=[0, 5]
    )

    await node.run(state)

    assert capture.urls == []
    assert admission.states == []
    assert state.research_observations[-1].kind == 'invalid_action'


@pytest.mark.anyio
async def test_selected_results_admit_without_source_input_and_skip_successful_duplicates():
    hit = WebSearchHit(title='A', url='https://example.test/a', snippet='A')
    node, _, capture, admission = _tools([hit])
    state = _state()
    state.search_results = [
        ResearchSearchResult(position=0, title='A', url=hit.url, snippet='A')
    ]
    state.latest_action = SubmitSourcesAction(
        kind='submit_sources', result_positions=[0, 0]
    )

    await node.run(state)
    state.latest_action = SubmitSourcesAction(
        kind='submit_sources', result_positions=[0]
    )
    await node.run(state)

    assert capture.urls == [hit.url]
    assert len(admission.states) == 1
    assert admission.states[0][0] == 'user'
    assert admission.states[0][1].url == hit.url
    assert len(state.persisted_source_revisions) == 1


@pytest.mark.anyio
async def test_submission_failure_does_not_block_independent_selected_result():
    urls = ['https://example.test/bad', 'https://example.test/good']
    node, _, capture, admission = _tools(
        [WebSearchHit(title=url, url=url, snippet='snippet') for url in urls]
    )
    state = _state()
    state.search_results = [
        ResearchSearchResult(position=i, title=url, url=url, snippet='snippet')
        for i, url in enumerate(urls)
    ]
    state.latest_action = SubmitSourcesAction(
        kind='submit_sources', result_positions=[0, 1]
    )

    await node.run(state)

    assert capture.urls == urls
    assert len(admission.states) == 1
    assert [failure.url for failure in state.failures] == [urls[0]]
    assert state.persisted_source_revisions


@pytest.mark.anyio
async def test_cancellation_waits_for_source_admission_to_finish():
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class _SlowAdmission:
        async def admit(self, user_uuid, capture):
            entered.set()
            await release.wait()
            finished.set()
            source = Source(uuid=str(uuid4()), key=capture.url)
            return SimpleNamespace(source=source)

    url = 'https://example.test/a'
    tools = ResearchToolsNode(_Capture(), _Web([]), _SlowAdmission())
    state = _state()
    state.search_results = [
        ResearchSearchResult(position=0, title='A', url=url, snippet='A')
    ]
    state.latest_action = SubmitSourcesAction(
        kind='submit_sources', result_positions=[0]
    )
    run = asyncio.create_task(tools.run(state))
    await entered.wait()
    run.cancel()
    await asyncio.sleep(0)
    assert not finished.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await run
    assert finished.is_set()
