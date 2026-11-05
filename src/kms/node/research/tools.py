"""Search the web and admit explicitly selected results."""

import asyncio

from kms.core.model.research.inference import (
    ResearchFailure,
    ResearchSearchResult,
    ResearchToolObservation,
    SubmitSourcesAction,
    WebSearchAction,
)
from kms.core.model.source import SourceRevisionReference
from kms.core.source_snapshot import SourceSnapshotConflict
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.langgraph.teaching.state import TeachingTurnState
from kms.ocr.mistral import MistralOCRError
from kms.retrieval import ExternalRetrievalError, TavilyWebClient
from kms.service.research_source_capture import ResearchSourceCaptureService
from kms.service.source_admission import SourceAdmissionService


class ResearchToolsNode:
    """Run bounded searches and persist only researcher-selected URLs."""

    def __init__(
        self,
        capture_service: ResearchSourceCaptureService,
        web_client: TavilyWebClient,
        source_admission: SourceAdmissionService,
    ) -> None:
        self._capture_service = capture_service
        self._web_client = web_client
        self._source_admission = source_admission

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        action = state.latest_action
        assert action is not None
        state.research_tool_calls += 1
        if isinstance(action, WebSearchAction):
            await self._search_web(state, action)
        elif isinstance(action, SubmitSourcesAction):
            await self._submit_sources(state, action)
        else:
            raise AssertionError(
                f'Unexpected research tool action: {action.kind}'
            )
        return {
            'research_tool_calls': state.research_tool_calls,
            'search_results': state.search_results,
            'failures': state.failures,
            'submitted_urls': state.submitted_urls,
            'persisted_source_revisions': state.persisted_source_revisions,
            'research_observations': state.research_observations,
        }

    async def _search_web(
        self, state: TeachingTurnState, action: WebSearchAction
    ) -> None:
        hits = await self._web_client.search(action.query)
        for hit in hits:
            if any(result.url == hit.url for result in state.search_results):
                continue
            state.search_results.append(
                ResearchSearchResult(
                    position=len(state.search_results),
                    title=hit.title,
                    url=hit.url,
                    snippet=hit.snippet,
                )
            )
        self._record(
            state,
            'web_search',
            f'Added {len(state.search_results)} ordered search results; snippets are discovery-only.',
        )

    async def _submit_sources(
        self, state: TeachingTurnState, action: SubmitSourcesAction
    ) -> None:
        offered = {result.position: result for result in state.search_results}
        selected_positions = list(dict.fromkeys(action.result_positions))
        invalid = set(selected_positions) - offered.keys()
        if invalid:
            self._record(
                state,
                'invalid_action',
                f'Result positions were not offered: {sorted(invalid)}.',
            )
            return

        failures: list[ResearchFailure] = []
        submitted: list[int] = []
        for position in selected_positions:
            result = offered[position]
            if result.url in state.submitted_urls:
                continue
            try:
                capture = await self._capture_service.capture_source(
                    result.url, title=result.title
                )
                admission = asyncio.create_task(
                    self._source_admission.admit(
                        state.request.user_uuid, capture
                    )
                )
                try:
                    admitted = await asyncio.shield(admission)
                except asyncio.CancelledError:
                    await admission
                    raise
            except (
                ExternalRetrievalError,
                InputBudgetExceeded,
                MistralOCRError,
                SourceSnapshotConflict,
            ) as error:
                failure = ResearchFailure(
                    operation='submit_sources',
                    url=result.url,
                    message=str(error),
                )
                failures.append(failure)
                continue
            state.submitted_urls.add(result.url)
            submitted.append(position)
            reference = SourceRevisionReference(
                source_uuid=admitted.source.uuid,
                revision=admitted.source.revision,
            )
            if reference not in state.persisted_source_revisions:
                state.persisted_source_revisions.append(reference)

        self._record(
            state,
            'submit_sources',
            f'Persisted {len(submitted)} selected search results.',
            result_positions=submitted,
            failures=failures,
        )

    @staticmethod
    def _record(
        state: TeachingTurnState,
        kind: str,
        summary: str,
        *,
        result_positions: list[int] | None = None,
        failures: list[ResearchFailure] | None = None,
    ) -> None:
        observed_failures = failures or []
        state.failures.extend(
            failure
            for failure in observed_failures
            if failure not in state.failures
        )
        state.research_observations.append(
            ResearchToolObservation(
                kind=kind,
                summary=summary,
                result_positions=result_positions or [],
                failures=observed_failures,
            )
        )


__all__ = ['ResearchToolsNode']
