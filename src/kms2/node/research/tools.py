"""Execute one bounded retrieval or evidence-read action."""

from kms2.core.model.research.inference import (
    FetchWebAction,
    ListEvidenceAction,
    OpenPdfAction,
    ReadEvidenceAction,
    ReadPdfPagesAction,
    ResearchFailure,
    ResearchToolObservation,
    SearchPdfAction,
    WebSearchAction,
)
from kms2.core.model.research.retrieval import ResearchDocument
from kms2.langgraph.research.state import ResearchState
from kms2.retrieval import ExternalRetrievalError, TavilyWebClient
from kms2.service.research_evidence import ResearchEvidenceService


class ResearchToolsNode:
    """Execute research tools and register invocation-owned positions."""

    def __init__(
        self,
        service: ResearchEvidenceService,
        web_client: TavilyWebClient,
    ) -> None:
        self._service = service
        self._web_client = web_client

    async def run(self, state: ResearchState) -> ResearchState:
        """Execute the selected tool exactly once and append its observation."""
        action = state.latest_action
        assert action is not None
        state.executed_tool_calls += 1
        state.visible_evidence_positions = []
        if isinstance(action, WebSearchAction):
            await self._search_web(state, action)
        elif isinstance(action, FetchWebAction):
            await self._fetch_web(state, action)
        elif isinstance(action, ListEvidenceAction):
            self._list_evidence(state, action)
        elif isinstance(action, ReadEvidenceAction):
            await self._read_evidence(state, action)
        elif isinstance(action, OpenPdfAction):
            await self._open_pdf(state, action)
        elif isinstance(action, SearchPdfAction):
            await self._search_pdf(state, action)
        elif isinstance(action, ReadPdfPagesAction):
            await self._read_pdf_pages(state, action)
        else:
            raise AssertionError(
                f'Unexpected research tool action: {action.kind}'
            )
        return state

    async def _search_web(
        self,
        state: ResearchState,
        action: WebSearchAction,
    ) -> None:
        hits = await self._web_client.search(action.query)
        state.search_titles.update((hit.url, hit.title) for hit in hits)
        lines = [
            f'{hit.title or hit.url} — {hit.url}: {hit.snippet}' for hit in hits
        ]
        state.observations.append(
            ResearchToolObservation(
                kind='web_search',
                summary='Search discovery results (not inspected evidence):\n'
                + ('\n'.join(lines) if lines else 'No results.'),
            )
        )

    async def _fetch_web(
        self,
        state: ResearchState,
        action: FetchWebAction,
    ) -> None:
        requested_urls = list(dict.fromkeys(action.urls))
        new_urls = [
            url for url in requested_urls if url not in state.document_positions
        ]
        document_positions = [
            state.document_positions[url]
            for url in requested_urls
            if url in state.document_positions
        ]
        evidence_positions: list[int] = []
        failures: list[ResearchFailure] = []
        extractions = []
        if new_urls:
            try:
                batch = await self._service.fetch_web(new_urls)
            except ExternalRetrievalError as error:
                failures.extend(
                    ResearchFailure(
                        operation='fetch_web',
                        url=url,
                        message=str(error),
                    )
                    for url in new_urls
                )
                self._record_observation(
                    state,
                    'fetch_web',
                    'The requested web URLs could not be extracted.',
                    document_positions=document_positions,
                    failures=failures,
                )
                return
            extractions = batch.results
            failures.extend(
                ResearchFailure(
                    operation='fetch_web',
                    url=failure.url,
                    message=failure.error,
                )
                for failure in batch.failures
            )

        for extraction in extractions:
            document = ResearchDocument(
                url=extraction.url,
                title=state.search_titles.get(extraction.url),
                retrieved_at=extraction.retrieved_at,
                media_type='text/markdown',
                raw_markdown=extraction.raw_markdown,
                image_urls=extraction.image_urls,
            )
            document_position = state.register_document(document)
            if document_position not in document_positions:
                document_positions.append(document_position)
            discovered = await self._service.discover_web(
                document,
                document_position,
            )
            evidence_positions.extend(
                state.register_evidence(entry) for entry in discovered
            )

        self._record_observation(
            state,
            'fetch_web',
            'Extracted selected web URLs; passage and image descriptors are discovery only.',
            document_positions=document_positions,
            evidence_positions=evidence_positions,
            failures=failures,
        )

    def _list_evidence(
        self,
        state: ResearchState,
        action: ListEvidenceAction,
    ) -> None:
        selected = [
            (position, entry)
            for position, entry in enumerate(state.evidence)
            if entry.document_position == action.document_position
            and position >= action.start_position
        ][: state.catalog_page_size]
        positions = [position for position, _ in selected]
        state.visible_evidence_positions = positions
        lines = [
            f'position={position} kind={entry.kind} '
            f'lines={entry.line_start}:{entry.line_end} page_index={entry.page_index} '
            f'tokens={entry.token_count} '
            f'heading={" / ".join(entry.heading_path)} '
            f'preview={entry.preview}'
            for position, entry in selected
        ]
        self._record_observation(
            state,
            'list_evidence',
            'Uninspected catalogue discovery:\n'
            + ('\n'.join(lines) if lines else 'No descriptors on this page.'),
            document_positions=[action.document_position],
            evidence_positions=positions,
        )

    async def _read_evidence(
        self,
        state: ResearchState,
        action: ReadEvidenceAction,
    ) -> None:
        output_positions: list[int] = []
        failures: list[ResearchFailure] = []
        for position in action.evidence_positions:
            entry = state.entry(position)
            try:
                if entry.content is not None or entry.media_bytes is not None:
                    updated = entry
                else:
                    result = await self._service.read_evidence([entry])
                    updated = result[0].model_copy(
                        update={
                            'position': position,
                            'document_position': entry.document_position,
                        }
                    )
            except ExternalRetrievalError as error:
                failures.append(
                    ResearchFailure(
                        operation='read_evidence',
                        url=entry.url,
                        message=str(error),
                    )
                )
                continue
            state.evidence[position] = updated
            if position not in state.pending_positions:
                state.pending_positions.append(position)
            output_positions.append(position)
        state.visible_evidence_positions = output_positions
        self._record_observation(
            state,
            'read_evidence',
            self._read_summary(state, output_positions),
            evidence_positions=output_positions,
            failures=failures,
        )

    async def _open_pdf(
        self,
        state: ResearchState,
        action: OpenPdfAction,
    ) -> None:
        existing = state.document_positions.get(action.url)
        if existing is not None:
            document = state.documents[existing]
        else:
            try:
                document = await self._service.open_pdf(action.url)
            except ExternalRetrievalError as error:
                failure = ResearchFailure(
                    operation='open_pdf', url=action.url, message=str(error)
                )
                self._record_observation(
                    state,
                    'open_pdf',
                    'The PDF could not be opened.',
                    failures=[failure],
                )
                return
            document_position = state.register_document(document)
            document_positions = [document_position]
        if existing is not None:
            document_positions = [existing]
        self._record_observation(
            state,
            'open_pdf',
            f'Opened PDF metadata only: {document.page_count} pages; no page inspected.',
            document_positions=document_positions,
        )

    async def _search_pdf(
        self,
        state: ResearchState,
        action: SearchPdfAction,
    ) -> None:
        document = state.documents[action.document_position]
        result = await self._service.search_pdf(document, action.query)
        matches = '\n'.join(
            f'page_index={match.page_index}: {match.snippet}'
            for match in result.matches
        )
        suffix = ' (results truncated)' if result.truncated else ''
        self._record_observation(
            state,
            'search_pdf',
            'Native PDF text locator results; not inspected evidence'
            f'{suffix}:\n{matches or "No native-text matches."}',
            document_positions=[action.document_position],
        )

    async def _read_pdf_pages(
        self,
        state: ResearchState,
        action: ReadPdfPagesAction,
    ) -> None:
        document = state.documents[action.document_position]
        cached_positions = [
            position
            for position, entry in enumerate(state.evidence)
            if entry.document_position == action.document_position
            and entry.kind == 'pdf_page'
            and entry.page_index in action.page_indexes
        ]
        cached_pages = {
            state.entry(position).page_index for position in cached_positions
        }
        missing_pages = [
            page_index
            for page_index in action.page_indexes
            if page_index not in cached_pages
        ]
        positions = cached_positions
        if missing_pages:
            try:
                entries = await self._service.prepare_pdf_pages(
                    document,
                    missing_pages,
                )
            except ExternalRetrievalError as error:
                failure = ResearchFailure(
                    operation='read_pdf_pages',
                    url=document.url,
                    message=str(error),
                )
                self._record_observation(
                    state,
                    'read_pdf_pages',
                    'The selected PDF pages could not be read.',
                    document_positions=[action.document_position],
                    failures=[failure],
                )
                return
            for entry in entries:
                registered = entry.model_copy(
                    update={'document_position': action.document_position}
                )
                position = state.register_evidence(registered)
                if position not in state.pending_positions:
                    state.pending_positions.append(position)
                positions.append(position)
        state.visible_evidence_positions = positions
        self._record_observation(
            state,
            'read_pdf_pages',
            self._read_summary(state, positions),
            document_positions=[action.document_position],
            evidence_positions=positions,
        )

    def _read_summary(self, state: ResearchState, positions: list[int]) -> str:
        sections = []
        for position in positions:
            entry = state.entry(position)
            content = (
                entry.content
                if entry.content is not None
                else entry.raw_content or ''
            )
            sections.append(
                f'Evidence position {position}, kind={entry.kind}, URL={entry.url}, '
                f'page_index={entry.page_index}, lines={entry.line_start}:{entry.line_end}:\n'
                f'{content}'
            )
        return (
            'Complete selected evidence, supplied for inspection:\n'
            + '\n\n'.join(sections)
        )

    @staticmethod
    def _record_observation(
        state: ResearchState,
        kind: str,
        summary: str,
        *,
        document_positions: list[int] | None = None,
        evidence_positions: list[int] | None = None,
        failures: list[ResearchFailure] | None = None,
    ) -> None:
        observed_failures = failures or []
        state.failures.extend(observed_failures)
        state.observations.append(
            ResearchToolObservation(
                kind=kind,
                summary=summary,
                document_positions=document_positions or [],
                evidence_positions=evidence_positions or [],
                failures=observed_failures,
            )
        )


__all__ = ['ResearchToolsNode']
