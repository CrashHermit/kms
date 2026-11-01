"""Budget the complete research context and run one researcher prediction."""

import dspy

from kms2.core.media_budget import ImagePayloadBudget, check_image_payload
from kms2.core.model.research.inference import (
    FinishResearchAction,
    InspectedResearchEvidence,
    ResearchContext,
    ResearchDocumentDescriptor,
    ResearchEvidenceDescriptor,
    ResearchResult,
)
from kms2.core.model.research.retrieval import ResearchEvidenceEntry
from kms2.core.model.source_processing.admission import (
    ResearchImageMaterial,
    ResearchPDFMaterial,
    ResearchPDFPageMaterial,
    ResearchSourceMaterial,
    ResearchWebMaterial,
)
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.research.state import ResearchState
from kms2.module.research.researcher import ResearcherModule


class ResearcherNode:
    """Build a complete private-safe context for one prediction."""

    def __init__(
        self,
        researcher: ResearcherModule,
        token_budget: TokenBudget,
        *,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._researcher = researcher
        self._token_budget = token_budget
        self._image_budget = image_budget

    async def run(self, state: ResearchState) -> ResearchState:
        """Predict one action, marking included evidence inspected on success."""
        context, evidence_images, included = self._input(state)
        serialized = context.model_dump_json()
        token_count = self._token_budget.counter.count_texts([serialized])[0]
        if token_count > self._token_budget.token_limit:
            raise InputBudgetExceeded(
                'Complete research context exceeds the configured input token budget'
            )
        action = await self._researcher.aforward(
            context=context,
            evidence_images=evidence_images,
        )
        state.latest_action = action
        for position in included:
            if position not in state.inspected_positions:
                state.inspected_positions.append(position)
            entry = state.entry(position)
            state.evidence[position] = entry.model_copy(
                update={'inspected': True}
            )
            if position in state.pending_positions:
                state.pending_positions.remove(position)
        if isinstance(action, FinishResearchAction):
            selected_evidence = [
                state.entry(position) for position in action.evidence_positions
            ]
            state.source_materials = _source_materials(state, selected_evidence)
            state.final_result = ResearchResult(
                brief=action.brief,
                selected_evidence=selected_evidence,
                failures=state.failures.copy(),
                source_materials=state.source_materials,
            )
        return state

    def _input(
        self,
        state: ResearchState,
    ) -> tuple[ResearchContext, list[dspy.Image], list[int]]:
        included = list(state.inspected_positions)
        for position in state.pending_positions:
            if position not in included:
                included.append(position)
        image_ordinals: dict[int, int] = {}
        image_bytes: list[bytes] = []
        for position in included:
            entry = state.entry(position)
            if entry.media_bytes is None:
                continue
            image_key = _native_image_key(entry)
            ordinal = next(
                (
                    existing_ordinal
                    for existing_position, existing_ordinal in image_ordinals.items()
                    if _native_image_key(state.entry(existing_position))
                    == image_key
                ),
                None,
            )
            if ordinal is None:
                ordinal = len(image_bytes)
                image_bytes.append(entry.media_bytes)
            image_ordinals[position] = ordinal

        context = ResearchContext(
            task=state.task,
            documents=[
                ResearchDocumentDescriptor(
                    position=position,
                    url=document.url,
                    title=document.title,
                    media_type=document.media_type,
                    page_count=document.page_count,
                )
                for position, document in enumerate(state.documents)
            ],
            evidence=[
                _descriptor(position, state.entry(position))
                for position in state.visible_evidence_positions
            ],
            observations=state.observations.copy(),
            tools_remaining=max(
                0, state.max_tool_calls - state.executed_tool_calls
            ),
            inspected_evidence=[
                _inspected_entry(
                    position,
                    state.entry(position),
                    image_ordinals,
                )
                for position in included
            ],
        )
        check_image_payload(image_bytes, self._image_budget)
        images = [dspy.Image(data) for data in image_bytes]
        return context, images, included


def _native_image_key(entry: ResearchEvidenceEntry) -> tuple[object, ...]:
    """Share an ordinal for repeated renderings of one PDF page."""
    if entry.kind == 'pdf_page':
        return entry.document_position, entry.kind, entry.page_index
    return entry.document_position, entry.kind, entry.url


def _descriptor(
    position: int,
    entry: ResearchEvidenceEntry,
) -> ResearchEvidenceDescriptor:
    """Project private evidence metadata into a catalogue-safe descriptor."""
    return ResearchEvidenceDescriptor(
        position=position,
        document_position=entry.document_position,
        kind=entry.kind,
        title=entry.title,
        url=entry.url,
        page_index=entry.page_index,
        line_start=entry.line_start,
        line_end=entry.line_end,
        token_count=entry.token_count,
        heading_path=entry.heading_path,
        preview=entry.preview,
    )


def _inspected_entry(
    position: int,
    entry: ResearchEvidenceEntry,
    image_ordinals: dict[int, int],
) -> InspectedResearchEvidence:
    """Project one evidence item with its exact native image ordinal."""
    return InspectedResearchEvidence(
        position=position,
        kind=entry.kind,
        url=entry.url,
        title=entry.title,
        retrieved_at=entry.retrieved_at,
        page_index=entry.page_index,
        line_start=entry.line_start,
        line_end=entry.line_end,
        heading_path=entry.heading_path,
        raw_content=entry.raw_content,
        content=entry.content,
        native_image_position=image_ordinals.get(position),
    )


__all__ = ['ResearcherNode']


def _source_materials(
    state: ResearchState,
    selected: list[ResearchEvidenceEntry],
) -> list[ResearchSourceMaterial]:
    """Transfer full captures only for inspected evidence selected by teacher."""
    selected_by_document: dict[int, list[ResearchEvidenceEntry]] = {}
    for entry in selected:
        if entry.inspected:
            selected_by_document.setdefault(entry.document_position, []).append(
                entry
            )

    for document_position, entries in selected_by_document.items():
        selected_positions = {entry.position for entry in entries}
        entries.extend(
            entry
            for entry in state.evidence
            if entry.document_position == document_position
            and entry.kind == 'image'
            and entry.inspected
            and entry.position not in selected_positions
        )

    materials: list[ResearchSourceMaterial] = []
    for document_position, entries in selected_by_document.items():
        document = state.documents[document_position]
        if document.media_type == 'application/pdf':
            pages_by_index: dict[int, ResearchPDFPageMaterial] = {}
            for entry in entries:
                if entry.kind != 'pdf_page':
                    continue
                page_index = entry.page_index
                pages_by_index[page_index] = ResearchPDFPageMaterial(
                    page_index=page_index,
                    markdown=entry.ocr_artifacts.markdown,
                    artifacts=entry.ocr_artifacts,
                    rendered_page=entry.rendered_page_bytes,
                )
            if pages_by_index:
                materials.append(
                    ResearchPDFMaterial(
                        url=document.url,
                        resolved_url=document.resolved_url,
                        title=document.title,
                        retrieved_at=document.retrieved_at,
                        pages=[
                            pages_by_index[index]
                            for index in sorted(pages_by_index)
                        ],
                    )
                )
            continue

        images = [
            ResearchImageMaterial(
                url=entry.url,
                resolved_url=entry.resolved_url,
                retrieved_at=entry.retrieved_at,
                media_bytes=entry.media_bytes,
                original_media_bytes=entry.original_media_bytes,
                original_media_type=entry.original_media_type,
            )
            for entry in entries
            if entry.kind == 'image'
        ]
        if any(entry.kind == 'web_passage' for entry in entries) or images:
            materials.append(
                ResearchWebMaterial(
                    url=document.url,
                    resolved_url=document.resolved_url,
                    title=document.title,
                    retrieved_at=document.retrieved_at,
                    raw_markdown=document.raw_markdown,
                    images=images,
                )
            )
    return materials
