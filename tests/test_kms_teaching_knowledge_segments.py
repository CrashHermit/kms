"""Segment-scoped evidence passed to instructional knowledge extraction."""

import asyncio
from types import SimpleNamespace

from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.block import SourceBlock
from kms.core.model.block_types import BlockType
from kms.core.model.source import Source
from kms.core.model.source_workspace import SourceWorkspaceBlockEntry
from kms.core.model.teaching.operation import (
    Clarify,
    RespondAction,
    TeachingResponseSegment,
)
from kms.core.model.teaching.workspace import (
    BlockInspection,
    TeachingEvidenceWorkspace,
)
from kms.node.teaching.knowledge import TeachingKnowledgeNode


class _KnowledgeGraphCapture:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def run(self, state):
        self.calls.append(state)
        return state


def _state(workspace: TeachingEvidenceWorkspace) -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(user_uuid='user-1'),
        interaction=SimpleNamespace(
            response=RespondAction(
                segments=[
                    TeachingResponseSegment(
                        content='First supported claim.',
                        cited_evidence_positions=[0],
                    ),
                    TeachingResponseSegment(
                        content='Second supported claim.',
                        cited_evidence_positions=[1],
                    ),
                    TeachingResponseSegment(content='A separate sentence.'),
                ],
                content_kind='instruction',
                operation=Clarify(kind='clarify'),
            ),
            workspace=workspace,
            visible_citation_receipt_indexes=[0, 1, 2],
        ),
    )


def test_knowledge_input_keeps_citations_with_their_response_segments() -> None:
    source = Source(uuid='source', key='notes')
    blocks = [
        SourceBlock(
            uuid=f'block-{position}',
            block_type=BlockType.PARAGRAPH,
            content=f'Evidence {position}.',
        )
        for position in range(3)
    ]
    workspace = TeachingEvidenceWorkspace(
        block_registry=[
            SourceWorkspaceBlockEntry(
                source=source,
                page_indexes=[0],
                block=block,
            )
            for block in blocks
        ],
        receipts=[
            BlockInspection(source=source, block=block, page_indexes=[0])
            for block in blocks
        ],
    )
    graph = _KnowledgeGraphCapture()

    asyncio.run(
        TeachingKnowledgeNode(
            graph,
            image_budget=ImagePayloadBudget(
                max_images=1, max_bytes=100_000, max_pixels=100
            ),
        ).run(_state(workspace))
    )

    request = graph.calls[0].request
    assert [segment.material.text for segment in request.segments] == [
        'First supported claim.',
        'Second supported claim.',
        'A separate sentence.',
    ]
    assert [segment.block_positions for segment in request.segments] == [
        [0],
        [1],
        [],
    ]
    assert [segment.image_positions for segment in request.segments] == [
        [],
        [],
        [],
    ]
    assert [block.source_block.material.text for block in request.blocks] == [
        'Evidence 0.',
        'Evidence 1.',
    ]


def test_coordination_response_is_not_submitted_for_knowledge_extraction() -> (
    None
):
    workspace = TeachingEvidenceWorkspace()
    state = _state(workspace)
    state.interaction.response = RespondAction(
        segments=[TeachingResponseSegment(content='Let us continue.')],
        content_kind='coordination',
        operation=Clarify(kind='clarify'),
    )
    graph = _KnowledgeGraphCapture()

    result = asyncio.run(
        TeachingKnowledgeNode(
            graph,
            image_budget=ImagePayloadBudget(
                max_images=1, max_bytes=100_000, max_pixels=100
            ),
        ).run(state)
    )

    assert result == {'knowledge_admission': None}
    assert graph.calls == []
