"""Build and atomically persist one complete teacher response."""

from typing import assert_never

from kms.core.model.source import Source, SourceRevisionReference
from kms.core.model.teaching.evidence import TeachingEvidenceReference
from kms.core.model.teaching.operation import RespondAction
from kms.core.model.teaching.session import (
    TeachingResponseSegmentRecord,
    TeachingSessionSnapshot,
    TeachingTurnCommit,
)
from kms.core.model.teaching.workflow import TeachingInteractionResult
from kms.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
)
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.teaching.operations import prepare_instructional_commit
from kms.service.teaching import TeachingService
from kms.service.teaching_workspace import serialize_workspace


class TeachingPersistenceNode:
    """Commit response, source references, observations, and knowledge atomically."""

    def __init__(self, teaching: TeachingService) -> None:
        self._teaching = teaching

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Persist one final response and its prepared knowledge admission."""
        interaction = state.interaction
        assert interaction is not None
        response = interaction.response
        commit = _commit(state.request.snapshot, interaction, state)
        final_view = await self._teaching.record_result(
            state.request.user_uuid,
            state.request.session_uuid,
            response.content,
            commit,
            expected_revision=state.request.expected_revision,
        )
        return {'final_view': final_view}


def _commit(
    snapshot: TeachingSessionSnapshot,
    interaction: TeachingInteractionResult,
    state: TeachingTurnState,
) -> TeachingTurnCommit:
    """Resolve response positions and persist a reference-only workspace."""
    response: RespondAction = interaction.response
    workspace = interaction.workspace
    saved_workspace = serialize_workspace(workspace)
    citations: list[TeachingEvidenceReference] = []
    workspace_sources: list[Source] = []
    response_segments: list[TeachingResponseSegmentRecord] = []
    for segment in response.segments:
        segment_citations: list[TeachingEvidenceReference] = []
        for visible_position in segment.cited_evidence_positions:
            receipt_position = interaction.visible_citation_receipt_indexes[
                visible_position
            ]
            receipt = workspace.receipts[receipt_position]
            if isinstance(receipt, BlockInspection):
                reference = TeachingEvidenceReference(
                    kind='block', uuid=receipt.block.uuid
                )
                workspace_sources.append(receipt.source)
            elif isinstance(receipt, ImageInspection):
                reference = TeachingEvidenceReference(
                    kind='image', uuid=receipt.asset.uuid
                )
                workspace_sources.append(receipt.source)
            elif isinstance(receipt, PropositionInspection):
                reference = TeachingEvidenceReference(
                    kind='proposition',
                    uuid=receipt.record.proposition.uuid,
                )
            else:
                assert_never(receipt)
            citations.append(reference)
            segment_citations.append(reference)
        response_segments.append(
            TeachingResponseSegmentRecord(
                content=segment.content,
                citations=segment_citations,
            )
        )

    source_revisions = {
        (source.uuid, source.revision): SourceRevisionReference(
            source_uuid=source.uuid,
            revision=source.revision,
        )
        for source in workspace_sources
    }
    knowledge = state.knowledge_admission
    if knowledge is not None:
        source_revisions.update(
            {
                (reference.source_uuid, reference.revision): reference
                for reference in knowledge.source_revisions
            }
        )

    commit = prepare_instructional_commit(
        snapshot,
        response,
        visible_turn_indexes=interaction.visible_turn_indexes,
        visible_observation_indexes=interaction.visible_observation_indexes,
        learner_observation_uuids=[
            entry.observation.uuid
            for entry in interaction.workspace.learner_observation_registry
        ],
        prepared_observations=state.prepared_observations,
    )
    return commit.model_copy(
        update={
            'workspace': saved_workspace,
            'source_revisions': list(source_revisions.values()),
            'knowledge': knowledge,
            'segments': response_segments,
            'citations': citations,
            'learner_observations': state.prepared_observations,
        }
    )
