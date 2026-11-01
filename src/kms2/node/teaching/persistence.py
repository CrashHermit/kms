"""Build and atomically persist one complete teacher response."""

from typing import assert_never

from kms2.core.model.source import Source, SourceRevisionReference
from kms2.core.model.teaching.evidence import TeachingEvidenceReference
from kms2.core.model.teaching.operation import RespondAction
from kms2.core.model.teaching.session import (
    TeachingSessionSnapshot,
    TeachingTurnCommit,
)
from kms2.core.model.teaching.workflow import TeachingInteractionResult
from kms2.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
)
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.node.teaching.operations import prepare_instructional_commit
from kms2.service.teaching import TeachingService
from kms2.service.teaching_workspace import serialize_workspace


class TeachingPersistenceNode:
    """Commit the final response, observations, sources, and knowledge together."""

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
    """Resolve response-local positions to stable durable references."""
    response: RespondAction = interaction.response
    workspace = interaction.workspace
    citations: list[TeachingEvidenceReference] = []
    source_snapshots: list[Source] = [
        acquired.source
        for acquired in workspace.acquired_sources
        if acquired.kind == 'owned'
    ]
    used_source_revisions: set[tuple[str, int]] = set()
    for visible_position in response.cited_evidence_positions:
        receipt_position = interaction.visible_citation_receipt_indexes[
            visible_position
        ]
        receipt = workspace.receipts[receipt_position]
        if isinstance(receipt, BlockInspection):
            citations.append(
                TeachingEvidenceReference(kind='block', uuid=receipt.block.uuid)
            )
            source_snapshots.append(receipt.source)
            used_source_revisions.add(
                (receipt.source.uuid, receipt.source.revision)
            )
        elif isinstance(receipt, ImageInspection):
            citations.append(
                TeachingEvidenceReference(kind='image', uuid=receipt.asset.uuid)
            )
            source_snapshots.append(receipt.source)
            used_source_revisions.add(
                (receipt.source.uuid, receipt.source.revision)
            )
        elif isinstance(receipt, PropositionInspection):
            citations.append(
                TeachingEvidenceReference(
                    kind='proposition',
                    uuid=receipt.record.proposition.uuid,
                )
            )
        else:
            assert_never(receipt)

    source_revisions = {
        (source.uuid, source.revision): SourceRevisionReference(
            source_uuid=source.uuid,
            revision=source.revision,
        )
        for source in source_snapshots
    }
    knowledge = state.knowledge_admission
    if knowledge is not None:
        used_source_revisions.update(
            (reference.source_uuid, reference.revision)
            for reference in knowledge.source_revisions
        )
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
        learner_history=state.learner_history,
        prepared_observations=state.prepared_observations,
    )
    return commit.model_copy(
        update={
            'workspace_json': serialize_workspace(workspace),
            'source_revisions': list(source_revisions.values()),
            'knowledge': knowledge,
            'citations': citations,
            'learner_observations': state.prepared_observations,
            'staged_sources': [
                staged
                for staged in workspace.staged_sources
                if (staged.source.uuid, staged.source.revision)
                in used_source_revisions
            ],
        }
    )
