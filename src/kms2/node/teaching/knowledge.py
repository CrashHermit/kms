"""Prepare knowledge admission from instruction and the final evidence packet."""

from kms2.core.media_budget import ImagePayloadBudget, check_image_payload
from kms2.core.model.context import SourceBlockContext
from kms2.core.model.knowledge.admission import (
    KnowledgeEvidenceBlock,
    KnowledgeExistingProposition,
    KnowledgeExtractionInput,
)
from kms2.core.model.source import SourceRevisionReference
from kms2.core.model.teaching.operation import RespondAction
from kms2.langgraph.knowledge.graph import KnowledgeAdmissionGraph
from kms2.langgraph.teaching.state import TeachingTurnState
from kms2.node.teaching.evidence import (
    active_block_positions,
    active_image_positions,
    active_proposition_positions,
)


class TeachingKnowledgeNode:
    """Extract supported subject knowledge from one instructional response."""

    def __init__(
        self,
        graph: KnowledgeAdmissionGraph,
        *,
        image_budget: ImagePayloadBudget,
    ) -> None:
        self._graph = graph
        self._image_budget = image_budget

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Prepare admission only for responses classified as instruction."""
        interaction = state.interaction
        assert interaction is not None
        response: RespondAction = interaction.response
        if response.content_kind == 'coordination':
            return {'knowledge_admission': None}

        workspace = interaction.workspace
        inspected_blocks = active_block_positions(workspace)
        inspected_images = active_image_positions(workspace)
        inspected_propositions = active_proposition_positions(workspace)
        image_positions = {
            position: native_position
            for native_position, position in enumerate(inspected_images)
        }
        evidence_block_positions = set(inspected_blocks)
        for image_position in inspected_images:
            evidence_block_positions.update(
                workspace.image_registry[image_position].block_positions
            )

        evidence_blocks: list[KnowledgeEvidenceBlock] = []
        supporting_block_uuids: list[str] = []
        block_source_revisions: list[SourceRevisionReference] = []
        for position in sorted(evidence_block_positions):
            entry = workspace.block_registry[position]
            active_text = position in inspected_blocks
            local_image_positions = [
                image_positions[image_position]
                for image_position in inspected_images
                if position
                in workspace.image_registry[image_position].block_positions
            ]
            evidence_blocks.append(
                KnowledgeEvidenceBlock(
                    position=len(evidence_blocks),
                    source_key=entry.source.key,
                    page_indexes=entry.page_indexes,
                    source_block=SourceBlockContext(
                        block_type=entry.block.block_type,
                        content=entry.block.content if active_text else None,
                    ),
                    image_positions=local_image_positions,
                )
            )
            supporting_block_uuids.append(entry.block.uuid)
            block_source_revisions.append(
                SourceRevisionReference(
                    source_uuid=entry.source.uuid,
                    revision=entry.source.revision,
                )
            )

        image_bytes: list[bytes] = []
        for position in inspected_images:
            payload = workspace.image_registry[position].data
            assert payload is not None
            image_bytes.append(payload)

        existing: list[KnowledgeExistingProposition] = []
        existing_uuids: list[str] = []
        for position in inspected_propositions:
            entry = workspace.proposition_registry[position]
            assert entry.proposition is not None
            existing.append(
                KnowledgeExistingProposition(
                    position=len(existing), content=entry.proposition.content
                )
            )
            existing_uuids.append(entry.uuid)

        request = KnowledgeExtractionInput(
            response_content=response.content,
            blocks=evidence_blocks,
            existing_propositions=existing,
        )
        check_image_payload(image_bytes, self._image_budget)
        admission = await self._graph.run(
            state.request.user_uuid,
            request,
            existing_proposition_uuids=existing_uuids,
            supporting_block_uuids=supporting_block_uuids,
            block_source_revisions=block_source_revisions,
            evidence_image_bytes=image_bytes,
        )
        return {'knowledge_admission': admission}
