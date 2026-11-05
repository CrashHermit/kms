"""Prepare knowledge admission from instruction and its cited graph evidence."""

from typing import assert_never, cast

from kms.core.content import ContentImage, MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget, check_image_payload
from kms.core.model.content import MultimodalContent
from kms.core.model.context import SourceBlockContext
from kms.core.model.knowledge.admission import (
    KnowledgeEvidenceBlock,
    KnowledgeExistingProposition,
    KnowledgeExtractionInput,
    KnowledgeResponseSegment,
)
from kms.core.model.source import SourceRevisionReference
from kms.core.model.teaching.operation import RespondAction
from kms.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
)
from kms.langgraph.knowledge.graph import KnowledgeAdmissionGraph
from kms.langgraph.knowledge.state import KnowledgeState
from kms.langgraph.teaching.state import TeachingTurnState


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
        cited_blocks: set[int] = set()
        cited_images: set[int] = set()
        cited_propositions: set[int] = set()
        segment_citations: list[tuple[str, set[int], set[int], set[int]]] = []
        for segment in response.segments:
            block_positions: set[int] = set()
            image_positions: set[int] = set()
            proposition_positions: set[int] = set()
            for visible_position in segment.cited_evidence_positions:
                receipt_position = interaction.visible_citation_receipt_indexes[
                    visible_position
                ]
                receipt = workspace.receipts[receipt_position]
                if isinstance(receipt, BlockInspection):
                    block_position = next(
                        position
                        for position, entry in enumerate(
                            workspace.block_registry
                        )
                        if entry.block.uuid == receipt.block.uuid
                    )
                    cited_blocks.add(block_position)
                    block_positions.add(block_position)
                elif isinstance(receipt, ImageInspection):
                    image_position = next(
                        position
                        for position, entry in enumerate(
                            workspace.image_registry
                        )
                        if entry.asset.uuid == receipt.asset.uuid
                    )
                    cited_images.add(image_position)
                    image_positions.add(image_position)
                    block_positions.update(
                        workspace.image_registry[image_position].block_positions
                    )
                elif isinstance(receipt, PropositionInspection):
                    proposition_position = next(
                        position
                        for position, entry in enumerate(
                            workspace.proposition_registry
                        )
                        if entry.uuid == receipt.record.proposition.uuid
                    )
                    cited_propositions.add(proposition_position)
                    proposition_positions.add(proposition_position)
                else:
                    assert_never(receipt)
            segment_citations.append(
                (
                    segment.content,
                    block_positions,
                    image_positions,
                    proposition_positions,
                )
            )

        evidence_block_positions = cited_blocks | {
            block_position
            for image_position in cited_images
            for block_position in workspace.image_registry[
                image_position
            ].block_positions
        }
        builder = MultimodalContentBuilder()
        local_image_positions: dict[int, int] = {}
        image_catalog: dict[int, ContentImage] = {}
        for position in sorted(cited_images):
            entry = workspace.image_registry[position]
            payload = entry.data
            assert payload is not None
            image = ContentImage(
                key=('path', entry.asset.path), payload=payload
            )
            image_catalog[position] = image
            material = builder.build(text=None, images=[image])
            local_image_positions[position] = material.image_positions[0]
        evidence_blocks: list[KnowledgeEvidenceBlock] = []
        supporting_block_uuids: list[str] = []
        block_source_revisions: list[SourceRevisionReference] = []
        local_block_positions: dict[int, int] = {}
        for position in sorted(evidence_block_positions):
            entry = workspace.block_registry[position]
            block_images = [
                image_catalog[image_position]
                for image_position in sorted(cited_images)
                if position
                in workspace.image_registry[image_position].block_positions
            ]
            local_block_positions[position] = len(evidence_blocks)
            evidence_blocks.append(
                KnowledgeEvidenceBlock(
                    position=len(evidence_blocks),
                    source_key=entry.source.key,
                    page_indexes=entry.page_indexes,
                    source_block=SourceBlockContext(
                        block_type=entry.block.block_type,
                        material=builder.build(
                            text=(
                                entry.block.content
                                if position in cited_blocks
                                else None
                            ),
                            images=block_images,
                        ),
                    ),
                )
            )
            supporting_block_uuids.append(entry.block.uuid)
            block_source_revisions.append(
                SourceRevisionReference(
                    source_uuid=entry.source.uuid,
                    revision=entry.source.revision,
                )
            )

        existing: list[KnowledgeExistingProposition] = []
        existing_uuids: list[str] = []
        local_proposition_positions: dict[int, int] = {}
        for position in sorted(cited_propositions):
            entry = workspace.proposition_registry[position]
            assert entry.proposition is not None
            local_proposition_positions[position] = len(existing)
            existing.append(
                KnowledgeExistingProposition(
                    position=len(existing),
                    material=MultimodalContent(text=entry.proposition.content),
                )
            )
            existing_uuids.append(entry.uuid)

        request = KnowledgeExtractionInput(
            segments=[
                KnowledgeResponseSegment(
                    material=MultimodalContent(text=content),
                    block_positions=sorted(
                        local_block_positions[position]
                        for position in block_positions
                    ),
                    image_positions=sorted(
                        local_image_positions[position]
                        for position in segment_image_positions
                    ),
                    proposition_positions=sorted(
                        local_proposition_positions[position]
                        for position in proposition_positions
                    ),
                )
                for (
                    content,
                    block_positions,
                    segment_image_positions,
                    proposition_positions,
                ) in segment_citations
            ],
            blocks=evidence_blocks,
            existing_propositions=existing,
        )
        image_bytes = [cast(bytes, image.payload) for image in builder.images]
        check_image_payload(image_bytes, self._image_budget)
        knowledge_state = await self._graph.run(
            KnowledgeState(
                user_uuid=state.request.user_uuid,
                request=request,
                existing_proposition_uuids=existing_uuids,
                supporting_block_uuids=supporting_block_uuids,
                block_source_revisions=block_source_revisions,
                content_image_bytes=image_bytes,
            )
        )
        return {'knowledge_admission': knowledge_state.admission}
