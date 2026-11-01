"""Execute one local evidence action against an invocation workspace."""

import asyncio
from pathlib import Path

from kms2.core.model.block import SourceBlock
from kms2.core.model.source_evidence import SourceImage, SourcePassage
from kms2.core.model.teaching.inference import (
    FocusEvidenceAction,
    ListAcquiredBlocksAction,
    ReadImagesAction,
    ReadPassagesAction,
    ReadProceduresAction,
    ReadPropositionsAction,
    ReadReasoningStepsAction,
    ReadStatementsAction,
    SearchProceduresAction,
    SearchPropositionsAction,
    SearchSourcesAction,
    SearchStatementsAction,
    TeachingContextGroup,
    TeachingToolObservation,
)
from kms2.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
    TeachingEvidenceWorkspace,
    TeachingProcedureRegistryEntry,
    TeachingStatementRegistryEntry,
)
from kms2.node.teaching.evidence import (
    append_unique,
    context_group,
    register_block,
    register_group,
    register_image,
    register_occurrence_blocks,
    register_proposition,
    register_reasoning_refs,
    staged_source_passages,
)
from kms2.service.knowledge import KnowledgeService
from kms2.service.knowledge_search import KnowledgeSearchService
from kms2.service.source_evidence import (
    SourceEvidenceService,
    SourceSearchService,
)


class TeachingEvidenceTools:
    """Run local evidence actions, mutating only their private workspace."""

    def __init__(
        self,
        source_search: SourceSearchService,
        source_evidence: SourceEvidenceService,
        knowledge_search: KnowledgeSearchService,
        knowledge_service: KnowledgeService,
        *,
        discovery_preview_characters: int,
        catalog_page_size: int,
    ) -> None:
        self._source_search = source_search
        self._source_evidence = source_evidence
        self._knowledge_search = knowledge_search
        self._knowledge_service = knowledge_service
        self._discovery_preview_characters = discovery_preview_characters
        self._catalog_page_size = catalog_page_size

    @property
    def catalog_page_size(self) -> int:
        """Return the configured maximum number of catalog blocks per page."""
        return self._catalog_page_size

    async def search_sources(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: SearchSourcesAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        hits = await self._source_search.search_sources(
            user_uuid, action.query, source_uuids=source_uuids
        )
        blocks: list[int] = []
        images: list[int] = []
        for hit in hits:
            position = register_block(
                workspace, hit.source, hit.block, hit.page_indexes
            )
            blocks.append(position)
            images.extend(
                register_image(workspace, asset, hit.source, position)
                for asset in hit.block.assets
            )
        workspace.discovered_block_positions = blocks
        workspace.discovered_image_positions = list(dict.fromkeys(images))
        return _observation('search_sources', action.query, blocks, images)

    async def read_passages(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadPassagesAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.block_registry[position]
            for position in action.block_positions
        ]
        passages = await _read_passages(
            self._source_evidence, user_uuid, entries, action, workspace
        )
        block_positions: list[int] = []
        image_positions: list[int] = []
        groups: list[TeachingContextGroup] = []
        for passage in passages:
            local_blocks: dict[str, int] = {}
            for evidence in passage.blocks:
                position = register_block(
                    workspace,
                    passage.source,
                    evidence.block,
                    evidence.page_indexes,
                )
                local_blocks[evidence.block.uuid] = position
                append_unique(block_positions, position)
                workspace.inspected_block_positions.add(position)
                _add_receipt(
                    workspace,
                    BlockInspection(
                        source=passage.source,
                        block=evidence.block,
                        page_indexes=evidence.page_indexes,
                    ),
                )
                for asset in evidence.block.assets:
                    append_unique(
                        image_positions,
                        register_image(
                            workspace, asset, passage.source, position
                        ),
                    )
            local_groups = {
                group.uuid: register_group(workspace, passage.source, group)
                for group in passage.groups
            }
            groups.extend(
                context_group(
                    group, local_groups[group.uuid], local_blocks, local_groups
                )
                for group in passage.groups
            )
        workspace.active_block_positions = block_positions
        return _observation(
            'read_passages',
            None,
            block_positions,
            image_positions,
            groups=groups,
        )

    async def read_images(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadImagesAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.image_registry[position]
            for position in action.image_positions
        ]
        images: list[SourceImage | None] = [None] * len(entries)
        owned_groups: dict[tuple[str, int], list[int]] = {}
        for index, entry in enumerate(entries):
            if _is_staged(workspace, entry.source):
                images[index] = SourceImage(
                    asset=entry.asset,
                    data=await asyncio.to_thread(
                        Path(entry.asset.path).read_bytes
                    ),
                )
            else:
                owned_groups.setdefault(
                    (entry.source.uuid, entry.source.revision), []
                ).append(index)
        for (source_uuid, revision), indexes in owned_groups.items():
            group_images = await self._source_evidence.read_images_at_revisions(
                user_uuid,
                [
                    (source_uuid, revision, entries[index].asset.uuid)
                    for index in indexes
                ],
            )
            for index, image in zip(indexes, group_images, strict=True):
                images[index] = image
        blocks: list[int] = []
        positions: list[int] = []
        for requested_position, image in zip(
            action.image_positions, images, strict=True
        ):
            assert image is not None
            entry = workspace.image_registry[requested_position]
            assert entry.asset.uuid == image.asset.uuid
            entry.data = image.data
            workspace.inspected_image_positions.add(requested_position)
            append_unique(positions, requested_position)
            for block_position in entry.block_positions:
                append_unique(blocks, block_position)
            _add_receipt(
                workspace,
                ImageInspection(
                    source=entry.source,
                    asset=entry.asset,
                    block_uuids=[
                        workspace.block_registry[pos].block.uuid
                        for pos in entry.block_positions
                    ],
                    data=image.data,
                ),
            )
        workspace.active_image_positions = positions
        return _observation('read_images', None, blocks, positions)

    async def search_propositions(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: SearchPropositionsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        hits = await self._knowledge_search.search_propositions(
            user_uuid, action.query
        )
        positions = [
            register_proposition(workspace, hit.proposition) for hit in hits
        ]
        workspace.discovered_proposition_positions = positions
        return _observation(
            'search_propositions', action.query, [], [], propositions=positions
        )

    async def read_propositions(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadPropositionsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.proposition_registry[position]
            for position in action.proposition_positions
        ]
        records = await self._knowledge_service.read_propositions(
            user_uuid, [entry.uuid for entry in entries]
        )
        block_positions: list[int] = []
        for position, record in zip(
            action.proposition_positions, records, strict=True
        ):
            entry = workspace.proposition_registry[position]
            entry.record = record
            entry.proposition = record.proposition
            workspace.inspected_proposition_positions.add(position)
            _add_receipt(workspace, PropositionInspection(record=record))
            for assertion in record.assertions:
                for block_uuid in assertion.supporting_block_uuids:
                    block_positions.extend(
                        await self._register_support_block(
                            user_uuid, workspace, block_uuid
                        )
                    )
        register_reasoning_refs(
            workspace,
            [
                uuid
                for record in records
                for uuid in record.reasoning_step_uuids
            ],
        )
        workspace.active_proposition_positions = list(
            action.proposition_positions
        )
        return _observation(
            'read_propositions',
            None,
            block_positions,
            [],
            propositions=list(action.proposition_positions),
            reasoning=list(workspace.discovered_reasoning_positions),
        )

    async def read_reasoning_steps(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadReasoningStepsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.reasoning_registry[position]
            for position in action.reasoning_positions
        ]
        records = await self._knowledge_service.read_reasoning_steps(
            user_uuid, [entry.uuid for entry in entries]
        )
        proposition_uuids: list[str] = []
        support_blocks: list[int] = []
        for position, record in zip(
            action.reasoning_positions, records, strict=True
        ):
            workspace.reasoning_registry[position].record = record
            workspace.inspected_reasoning_positions.add(position)
            for uuid in [*record.premise_uuids, record.conclusion_uuid]:
                append_unique(proposition_uuids, uuid)
                register_proposition(workspace, uuid)
            for uuid in record.supporting_block_uuids:
                support_blocks.extend(
                    await self._register_support_block(
                        user_uuid, workspace, uuid
                    )
                )
        proposition_records = await self._knowledge_service.read_propositions(
            user_uuid, list(dict.fromkeys(proposition_uuids))
        )
        proposition_positions: list[int] = []
        for record in proposition_records:
            position = register_proposition(workspace, record.proposition)
            workspace.proposition_registry[position].record = record
            workspace.inspected_proposition_positions.add(position)
            _add_receipt(workspace, PropositionInspection(record=record))
            append_unique(proposition_positions, position)
        register_reasoning_refs(
            workspace,
            [
                uuid
                for record in proposition_records
                for uuid in record.reasoning_step_uuids
            ],
        )
        workspace.active_reasoning_positions = list(action.reasoning_positions)
        workspace.active_proposition_positions = proposition_positions
        return _observation(
            'read_reasoning_steps',
            None,
            support_blocks,
            [],
            propositions=proposition_positions,
            reasoning=list(action.reasoning_positions),
        )

    async def search_statements(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: SearchStatementsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        hits = await self._source_search.search_statements(
            user_uuid,
            action.query,
            source_uuids=source_uuids,
            is_exercise=action.is_exercise,
        )
        start = len(workspace.statement_registry)
        workspace.statement_registry.extend(
            TeachingStatementRegistryEntry(
                source=hit.source,
                page_indexes=hit.page_indexes,
                statement=hit.statement,
            )
            for hit in hits
        )
        positions = list(range(start, len(workspace.statement_registry)))
        workspace.discovered_statement_positions = positions
        return _observation(
            'search_statements', action.query, [], [], statements=positions
        )

    async def read_statements(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadStatementsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.statement_registry[position]
            for position in action.statement_positions
        ]
        records = await self._source_evidence.read_statements(
            user_uuid, [entry.statement.uuid for entry in entries]
        )
        blocks: list[int] = []
        for position, record in zip(
            action.statement_positions, records, strict=True
        ):
            workspace.statement_registry[position].passage = record
            workspace.inspected_statement_positions.add(position)
            register_occurrence_blocks(
                workspace, record.passages, inspected=True
            )
            for passage in record.passages:
                blocks.extend(_receipt_passage(workspace, passage))
        workspace.active_block_positions = list(dict.fromkeys(blocks))
        workspace.active_statement_positions = list(action.statement_positions)
        return _observation(
            'read_statements',
            None,
            workspace.active_block_positions,
            [],
            statements=list(action.statement_positions),
        )

    async def search_procedures(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: SearchProceduresAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        hits = await self._source_search.search_procedures(
            user_uuid, action.query, source_uuids=source_uuids
        )
        start = len(workspace.procedure_registry)
        workspace.procedure_registry.extend(
            TeachingProcedureRegistryEntry(
                source=hit.source,
                page_indexes=hit.page_indexes,
                procedure=hit.procedure,
            )
            for hit in hits
        )
        positions = list(range(start, len(workspace.procedure_registry)))
        workspace.discovered_procedure_positions = positions
        return _observation(
            'search_procedures', action.query, [], [], procedures=positions
        )

    async def read_procedures(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadProceduresAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.procedure_registry[position]
            for position in action.procedure_positions
        ]
        records = await self._source_evidence.read_procedures(
            user_uuid, [entry.procedure.uuid for entry in entries]
        )
        blocks: list[int] = []
        for position, record in zip(
            action.procedure_positions, records, strict=True
        ):
            workspace.procedure_registry[position].passage = record
            workspace.inspected_procedure_positions.add(position)
            register_occurrence_blocks(
                workspace, record.passages, inspected=True
            )
            for passage in record.passages:
                blocks.extend(_receipt_passage(workspace, passage))
        workspace.active_block_positions = list(dict.fromkeys(blocks))
        workspace.active_procedure_positions = list(action.procedure_positions)
        return _observation(
            'read_procedures',
            None,
            workspace.active_block_positions,
            [],
            procedures=list(action.procedure_positions),
        )

    async def focus_evidence(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: FocusEvidenceAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        workspace.active_block_positions = list(action.block_positions)
        workspace.active_image_positions = list(action.image_positions)
        workspace.active_proposition_positions = list(
            action.proposition_positions
        )
        workspace.active_reasoning_positions = list(action.reasoning_positions)
        workspace.active_statement_positions = list(action.statement_positions)
        workspace.active_procedure_positions = list(action.procedure_positions)
        return _observation(
            'focus_evidence',
            None,
            workspace.active_block_positions,
            workspace.active_image_positions,
            propositions=workspace.active_proposition_positions,
            reasoning=workspace.active_reasoning_positions,
            statements=workspace.active_statement_positions,
            procedures=workspace.active_procedure_positions,
        )

    async def list_acquired_blocks(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ListAcquiredBlocksAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        acquired = workspace.acquired_sources[action.source_position]
        source = acquired.source
        blocks: list[SourceBlock] = []
        page_indexes_by_block: dict[str, list[int]] = {}
        for page in acquired.pages:
            for block in page.blocks:
                if block.uuid not in page_indexes_by_block:
                    blocks.append(block)
                    page_indexes_by_block[block.uuid] = []
                append_unique(page_indexes_by_block[block.uuid], page.index)
        page_entries = blocks[
            action.start : action.start + self._catalog_page_size
        ]
        positions: list[int] = []
        images: list[int] = []
        for block in page_entries:
            position = register_block(
                workspace, source, block, page_indexes_by_block[block.uuid]
            )
            positions.append(position)
            for asset in block.assets:
                append_unique(
                    images, register_image(workspace, asset, source, position)
                )
        workspace.discovered_block_positions = positions
        workspace.discovered_image_positions = images
        return _observation(
            'list_acquired_blocks',
            None,
            positions,
            images,
            acquired=[action.source_position],
            catalog_start=action.start,
        )

    async def _register_support_block(
        self,
        user_uuid: str,
        workspace: TeachingEvidenceWorkspace,
        block_uuid: str,
    ) -> list[int]:
        passages = await self._source_evidence.read_passages(
            user_uuid, [block_uuid], expand_memberships=False, before=0, after=0
        )
        register_occurrence_blocks(workspace, passages, inspected=True)
        return [
            position
            for passage in passages
            for position in _receipt_passage(workspace, passage)
        ]


def _observation(
    kind: str,
    query: str | None,
    blocks: list[int],
    images: list[int],
    *,
    groups: list[TeachingContextGroup] | None = None,
    propositions: list[int] | None = None,
    reasoning: list[int] | None = None,
    statements: list[int] | None = None,
    procedures: list[int] | None = None,
    local_outcome_position: int | None = None,
    acquired: list[int] | None = None,
    catalog_start: int | None = None,
) -> TeachingToolObservation:
    return TeachingToolObservation(
        kind=kind,
        query=query,
        block_positions=blocks,
        image_positions=images,
        groups=groups or [],
        proposition_positions=propositions or [],
        reasoning_positions=reasoning or [],
        statement_positions=statements or [],
        procedure_positions=procedures or [],
        local_outcome_position=local_outcome_position,
        acquired_source_positions=acquired or [],
        catalog_start=catalog_start,
    )


def _add_receipt(
    workspace: TeachingEvidenceWorkspace,
    receipt: BlockInspection | ImageInspection | PropositionInspection,
) -> None:
    if receipt.kind == 'block':
        key = (
            receipt.kind,
            receipt.source.uuid,
            receipt.source.revision,
            receipt.block.uuid,
        )
        exists = any(
            item.kind == 'block'
            and (item.source.uuid, item.source.revision, item.block.uuid)
            == key[1:]
            for item in workspace.receipts
        )
    elif receipt.kind == 'image':
        key = (
            receipt.kind,
            receipt.source.uuid,
            receipt.source.revision,
            receipt.asset.uuid,
        )
        exists = any(
            item.kind == 'image'
            and (item.source.uuid, item.source.revision, item.asset.uuid)
            == key[1:]
            for item in workspace.receipts
        )
    else:
        exists = any(
            item.kind == 'proposition'
            and item.record.proposition.uuid == receipt.record.proposition.uuid
            for item in workspace.receipts
        )
    if not exists:
        workspace.receipts.append(receipt)


def _receipt_passage(
    workspace: TeachingEvidenceWorkspace, passage: SourcePassage
) -> list[int]:
    positions: list[int] = []
    for evidence in passage.blocks:
        position = register_block(
            workspace, passage.source, evidence.block, evidence.page_indexes
        )
        positions.append(position)
        _add_receipt(
            workspace,
            BlockInspection(
                source=passage.source,
                block=evidence.block,
                page_indexes=evidence.page_indexes,
            ),
        )
    return positions


def _is_staged(workspace: TeachingEvidenceWorkspace, source: object) -> bool:
    return any(
        item.source.uuid == source.uuid
        and item.source.revision == source.revision
        for item in workspace.staged_sources
    )


async def _read_passages(
    source_evidence: SourceEvidenceService,
    user_uuid: str,
    entries: list[object],
    action: ReadPassagesAction,
    workspace: TeachingEvidenceWorkspace,
) -> list[SourcePassage]:
    grouped: dict[tuple[str, int], list[object]] = {}
    for entry in entries:
        grouped.setdefault(
            (entry.source.uuid, entry.source.revision), []
        ).append(entry)
    passages: list[SourcePassage] = []
    for (_source_uuid, _revision), group in grouped.items():
        source = group[0].source
        uuids = [entry.block.uuid for entry in group]
        if _is_staged(workspace, source):
            snapshot = next(
                item
                for item in workspace.staged_sources
                if item.source.uuid == source.uuid
                and item.source.revision == source.revision
            )
            passages.append(
                staged_source_passages(
                    snapshot, uuids, before=action.before, after=action.after
                )
            )
        else:
            passages.extend(
                await source_evidence.read_passages(
                    user_uuid,
                    uuids,
                    expand_memberships=action.expand_memberships,
                    before=action.before,
                    after=action.after,
                )
            )
    return passages
