"""Execute one local evidence action against an invocation workspace."""

from kms.core.media_budget import ImagePayloadBudget, check_image_payload
from kms.core.model.material import SourceGroupView
from kms.core.model.source_evidence import SourcePassage
from kms.core.model.teaching.inference import (
    FocusEvidenceAction,
    ReadArgumentsAction,
    ReadImagesAction,
    ReadLearnerObservationsAction,
    ReadPassagesAction,
    ReadProceduresAction,
    ReadPropositionsAction,
    ReadStatementsAction,
    SearchArgumentsAction,
    SearchLearnerObservationsAction,
    SearchProceduresAction,
    SearchPropositionsAction,
    SearchSourcesAction,
    SearchStatementsAction,
    TeachingToolObservation,
)
from kms.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
    TeachingEvidenceWorkspace,
    TeachingProcedureRegistryEntry,
    TeachingStatementRegistryEntry,
)
from kms.core.source_workspace import (
    append_unique,
    register_block,
    register_image,
)
from kms.core.tokenization.budget import InputBudgetExceeded
from kms.node.teaching.evidence import (
    context_group,
    register_argument_refs,
    register_group,
    register_learner_observation,
    register_occurrence_blocks,
    register_proposition,
)
from kms.service.knowledge import KnowledgeService
from kms.service.knowledge_search import KnowledgeSearchService
from kms.service.learner_observation_search import (
    LearnerObservationSearchService,
)
from kms.service.learner_observations import LearnerObservationService
from kms.service.source_evidence import (
    SourceEvidenceService,
    SourceSearchService,
)
from kms.service.source_workspace import SourceWorkspaceService


class TeachingEvidenceTools:
    """Run local evidence actions, mutating only their private workspace."""

    def __init__(
        self,
        source_search: SourceSearchService,
        source_evidence: SourceEvidenceService,
        source_workspace: SourceWorkspaceService,
        knowledge_search: KnowledgeSearchService,
        knowledge_service: KnowledgeService,
        *,
        learner_observation_search: LearnerObservationSearchService,
        learner_observations: LearnerObservationService,
        image_budget: ImagePayloadBudget,
        discovery_preview_characters: int,
        catalog_page_size: int,
    ) -> None:
        self._source_search = source_search
        self._source_evidence = source_evidence
        self._source_workspace = source_workspace
        self._knowledge_search = knowledge_search
        self._knowledge_service = knowledge_service
        self._learner_observation_search = learner_observation_search
        self._learner_observations = learner_observations
        self._image_budget = image_budget
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
        read = await self._source_workspace.read_blocks(
            user_uuid,
            workspace,
            action.block_positions,
            expand_memberships=action.expand_memberships,
            before=action.before,
            after=action.after,
            load_images=False,
        )
        passages = read.passages
        block_positions = read.block_positions
        image_positions = read.image_positions
        workspace.read_block_positions.update(block_positions)
        for position in image_positions:
            append_unique(workspace.discovered_image_positions, position)
        workspace.active_block_positions = block_positions.copy()
        workspace.active_image_positions = []
        groups: list[SourceGroupView] = []
        for passage in passages:
            local_blocks = {
                evidence.block.uuid: next(
                    position
                    for position in block_positions
                    if workspace.block_registry[position].block.uuid
                    == evidence.block.uuid
                    and workspace.block_registry[position].source.uuid
                    == passage.source.uuid
                    and workspace.block_registry[position].source.revision
                    == passage.source.revision
                )
                for evidence in passage.blocks
            }
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
        positions = list(dict.fromkeys(action.image_positions))
        blocks: list[int] = []
        for position in positions:
            for block_position in workspace.image_registry[
                position
            ].block_positions:
                append_unique(blocks, block_position)
        if len(positions) > self._image_budget.max_images:
            return _observation(
                'read_images',
                None,
                blocks,
                [],
                error=(
                    'Selected images exceed the configured payload budget; '
                    'select fewer images.'
                ),
            )
        previous_data = {
            position: workspace.image_registry[position].data
            for position in positions
        }
        await self._source_workspace.read_images(
            user_uuid, workspace, positions
        )
        try:
            check_image_payload(
                [
                    workspace.image_registry[position].data
                    for position in positions
                ],
                self._image_budget,
            )
        except InputBudgetExceeded as error:
            for position, data in previous_data.items():
                workspace.image_registry[position].data = data
            return _observation(
                'read_images',
                None,
                blocks,
                [],
                error=(f'{error}; select fewer or smaller images.'),
            )
        workspace.read_image_positions.update(positions)
        workspace.active_image_positions = positions.copy()
        return _observation('read_images', None, blocks, positions)

    def _materialize_passages(
        self,
        passages: list[SourcePassage],
        workspace: TeachingEvidenceWorkspace,
    ) -> tuple[list[int], list[int]]:
        block_positions: list[int] = []
        image_positions: list[int] = []
        for passage in passages:
            for evidence in passage.blocks:
                block_position = register_block(
                    workspace,
                    passage.source,
                    evidence.block,
                    evidence.page_indexes,
                )
                append_unique(block_positions, block_position)
                for asset in evidence.block.assets:
                    append_unique(
                        image_positions,
                        register_image(
                            workspace, asset, passage.source, block_position
                        ),
                    )
        workspace.read_block_positions.update(block_positions)
        for position in image_positions:
            append_unique(workspace.discovered_image_positions, position)
        workspace.active_block_positions = block_positions.copy()
        workspace.active_image_positions = []
        return block_positions, image_positions

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

    async def search_arguments(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: SearchArgumentsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        hits = await self._knowledge_search.search_arguments(
            user_uuid, action.query
        )
        register_argument_refs(
            workspace, [hit.record.argument.uuid for hit in hits]
        )
        for position, hit in zip(
            workspace.discovered_argument_positions, hits, strict=True
        ):
            workspace.argument_registry[position].preview = hit.content
        return _observation(
            'search_arguments',
            action.query,
            [],
            [],
            argument=list(workspace.discovered_argument_positions),
        )

    async def search_learner_observations(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: SearchLearnerObservationsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        hits = await self._learner_observation_search.search(
            user_uuid, action.query
        )
        positions = []
        for hit in hits:
            position = register_learner_observation(
                workspace, hit.record.observation
            )
            append_unique(positions, position)
        workspace.discovered_learner_observation_positions = positions
        return _observation(
            'search_learner_observations',
            action.query,
            [],
            [],
            learner_observations=positions,
        )

    async def read_learner_observations(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadLearnerObservationsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.learner_observation_registry[position]
            for position in action.observation_positions
        ]
        records = (
            await self._learner_observations.read(
                user_uuid,
                [entry.observation.uuid for entry in entries],
            )
            if entries
            else []
        )
        for position, record in zip(
            action.observation_positions, records, strict=True
        ):
            entry = workspace.learner_observation_registry[position]
            entry.observation = record.observation
            entry.record = record
        workspace.active_learner_observation_positions = list(
            action.observation_positions
        )
        return _observation(
            'read_learner_observations',
            None,
            [],
            [],
            learner_observations=list(action.observation_positions),
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
        register_argument_refs(
            workspace,
            [uuid for record in records for uuid in record.argument_uuids],
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
            argument=list(workspace.discovered_argument_positions),
        )

    async def read_arguments(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: ReadArgumentsAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        entries = [
            workspace.argument_registry[position]
            for position in action.argument_positions
        ]
        records = await self._knowledge_service.read_arguments(
            user_uuid, [entry.uuid for entry in entries]
        )
        proposition_uuids: list[str] = []
        support_blocks: list[int] = []
        for position, record in zip(
            action.argument_positions, records, strict=True
        ):
            workspace.argument_registry[position].record = record
            workspace.inspected_argument_positions.add(position)
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
        register_argument_refs(
            workspace,
            [
                uuid
                for record in proposition_records
                for uuid in record.argument_uuids
            ],
        )
        workspace.active_argument_positions = list(action.argument_positions)
        workspace.active_proposition_positions = proposition_positions
        return _observation(
            'read_arguments',
            None,
            support_blocks,
            [],
            propositions=proposition_positions,
            argument=list(action.argument_positions),
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
        passages = [
            passage for record in records for passage in record.passages
        ]
        blocks, images = self._materialize_passages(passages, workspace)
        for position, record in zip(
            action.statement_positions, records, strict=True
        ):
            workspace.statement_registry[position].passage = record
        workspace.active_statement_positions = list(action.statement_positions)
        return _observation(
            'read_statements',
            None,
            blocks,
            images,
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
        passages = [
            passage for record in records for passage in record.passages
        ]
        blocks, images = self._materialize_passages(passages, workspace)
        for position, record in zip(
            action.procedure_positions, records, strict=True
        ):
            workspace.procedure_registry[position].passage = record
        workspace.active_procedure_positions = list(action.procedure_positions)
        return _observation(
            'read_procedures',
            None,
            blocks,
            images,
            procedures=list(action.procedure_positions),
        )

    async def focus_evidence(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        action: FocusEvidenceAction,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        readable_images = (
            workspace.read_image_positions | workspace.inspected_image_positions
        )
        images = [
            position
            for position in dict.fromkeys(action.image_positions)
            if position in readable_images
        ]
        try:
            check_image_payload(
                [
                    workspace.image_registry[position].data
                    for position in images
                ],
                self._image_budget,
            )
        except InputBudgetExceeded as error:
            return _observation(
                'focus_evidence',
                None,
                workspace.active_block_positions,
                workspace.active_image_positions,
                propositions=workspace.active_proposition_positions,
                argument=workspace.active_argument_positions,
                statements=workspace.active_statement_positions,
                procedures=workspace.active_procedure_positions,
                error=f'{error}; select fewer or smaller images.',
            )
        workspace.active_block_positions = list(action.block_positions)
        workspace.active_image_positions = images
        workspace.active_proposition_positions = list(
            action.proposition_positions
        )
        workspace.active_argument_positions = list(action.argument_positions)
        workspace.active_statement_positions = list(action.statement_positions)
        workspace.active_procedure_positions = list(action.procedure_positions)
        return _observation(
            'focus_evidence',
            None,
            workspace.active_block_positions,
            workspace.active_image_positions,
            propositions=workspace.active_proposition_positions,
            argument=workspace.active_argument_positions,
            statements=workspace.active_statement_positions,
            procedures=workspace.active_procedure_positions,
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
        register_occurrence_blocks(workspace, passages)
        return [
            position
            for passage in passages
            for evidence in passage.blocks
            for position, entry in enumerate(workspace.block_registry)
            if entry.source.uuid == passage.source.uuid
            and entry.source.revision == passage.source.revision
            and entry.block.uuid == evidence.block.uuid
        ]


def _observation(
    kind: str,
    query: str | None,
    blocks: list[int],
    images: list[int],
    *,
    groups: list[SourceGroupView] | None = None,
    propositions: list[int] | None = None,
    argument: list[int] | None = None,
    statements: list[int] | None = None,
    procedures: list[int] | None = None,
    error: str | None = None,
    local_outcome_position: int | None = None,
    learner_observations: list[int] | None = None,
) -> TeachingToolObservation:
    return TeachingToolObservation(
        kind=kind,
        query=query,
        block_positions=blocks,
        image_positions=images,
        groups=groups or [],
        proposition_positions=propositions or [],
        argument_positions=argument or [],
        statement_positions=statements or [],
        procedure_positions=procedures or [],
        learner_observation_positions=learner_observations or [],
        error=error,
        local_outcome_position=local_outcome_position,
    )


def _add_receipt(
    workspace: TeachingEvidenceWorkspace,
    receipt: BlockInspection | ImageInspection | PropositionInspection,
) -> None:
    if receipt.kind == 'image':
        existing = next(
            (
                item
                for item in workspace.receipts
                if item.kind == 'image'
                and (item.source.uuid, item.source.revision, item.asset.uuid)
                == (
                    receipt.source.uuid,
                    receipt.source.revision,
                    receipt.asset.uuid,
                )
            ),
            None,
        )
        if existing is None:
            workspace.receipts.append(receipt)
        else:
            for block_uuid in receipt.block_uuids:
                append_unique(existing.block_uuids, block_uuid)
        return
    if receipt.kind == 'block':
        exists = any(
            item.kind == 'block'
            and (item.source.uuid, item.source.revision, item.block.uuid)
            == (
                receipt.source.uuid,
                receipt.source.revision,
                receipt.block.uuid,
            )
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
