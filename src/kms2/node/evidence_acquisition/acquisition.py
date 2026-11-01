"""Execute one teaching evidence action without interaction bookkeeping."""

from typing import cast

from kms2.core.model.research.inference import (
    ResearchLocalFinding,
    ResearchTask,
)
from kms2.core.model.source_processing.admission import (
    SourceAdmissionResult,
    StagedSourceSnapshot,
)
from kms2.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    EvidenceResearchOutcome,
    ExternalEvidenceRequest,
    LocalEvidenceOutcome,
    LocalEvidenceRequest,
)
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
    TeachingToolObservation,
)
from kms2.core.model.teaching.workflow import (
    EvidenceExecutionResult,
)
from kms2.core.model.teaching.workspace import TeachingEvidenceWorkspace
from kms2.core.source_snapshot import assert_matching_snapshot
from kms2.langgraph.evidence_acquisition.state import EvidenceAcquisitionState
from kms2.langgraph.research.graph import ResearchGraph
from kms2.langgraph.source_admission.graph import SourceAdmissionGraph
from kms2.node.teaching.evidence import (
    active_block_positions,
    active_proposition_positions,
    append_unique,
    register_block,
    register_image,
    register_proposition,
)
from kms2.node.teaching.tools import TeachingEvidenceTools
from kms2.service.knowledge_search import KnowledgeSearchService
from kms2.service.source_evidence import SourceSearchService


class EvidenceAcquisitionNode:
    """Dispatch an action to the workspace-owned evidence tools."""

    def __init__(
        self,
        tools: TeachingEvidenceTools,
        source_search: SourceSearchService,
        knowledge_search: KnowledgeSearchService,
        researcher: ResearchGraph | None,
        source_admission: SourceAdmissionGraph | None,
        *,
        discovery_preview_characters: int,
    ) -> None:
        self._tools = tools
        self._source_search = source_search
        self._knowledge_search = knowledge_search
        self._researcher = researcher
        self._source_admission = source_admission
        self._discovery_preview_characters = discovery_preview_characters

    async def run(self, state: EvidenceAcquisitionState) -> dict[str, object]:
        request = state.request
        action = request.action
        workspace = request.workspace
        if isinstance(action, SearchSourcesAction):
            observation = await self._tools.search_sources(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadPassagesAction):
            observation = await self._tools.read_passages(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadImagesAction):
            observation = await self._tools.read_images(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, SearchPropositionsAction):
            observation = await self._tools.search_propositions(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadPropositionsAction):
            observation = await self._tools.read_propositions(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadReasoningStepsAction):
            observation = await self._tools.read_reasoning_steps(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, SearchStatementsAction):
            observation = await self._tools.search_statements(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadStatementsAction):
            observation = await self._tools.read_statements(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, SearchProceduresAction):
            observation = await self._tools.search_procedures(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadProceduresAction):
            observation = await self._tools.read_procedures(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, FocusEvidenceAction):
            observation = await self._tools.focus_evidence(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ListAcquiredBlocksAction):
            observation = await self._tools.list_acquired_blocks(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, AcquireEvidenceAction):
            if isinstance(action.request, LocalEvidenceRequest):
                source_hits = await self._source_search.search_sources(
                    request.user_uuid,
                    action.request.question,
                    source_uuids=request.source_uuids,
                )
                block_positions = [
                    register_block(
                        workspace, hit.source, hit.block, hit.page_indexes
                    )
                    for hit in source_hits
                ]
                proposition_hits = (
                    await self._knowledge_search.search_propositions(
                        request.user_uuid, action.request.question
                    )
                )
                proposition_positions = [
                    register_proposition(workspace, hit.proposition)
                    for hit in proposition_hits
                ]
                image_positions: list[int] = []
                for position in block_positions:
                    entry = workspace.block_registry[position]
                    for asset in entry.block.assets:
                        append_unique(
                            image_positions,
                            register_image(
                                workspace, asset, entry.source, position
                            ),
                        )
                workspace.discovered_block_positions = block_positions
                workspace.discovered_image_positions = image_positions
                workspace.discovered_proposition_positions = (
                    proposition_positions
                )
                workspace.local_outcomes.append(
                    LocalEvidenceOutcome(
                        request=action.request,
                        block_positions=block_positions,
                        proposition_positions=proposition_positions,
                    )
                )
                outcome_position = len(workspace.local_outcomes) - 1
                observation = TeachingToolObservation(
                    kind='acquire_local',
                    query=action.request.question,
                    block_positions=block_positions,
                    image_positions=image_positions,
                    groups=[],
                    proposition_positions=proposition_positions,
                    local_outcome_position=outcome_position,
                )
            else:
                external = cast(ExternalEvidenceRequest, action.request)
                observation = await self._acquire_external(
                    request.user_uuid,
                    external.local_outcome_position,
                    workspace,
                )
        else:
            raise AssertionError(f'Unhandled evidence action: {action.kind}')
        return {
            'result': EvidenceExecutionResult(
                workspace=workspace, observation=observation
            )
        }

    async def _acquire_external(
        self,
        user_uuid: str,
        local_outcome_position: int,
        workspace: TeachingEvidenceWorkspace,
    ) -> TeachingToolObservation:
        if self._researcher is None or self._source_admission is None:
            raise RuntimeError(
                'External research is disabled for this teaching interaction'
            )

        outcome = workspace.local_outcomes[local_outcome_position]
        local_findings: list[ResearchLocalFinding] = []
        active_blocks = set(active_block_positions(workspace))
        active_propositions = set(active_proposition_positions(workspace))
        for position in outcome.block_positions:
            entry = workspace.block_registry[position]
            inspected = position in active_blocks
            content = entry.block.content or ''
            if not inspected:
                content = content[: self._discovery_preview_characters]
            local_findings.append(
                ResearchLocalFinding(
                    kind='block', text=content, inspected=inspected
                )
            )
        for position in outcome.proposition_positions:
            entry = workspace.proposition_registry[position]
            inspected = position in active_propositions
            content = (
                entry.proposition.content
                if inspected and entry.proposition is not None
                else (
                    entry.proposition.content[
                        : self._discovery_preview_characters
                    ]
                    if entry.proposition is not None
                    else ''
                )
            )
            local_findings.append(
                ResearchLocalFinding(
                    kind='proposition', text=content, inspected=inspected
                )
            )

        result = await self._researcher.run(
            ResearchTask(
                question=outcome.request.question,
                context=outcome.request.context,
                urls=outcome.request.urls,
                local_findings=local_findings,
            )
        )
        source_positions: list[int] = []
        block_positions: list[int] = []
        image_positions: list[int] = []
        for material in result.source_materials:
            acquired = await self._source_admission.run(user_uuid, material)
            if isinstance(acquired, StagedSourceSnapshot):
                existing = next(
                    (
                        candidate
                        for candidate in workspace.staged_sources
                        if candidate.source.uuid == acquired.source.uuid
                    ),
                    None,
                )
                if existing is None:
                    workspace.staged_sources.append(acquired)
                    snapshot: SourceAdmissionResult = acquired
                else:
                    assert_matching_snapshot(acquired, existing)
                    snapshot = existing
            else:
                staged = next(
                    (
                        candidate
                        for candidate in workspace.staged_sources
                        if candidate.source.uuid == acquired.source.uuid
                    ),
                    None,
                )
                if staged is not None:
                    assert_matching_snapshot(staged, acquired)
                    snapshot = staged
                else:
                    snapshot = acquired
            source_position = self._register_acquired_source(
                workspace, snapshot
            )
            append_unique(source_positions, source_position)
            for page in snapshot.pages:
                for block in page.blocks:
                    block_position = register_block(
                        workspace, snapshot.source, block, [page.index]
                    )
                    append_unique(block_positions, block_position)
                    for asset in block.assets:
                        append_unique(
                            image_positions,
                            register_image(
                                workspace,
                                asset,
                                snapshot.source,
                                block_position,
                            ),
                        )

        outcome.expanded = True
        outcome.research = EvidenceResearchOutcome(
            brief=result.brief,
            acquired_source_positions=source_positions,
            failures=[failure.message for failure in result.failures],
        )
        if source_positions:
            first = workspace.acquired_sources[source_positions[0]]
            first_blocks = [
                position
                for position in block_positions
                if workspace.block_registry[position].source.uuid
                == first.source.uuid
                and workspace.block_registry[position].source.revision
                == first.source.revision
            ][: self._tools.catalog_page_size]
            first_block_set = set(first_blocks)
            workspace.discovered_block_positions = first_blocks
            workspace.discovered_image_positions = [
                position
                for position in image_positions
                if first_block_set.intersection(
                    workspace.image_registry[position].block_positions
                )
            ]

        return TeachingToolObservation(
            kind='acquire_external',
            query=outcome.request.question,
            block_positions=workspace.discovered_block_positions
            if source_positions
            else [],
            image_positions=workspace.discovered_image_positions
            if source_positions
            else [],
            groups=[],
            local_outcome_position=local_outcome_position,
            acquired_source_positions=source_positions,
        )

    @staticmethod
    def _register_acquired_source(
        workspace: TeachingEvidenceWorkspace,
        snapshot: SourceAdmissionResult,
    ) -> int:
        for position, existing in enumerate(workspace.acquired_sources):
            if (
                existing.source.uuid == snapshot.source.uuid
                and existing.source.revision == snapshot.source.revision
            ):
                if isinstance(snapshot, StagedSourceSnapshot) and isinstance(
                    existing, StagedSourceSnapshot
                ):
                    assert_matching_snapshot(snapshot, existing)
                return position
        workspace.acquired_sources.append(snapshot)
        return len(workspace.acquired_sources) - 1
