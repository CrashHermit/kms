"""Execute one teaching evidence action through owned graph retrieval."""

from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    EvidenceResearchOutcome,
    LocalEvidenceOutcome,
    LocalEvidenceRequest,
)
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
from kms.core.model.teaching.workflow import (
    EvidenceExecutionRequest,
    EvidenceExecutionResult,
)
from kms.core.model.teaching.workspace import TeachingEvidenceWorkspace
from kms.core.source_workspace import (
    append_unique,
    register_block,
    register_image,
)
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.teaching.evidence import register_proposition
from kms.node.teaching.tools import TeachingEvidenceTools
from kms.service.knowledge_search import KnowledgeSearchService
from kms.service.source_evidence import SourceSearchService


class EvidenceAcquisitionNode:
    """Dispatch one local or external evidence action."""

    def __init__(
        self,
        tools: TeachingEvidenceTools,
        source_search: SourceSearchService,
        knowledge_search: KnowledgeSearchService,
        *,
        max_tool_calls: int,
    ) -> None:
        self._tools = tools
        self._source_search = source_search
        self._knowledge_search = knowledge_search
        self._max_tool_calls = max_tool_calls

    @property
    def max_tool_calls(self) -> int:
        return self._max_tool_calls

    async def run(
        self, request: EvidenceExecutionRequest
    ) -> EvidenceExecutionResult:
        action = request.action
        workspace = request.workspace
        if isinstance(action, SearchSourcesAction):
            search_scope = (
                action.source_uuids
                if action.source_uuids is not None
                else request.source_uuids
            )
            observation = await self._tools.search_sources(
                request.user_uuid, search_scope, action, workspace
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
        elif isinstance(action, SearchArgumentsAction):
            observation = await self._tools.search_arguments(
                request.user_uuid, request.source_uuids, action, workspace
            )

        elif isinstance(action, ReadArgumentsAction):
            observation = await self._tools.read_arguments(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, SearchLearnerObservationsAction):
            observation = await self._tools.search_learner_observations(
                request.user_uuid, request.source_uuids, action, workspace
            )
        elif isinstance(action, ReadLearnerObservationsAction):
            observation = await self._tools.read_learner_observations(
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
        elif isinstance(action, AcquireEvidenceAction):
            if isinstance(action.request, LocalEvidenceRequest):
                (
                    block_positions,
                    image_positions,
                    proposition_positions,
                ) = await self._search_local(
                    request.user_uuid,
                    request.source_uuids,
                    action.request,
                    workspace,
                )
                workspace.local_outcomes.append(
                    LocalEvidenceOutcome(
                        request=action.request,
                        block_positions=block_positions,
                        proposition_positions=proposition_positions,
                    )
                )
                observation = TeachingToolObservation(
                    kind='acquire_local',
                    query=action.request.question,
                    block_positions=block_positions,
                    image_positions=image_positions,
                    groups=[],
                    proposition_positions=proposition_positions,
                    local_outcome_position=len(workspace.local_outcomes) - 1,
                )
            else:
                raise AssertionError(
                    'External evidence must run in the parent graph'
                )
        else:
            raise AssertionError(f'Unhandled evidence action: {action.kind}')
        return EvidenceExecutionResult(
            workspace=workspace, observation=observation
        )

    async def _search_sources(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        query: str,
        workspace: TeachingEvidenceWorkspace,
    ) -> tuple[list[int], list[int]]:
        source_hits = await self._source_search.search_sources(
            user_uuid,
            query,
            source_uuids=source_uuids,
        )
        block_positions = [
            register_block(workspace, hit.source, hit.block, hit.page_indexes)
            for hit in source_hits
        ]
        image_positions: list[int] = []
        for position in block_positions:
            entry = workspace.block_registry[position]
            for asset in entry.block.assets:
                append_unique(
                    image_positions,
                    register_image(workspace, asset, entry.source, position),
                )
        workspace.discovered_block_positions = block_positions
        workspace.discovered_image_positions = image_positions
        return block_positions, image_positions

    async def _search_local(
        self,
        user_uuid: str,
        source_uuids: list[str] | None,
        request: LocalEvidenceRequest,
        workspace: TeachingEvidenceWorkspace,
    ) -> tuple[list[int], list[int], list[int]]:
        block_positions, image_positions = await self._search_sources(
            user_uuid, source_uuids, request.question, workspace
        )
        proposition_hits = await self._knowledge_search.search_propositions(
            user_uuid, request.question
        )
        proposition_positions = [
            register_proposition(workspace, hit.proposition)
            for hit in proposition_hits
        ]
        workspace.discovered_proposition_positions = proposition_positions
        return block_positions, image_positions, proposition_positions

    async def postresearch(self, state: TeachingTurnState) -> dict[str, object]:
        """Attach admitted research sources and retrieve only those sources."""
        research = state.final_research_result
        outcome_position = state.research_outcome_position
        assert research is not None and outcome_position is not None
        workspace = state.workspace.model_copy(deep=True)
        outcome = workspace.local_outcomes[outcome_position]
        outcome.expanded = True
        outcome.research = EvidenceResearchOutcome(
            failures=[failure.message for failure in state.failures],
            source_revisions=research.source_revisions,
        )
        source_uuids = [
            revision.source_uuid for revision in research.source_revisions
        ]
        block_positions, image_positions = await self._search_sources(
            state.request.user_uuid,
            source_uuids,
            outcome.request.question,
            workspace,
        )
        workspace.observations.append(
            TeachingToolObservation(
                kind='acquire_external',
                query=outcome.request.question,
                block_positions=block_positions,
                image_positions=image_positions,
                groups=[],
                local_outcome_position=outcome_position,
                source_revisions=research.source_revisions,
            )
        )
        return {'workspace': workspace, 'research_task': None}
