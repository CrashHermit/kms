"""Adapt one evidence action to the bounded teaching interaction state."""

from collections.abc import Sequence
from typing import Protocol, cast

from kms.core.model.research.inference import ResearchTask
from kms.core.model.teaching.acquisition import (
    AcquireEvidenceAction,
    ExternalEvidenceRequest,
)
from kms.core.model.teaching.inference import (
    EvidenceToolAction,
    FocusEvidenceAction,
    ReadArgumentsAction,
    ReadImagesAction,
    ReadLearnerObservationsAction,
    ReadPassagesAction,
    ReadProceduresAction,
    ReadPropositionsAction,
    ReadStatementsAction,
    SearchSourcesAction,
    TeachingContext,
    TeachingToolObservation,
)
from kms.core.model.teaching.operation import RespondAction
from kms.core.model.teaching.workflow import (
    EvidenceExecutionRequest,
    TeachingInteractionResult,
)
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.evidence_acquisition.acquisition import EvidenceAcquisitionNode


class TeachingInteractionToolsNode:
    """Execute one action only after checking its references against its prompt."""

    def __init__(self, evidence: EvidenceAcquisitionNode) -> None:
        self._evidence = evidence

    @staticmethod
    def action_error(state: TeachingTurnState) -> str | None:
        """Return an actionable error when the action cites an unoffered handle."""
        action = state.latest_action
        context = state.latest_context
        assert action is not None and context is not None
        if isinstance(action, RespondAction):
            citation_positions = {
                candidate.position for candidate in context.citation_candidates
            }
            for segment in action.segments:
                if invalid := _invalid_positions(
                    segment.cited_evidence_positions, citation_positions
                ):
                    return (
                        'Cite only evidence positions from citation_candidates; '
                        f'unavailable positions: {invalid}.'
                    )
            return None
        return _evidence_action_error(
            cast(EvidenceToolAction, action),
            context,
            state.request.source_uuids,
        )

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Run a valid action or append a rejection observation without dispatch."""
        action = state.latest_action
        assert action is not None
        if error := self.action_error(state):
            observation = TeachingToolObservation(
                kind='invalid_action',
                query=None,
                block_positions=[],
                image_positions=[],
                groups=[],
                error=error,
            )
            workspace = state.workspace.model_copy(deep=True)
            workspace.observations.append(observation)
            return {
                'workspace': workspace,
                'latest_action': None,
                'tool_calls': state.tool_calls + 1,
            }
        if isinstance(action, RespondAction):
            raise AssertionError('Valid response actions are not tool calls')
        if isinstance(action, AcquireEvidenceAction) and isinstance(
            action.request, ExternalEvidenceRequest
        ):
            outcome = state.workspace.local_outcomes[
                action.request.local_outcome_position
            ]
            return {
                'research_task': ResearchTask(
                    question=outcome.request.question,
                    context=outcome.request.context,
                ),
                'research_outcome_position': action.request.local_outcome_position,
                'research_max_tool_calls': self._evidence.max_tool_calls,
                'search_results': [],
                'failures': [],
                'submitted_urls': set(),
                'persisted_source_revisions': [],
                'research_observations': [],
                'research_tool_calls': 0,
                'final_research_result': None,
                'tool_calls': state.tool_calls + 1,
            }
        result = await self._evidence.run(
            EvidenceExecutionRequest(
                user_uuid=state.request.user_uuid,
                source_uuids=state.request.source_uuids,
                action=cast(EvidenceToolAction, action),
                workspace=state.workspace,
            )
        )
        workspace = result.workspace
        workspace.observations.append(result.observation)
        return {
            'workspace': workspace,
            'tool_calls': state.tool_calls + 1,
        }

    @staticmethod
    def finish(state: TeachingTurnState) -> dict[str, object]:
        """Project the completed response and its final visible-position maps."""
        response = state.latest_action
        assert isinstance(response, RespondAction)
        return {
            'interaction': TeachingInteractionResult(
                response=response,
                workspace=state.workspace,
                visible_turn_indexes=state.visible_turn_indexes,
                visible_observation_indexes=state.visible_observation_indexes,
                visible_citation_receipt_indexes=(
                    state.visible_citation_receipt_indexes
                ),
            )
        }


def _evidence_action_error(
    action: EvidenceToolAction,
    context: TeachingContext,
    source_uuids: list[str] | None,
) -> str | None:
    if (
        isinstance(action, SearchSourcesAction)
        and action.source_uuids is not None
    ):
        offered_source_uuids = (
            set(source_uuids)
            if source_uuids is not None
            else {
                revision.source_uuid
                for outcome in context.local_outcomes
                if outcome.research is not None
                for revision in outcome.research.source_revisions
            }
        )
        if invalid := [
            source_uuid
            for source_uuid in dict.fromkeys(action.source_uuids)
            if source_uuid not in offered_source_uuids
        ]:
            return (
                'Use only offered source identifiers; '
                f'unavailable sources: {invalid}.'
            )
        return None
    if isinstance(action, ReadPassagesAction):
        return _check(
            action.block_positions, _offered_blocks(context), 'passage'
        )
    if isinstance(action, ReadImagesAction):
        return _check(action.image_positions, _offered_images(context), 'image')
    if isinstance(action, ReadPropositionsAction):
        return _check(
            action.proposition_positions,
            _offered_propositions(context),
            'proposition',
        )
    if isinstance(action, ReadArgumentsAction):
        return _check(
            action.argument_positions,
            _offered_arguments(context),
            'argument',
        )
    if isinstance(action, ReadLearnerObservationsAction):
        offered = _positions(context.discovery.learner_observations)
        offered.update(
            observation.registry_position
            for observation in context.learner_history
        )
        return _check(
            action.observation_positions,
            offered,
            'learner-observation',
        )
    if isinstance(action, ReadStatementsAction):
        return _check(
            action.statement_positions,
            _offered_statements(context),
            'statement',
        )
    if isinstance(action, ReadProceduresAction):
        return _check(
            action.procedure_positions,
            _offered_procedures(context),
            'procedure',
        )
    if isinstance(action, FocusEvidenceAction):
        for positions, offered, name in (
            (
                action.block_positions,
                _offered_blocks(context),
                'passage',
            ),
            (
                action.image_positions,
                _offered_images(context),
                'image',
            ),
            (
                action.proposition_positions,
                _offered_propositions(context),
                'proposition',
            ),
            (
                action.argument_positions,
                _offered_arguments(context),
                'argument',
            ),
            (
                action.statement_positions,
                _offered_statements(context),
                'statement',
            ),
            (
                action.procedure_positions,
                _offered_procedures(context),
                'procedure',
            ),
        ):
            if error := _check(positions, offered, name):
                return error
    if isinstance(action, AcquireEvidenceAction):
        request = action.request
        if request.kind == 'external':
            if not context.research_enabled:
                return (
                    'External expansion is not available in this interaction.'
                )
            if source_uuids is not None:
                return (
                    'External expansion is unavailable when sources are '
                    'restricted.'
                )
            outcomes = {
                outcome.position
                for outcome in context.local_outcomes
                if not outcome.expanded
            }
            return _check(
                [request.local_outcome_position],
                outcomes,
                'unexpanded local outcome',
            )


def _offered_blocks(context: TeachingContext) -> set[int]:
    observation_groups = [
        group
        for observation in context.observations
        for group in observation.groups
    ]
    return (
        _positions(context.material.blocks)
        | _positions(context.discovery.blocks)
        | {
            position
            for group in [*context.groups, *observation_groups]
            for position in group.member_block_positions
        }
        | {
            position
            for proposition in context.material.propositions
            for position in proposition.supporting_block_positions
        }
        | {
            position
            for argument in context.material.arguments
            for position in argument.supporting_block_positions
        }
        | {
            position
            for image in context.images
            for position in image.block_positions
        }
        | {
            position
            for image in context.discovery.images
            for position in image.block_positions
        }
        | {
            position
            for outcome in context.local_outcomes
            for position in outcome.block_positions
        }
    )


def _offered_images(context: TeachingContext) -> set[int]:
    return (
        _positions(context.images)
        | _positions(context.discovery.images)
        | {
            position
            for block in context.material.blocks
            for position in block.asset_positions
        }
    )


def _offered_propositions(context: TeachingContext) -> set[int]:
    return (
        _positions(context.material.propositions)
        | _positions(context.discovery.propositions)
        | {
            position
            for argument in context.material.arguments
            for position in argument.premise_positions
        }
        | {
            argument.conclusion_position
            for argument in context.material.arguments
            if argument.conclusion_position is not None
        }
        | {
            position
            for outcome in context.local_outcomes
            for position in outcome.proposition_positions
        }
    )


def _offered_arguments(context: TeachingContext) -> set[int]:
    return (
        _positions(context.material.arguments)
        | _positions(context.discovery.arguments)
        | {
            position
            for proposition in context.material.propositions
            for position in proposition.argument_positions
        }
    )


def _offered_statements(context: TeachingContext) -> set[int]:
    return (
        _positions(
            [group for group in context.groups if group.kind == 'statement']
        )
        | _positions(context.discovery.statements)
        | {
            position
            for observation in context.observations
            for group in observation.groups
            for position in group.governed_statement_positions
        }
    )


def _offered_procedures(context: TeachingContext) -> set[int]:
    return _positions(
        [group for group in context.groups if group.kind == 'procedure']
    ) | _positions(context.discovery.procedures)


class _Positioned(Protocol):
    @property
    def position(self) -> int: ...


def _positions(items: Sequence[_Positioned]) -> set[int]:
    return {item.position for item in items}


def _check(requested: list[int], offered: set[int], label: str) -> str | None:
    if invalid := _invalid_positions(requested, offered):
        return (
            f'Use only offered {label} positions; '
            f'unavailable positions: {invalid}.'
        )
    return None


def _invalid_positions(requested: list[int], offered: set[int]) -> list[int]:
    return list(
        dict.fromkeys(
            position for position in requested if position not in offered
        )
    )
