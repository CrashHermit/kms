"""Build priority-bounded teaching packets and run one teacher prediction."""

from dataclasses import dataclass
from typing import Protocol

import dspy

from kms2.core.media_budget import ImagePayloadBudget, check_image_payload
from kms2.core.model.knowledge.reasoning import ReasoningStepRecord
from kms2.core.model.teaching.acquisition import TeachingLocalOutcome
from kms2.core.model.teaching.inference import (
    TeachingAcquiredSource,
    TeachingAction,
    TeachingAssessmentContext,
    TeachingCitationCandidate,
    TeachingContext,
    TeachingContextBlock,
    TeachingContextImage,
    TeachingContextProcedure,
    TeachingContextProposition,
    TeachingContextReasoningStep,
    TeachingContextStatement,
    TeachingContextTurn,
    TeachingDiscoveryCatalog,
    TeachingLearnerEvidence,
    TeachingLearnerObservation,
    TeachingLearnerQuestion,
    TeachingPlanDraft,
    TeachingQuestionContext,
)
from kms2.core.model.teaching.observation import LearnerObservationRecord
from kms2.core.model.teaching.operation import TeachingPlanStepDraft
from kms2.core.model.teaching.plan import TeachingPlan
from kms2.core.model.teaching.question import TeachingQuestion
from kms2.core.model.teaching.session import TeachingSessionSnapshot
from kms2.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
    TeachingEvidenceWorkspace,
)
from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.teaching.interaction_state import TeachingInteractionState
from kms2.node.teaching.evidence import (
    active_block_positions,
    active_image_positions,
    active_procedure_positions,
    active_proposition_positions,
    active_reasoning_positions,
    active_statement_positions,
)


class TeachingModule(Protocol):
    """Shared predictor API for independently configured teaching phases."""

    async def aforward(
        self,
        *,
        context: TeachingContext,
        evidence_images: list[dspy.Image],
    ) -> TeachingAction: ...


@dataclass(frozen=True, slots=True)
class TeachingPredictionPacket:
    """Model input and exact parent-position mappings for one prediction."""

    context: TeachingContext
    evidence_images: list[dspy.Image]
    visible_turn_indexes: list[int]
    visible_observation_indexes: list[int]
    visible_citation_receipt_indexes: list[int]


class TeachingTeacherNode:
    """Run one prediction over active evidence and ranked learner history."""

    def __init__(
        self,
        module: TeachingModule,
        budget: TokenBudget,
        max_tool_calls: int,
        *,
        image_budget: ImagePayloadBudget,
        discovery_preview_characters: int,
        research_enabled: bool,
    ) -> None:
        self._module = module
        self._budget = budget
        self._max_tool_calls = max_tool_calls
        self._image_budget = image_budget
        self._discovery_preview_characters = discovery_preview_characters
        self._research_enabled = research_enabled

    async def run(self, state: TeachingInteractionState) -> dict[str, object]:
        """Choose one typed action from the bounded interaction state."""
        packet = self._build_context(state)
        action = await self._module.aforward(
            context=packet.context,
            evidence_images=packet.evidence_images,
        )
        return {
            'latest_action': action,
            'visible_turn_indexes': packet.visible_turn_indexes,
            'visible_observation_indexes': packet.visible_observation_indexes,
            'visible_citation_receipt_indexes': (
                packet.visible_citation_receipt_indexes
            ),
        }

    def _build_context(
        self,
        state: TeachingInteractionState,
    ) -> TeachingPredictionPacket:
        """Fit pinned current state, ranked observations, then whole turns."""
        snapshot = state.request.turn.snapshot
        workspace = state.workspace
        pending_question_uuid = snapshot.pending_question_uuid
        if pending_question_uuid is None:
            pinned_start = max(0, len(snapshot.turns) - 1)
        else:
            pinned_start = next(
                index
                for index, record in enumerate(snapshot.turns)
                if record.question is not None
                and record.question.uuid == pending_question_uuid
            )
        turn_indexes = list(range(pinned_start, len(snapshot.turns)))
        visible_observation_indexes: list[int] = []
        learner_history_truncated = False
        context = self._context(
            state,
            turn_indexes,
            visible_observation_indexes,
            history_truncated=pinned_start > 0,
            learner_history_truncated=False,
        )
        if self._count(context) > self._budget.token_limit:
            raise InputBudgetExceeded(
                'Pinned teaching state and active interaction exceed the input budget'
            )

        for observation_index in range(len(state.request.learner_history)):
            candidate_indexes = [
                *visible_observation_indexes,
                observation_index,
            ]
            candidate = self._context(
                state,
                turn_indexes,
                candidate_indexes,
                history_truncated=pinned_start > 0,
                learner_history_truncated=False,
            )
            if self._count(candidate) > self._budget.token_limit:
                learner_history_truncated = True
                break
            visible_observation_indexes = candidate_indexes
            context = candidate
        if len(visible_observation_indexes) < len(
            state.request.learner_history
        ):
            learner_history_truncated = True

        turn_start = pinned_start
        while turn_start > 0:
            candidate_start = turn_start - 1
            candidate_indexes = list(
                range(candidate_start, len(snapshot.turns))
            )
            candidate = self._context(
                state,
                candidate_indexes,
                visible_observation_indexes,
                history_truncated=candidate_start > 0,
                learner_history_truncated=learner_history_truncated,
            )
            if self._count(candidate) > self._budget.token_limit:
                break
            turn_start = candidate_start
            turn_indexes = candidate_indexes
            context = candidate

        context = self._context(
            state,
            turn_indexes,
            visible_observation_indexes,
            history_truncated=turn_start > 0,
            learner_history_truncated=learner_history_truncated,
        )
        image_payloads = []
        for position in active_image_positions(workspace):
            payload = workspace.image_registry[position].data
            assert payload is not None
            image_payloads.append(payload)
        check_image_payload(image_payloads, self._image_budget)
        evidence_images = [dspy.Image(payload) for payload in image_payloads]
        return TeachingPredictionPacket(
            context=context,
            evidence_images=evidence_images,
            visible_turn_indexes=turn_indexes,
            visible_observation_indexes=visible_observation_indexes,
            visible_citation_receipt_indexes=self._citation_receipts(workspace),
        )

    def _context(
        self,
        state: TeachingInteractionState,
        visible_turn_indexes: list[int],
        visible_observation_indexes: list[int],
        *,
        history_truncated: bool,
        learner_history_truncated: bool,
    ) -> TeachingContext:
        """Project only active inspected evidence and bounded discoveries."""
        request = state.request.turn
        snapshot = request.snapshot
        workspace = state.workspace
        active_blocks = active_block_positions(workspace)
        active_images = active_image_positions(workspace)
        active_propositions = active_proposition_positions(workspace)
        active_reasoning = set(active_reasoning_positions(workspace))
        active_statements = set(active_statement_positions(workspace))
        active_procedures = set(active_procedure_positions(workspace))

        current_plan_positions = (
            {
                step.uuid: position
                for position, step in enumerate(snapshot.current_plan.steps)
            }
            if snapshot.current_plan is not None
            else {}
        )
        current_step_position = (
            current_plan_positions.get(snapshot.current_step_uuid)
            if snapshot.current_step_uuid is not None
            else None
        )
        current_assessment = (
            TeachingAssessmentContext(
                summary=snapshot.current_assessment.summary,
                strands=snapshot.current_assessment.strands,
            )
            if snapshot.current_assessment is not None
            else None
        )
        current_plan = _plan_draft(snapshot.current_plan)
        pending_question = _pending_question_context(
            snapshot, current_plan_positions
        )
        current_plan_accepted = (
            snapshot.current_plan is not None
            and snapshot.current_plan_acceptance is not None
            and snapshot.current_plan_acceptance.plan_uuid
            == snapshot.current_plan.uuid
        )

        context_turns = [
            TeachingContextTurn(
                position=position,
                role=snapshot.turns[turn_index].turn.role,
                content=snapshot.turns[turn_index].turn.content,
                question=(
                    _question_context(
                        snapshot.turns[turn_index].question,
                        snapshot.turns[turn_index].question_step_uuid,
                        current_plan_positions,
                    )
                    if snapshot.turns[turn_index].question is not None
                    else None
                ),
                responds_to_pending_question=(
                    snapshot.pending_question_uuid is not None
                    and snapshot.turns[turn_index].responds_to_question_uuid
                    == snapshot.pending_question_uuid
                ),
                presents_current_plan=(
                    snapshot.current_plan is not None
                    and snapshot.turns[turn_index].presented_plan_uuid
                    == snapshot.current_plan.uuid
                ),
                accepts_current_plan=(
                    snapshot.current_plan is not None
                    and snapshot.turns[turn_index].accepted_plan_uuid
                    == snapshot.current_plan.uuid
                ),
            )
            for position, turn_index in enumerate(visible_turn_indexes)
        ]
        images_by_block: dict[int, list[int]] = {}
        for image_position, image in enumerate(workspace.image_registry):
            for block_position in image.block_positions:
                images_by_block.setdefault(block_position, []).append(
                    image_position
                )
        context_blocks = [
            TeachingContextBlock(
                position=position,
                source_key=entry.source.key,
                page_indexes=entry.page_indexes,
                block_type=entry.block.block_type.value,
                content=entry.block.content,
                image_positions=images_by_block.get(position, []),
                inspected=True,
            )
            for position, entry in enumerate(workspace.block_registry)
            if position in active_blocks
            and position in workspace.inspected_block_positions
        ]
        native_image_positions = {
            registry_position: native_position
            for native_position, registry_position in enumerate(active_images)
        }
        context_images = [
            TeachingContextImage(
                position=position,
                block_positions=entry.block_positions,
                inspected=True,
                native_image_position=native_image_positions[position],
            )
            for position, entry in enumerate(workspace.image_registry)
            if position in active_images
            and position in workspace.inspected_image_positions
        ]
        context_propositions = [
            TeachingContextProposition(
                position=position,
                content=entry.proposition.content,
                inspected=True,
                supporting_block_positions=self._supporting_block_positions(
                    workspace, entry.uuid
                ),
                reasoning_positions=[
                    reasoning_position
                    for reasoning_position, reasoning in enumerate(
                        workspace.reasoning_registry
                    )
                    if reasoning_position in active_reasoning
                    and reasoning.record is not None
                    and entry.uuid
                    in [
                        *reasoning.record.premise_uuids,
                        reasoning.record.conclusion_uuid,
                    ]
                ],
            )
            for position, entry in enumerate(workspace.proposition_registry)
            if position in active_propositions
            and position in workspace.inspected_proposition_positions
            and entry.proposition is not None
        ]
        context_reasoning = [
            _reasoning_context(workspace, position, entry.record, active_blocks)
            for position, entry in enumerate(workspace.reasoning_registry)
            if position in active_reasoning
            and position in workspace.inspected_reasoning_positions
            and entry.record is not None
        ]
        context_statements = [
            TeachingContextStatement(
                position=position,
                content=entry.statement.content,
                is_exercise=entry.statement.is_exercise,
                block_positions=self._member_block_positions(
                    workspace,
                    entry.source.uuid,
                    entry.source.revision,
                    entry.statement.member_block_uuids,
                ),
                inspected=True,
            )
            for position, entry in enumerate(workspace.statement_registry)
            if position in active_statements
            and position in workspace.inspected_statement_positions
        ]
        context_procedures = [
            TeachingContextProcedure(
                position=position,
                content=entry.procedure.content,
                block_positions=self._member_block_positions(
                    workspace,
                    entry.source.uuid,
                    entry.source.revision,
                    entry.procedure.member_block_uuids,
                ),
                inspected=True,
            )
            for position, entry in enumerate(workspace.procedure_registry)
            if position in active_procedures
            and position in workspace.inspected_procedure_positions
        ]
        observations = workspace.observations[-1:]
        local_outcomes = [
            TeachingLocalOutcome(
                position=position,
                question=outcome.request.question,
                block_positions=outcome.block_positions,
                proposition_positions=outcome.proposition_positions,
                expanded=outcome.expanded,
                research=outcome.research,
            )
            for position, outcome in enumerate(workspace.local_outcomes)
        ]
        return TeachingContext(
            goal=snapshot.session.goal,
            phase=snapshot.session.phase,
            current_assessment=current_assessment,
            current_plan=current_plan,
            current_plan_accepted=current_plan_accepted,
            current_step_position=current_step_position,
            pending_question=pending_question,
            turns=context_turns,
            blocks=context_blocks,
            learner_history=[
                _learner_observation(observation, position=visible_position)
                for visible_position, observation in enumerate(
                    state.request.learner_history
                )
                if visible_position in visible_observation_indexes
            ],
            learner_history_truncated=learner_history_truncated,
            images=context_images,
            propositions=context_propositions,
            image_limit=self._image_budget.max_images,
            reasoning_steps=context_reasoning,
            statements=context_statements,
            procedures=context_procedures,
            observations=observations,
            tools_remaining=max(0, self._max_tool_calls - state.tool_calls),
            history_truncated=history_truncated,
            research_enabled=self._research_enabled,
            discovery=self._discovery(workspace),
            acquired_sources=self._acquired_sources(workspace),
            citation_candidates=self._citation_candidates(workspace),
            local_outcomes=local_outcomes,
        )

    def _count(self, context: TeachingContext) -> int:
        return self._budget.counter.count_texts([context.model_dump_json()])[0]

    def _discovery(
        self, workspace: TeachingEvidenceWorkspace
    ) -> TeachingDiscoveryCatalog:
        limit = self._discovery_preview_characters
        return TeachingDiscoveryCatalog(
            blocks=[
                _block_discovery(workspace, position, limit)
                for position in workspace.discovered_block_positions
            ],
            images=[
                {
                    'position': position,
                    'block_positions': workspace.image_registry[
                        position
                    ].block_positions,
                }
                for position in workspace.discovered_image_positions
            ],
            propositions=[
                {
                    'position': position,
                    'preview': workspace.proposition_registry[
                        position
                    ].proposition.content[:limit],
                }
                for position in workspace.discovered_proposition_positions
                if workspace.proposition_registry[position].proposition
                is not None
            ],
            reasoning_steps=[
                {'position': position}
                for position in workspace.discovered_reasoning_positions
            ],
            statements=[
                {
                    'position': position,
                    'source_key': workspace.statement_registry[
                        position
                    ].source.key,
                    'is_exercise': workspace.statement_registry[
                        position
                    ].statement.is_exercise,
                    'preview': workspace.statement_registry[
                        position
                    ].statement.content[:limit],
                }
                for position in workspace.discovered_statement_positions
            ],
            procedures=[
                {
                    'position': position,
                    'source_key': workspace.procedure_registry[
                        position
                    ].source.key,
                    'preview': workspace.procedure_registry[
                        position
                    ].procedure.content[:limit],
                }
                for position in workspace.discovered_procedure_positions
            ],
        )

    @staticmethod
    def _acquired_sources(
        workspace: TeachingEvidenceWorkspace,
    ) -> list[TeachingAcquiredSource]:
        return [
            TeachingAcquiredSource(
                position=position,
                source_key=source.source.key,
                page_count=len(source.pages),
                block_count=sum(len(page.blocks) for page in source.pages),
                image_count=sum(
                    len(block.assets)
                    for page in source.pages
                    for block in page.blocks
                ),
            )
            for position, source in enumerate(workspace.acquired_sources)
        ]

    @staticmethod
    def _member_block_positions(
        workspace: TeachingEvidenceWorkspace,
        source_uuid: str,
        revision: int,
        block_uuids: list[str],
    ) -> list[int]:
        return [
            position
            for position, entry in enumerate(workspace.block_registry)
            if entry.source.uuid == source_uuid
            and entry.source.revision == revision
            and entry.block.uuid in block_uuids
        ]

    @staticmethod
    def _supporting_block_positions(
        workspace: TeachingEvidenceWorkspace, proposition_uuid: str
    ) -> list[int]:
        entry = next(
            item
            for item in workspace.proposition_registry
            if item.uuid == proposition_uuid
        )
        if entry.record is None:
            return []
        return [
            position
            for assertion in entry.record.assertions
            for block_uuid in assertion.supporting_block_uuids
            for position, block in enumerate(workspace.block_registry)
            if block.block.uuid == block_uuid
        ]

    @staticmethod
    def _citation_candidates(
        workspace: TeachingEvidenceWorkspace,
    ) -> list[TeachingCitationCandidate]:
        return [
            TeachingCitationCandidate(
                position=position,
                kind=kind,
                evidence_position=evidence_position,
            )
            for position, (
                kind,
                evidence_position,
                _receipt_position,
            ) in enumerate(_citation_entries(workspace))
        ]

    @staticmethod
    def _citation_receipts(
        workspace: TeachingEvidenceWorkspace,
    ) -> list[int]:
        return [
            receipt_position
            for _, _, receipt_position in _citation_entries(workspace)
        ]


def _citation_entries(
    workspace: TeachingEvidenceWorkspace,
) -> list[tuple[str, int, int]]:
    blocks = active_block_positions(workspace)
    images = active_image_positions(workspace)
    propositions = active_proposition_positions(workspace)
    entries: list[tuple[str, int, int]] = []
    for kind, active_positions in (
        ('block', blocks),
        ('image', images),
        ('proposition', propositions),
    ):
        for evidence_position in active_positions:
            for receipt_position, receipt in enumerate(workspace.receipts):
                if _receipt_matches(
                    workspace, kind, evidence_position, receipt
                ):
                    entries.append((kind, evidence_position, receipt_position))
                    break
    return entries


def _receipt_matches(
    workspace: TeachingEvidenceWorkspace,
    kind: str,
    position: int,
    receipt: object,
) -> bool:
    if kind == 'block' and isinstance(receipt, BlockInspection):
        entry = workspace.block_registry[position]
        return (
            receipt.source.uuid == entry.source.uuid
            and receipt.source.revision == entry.source.revision
            and receipt.block.uuid == entry.block.uuid
        )
    if kind == 'image' and isinstance(receipt, ImageInspection):
        entry = workspace.image_registry[position]
        return (
            receipt.source.uuid == entry.source.uuid
            and receipt.source.revision == entry.source.revision
            and receipt.asset.uuid == entry.asset.uuid
        )
    return (
        kind == 'proposition'
        and isinstance(receipt, PropositionInspection)
        and receipt.record.proposition.uuid
        == workspace.proposition_registry[position].uuid
    )


def _block_discovery(
    workspace: TeachingEvidenceWorkspace, position: int, limit: int
) -> dict[str, object]:
    entry = workspace.block_registry[position]
    content = entry.block.content
    return {
        'position': position,
        'source_key': entry.source.key,
        'page_indexes': entry.page_indexes,
        'block_type': entry.block.block_type.value,
        'preview': content[:limit] if content is not None else None,
    }


def _reasoning_context(
    workspace: TeachingEvidenceWorkspace,
    position: int,
    record: ReasoningStepRecord,
    active_blocks: set[int],
) -> TeachingContextReasoningStep:
    proposition_positions = {
        entry.uuid: index
        for index, entry in enumerate(workspace.proposition_registry)
    }
    return TeachingContextReasoningStep(
        position=position,
        explanation=record.step.explanation,
        premise_positions=[
            proposition_positions[uuid] for uuid in record.premise_uuids
        ],
        conclusion_position=proposition_positions[record.conclusion_uuid],
        supporting_block_positions=[
            block_position
            for block_position in sorted(active_blocks)
            if workspace.block_registry[block_position].block.uuid
            in record.supporting_block_uuids
        ],
        inspected=True,
    )


def _learner_observation(
    record: LearnerObservationRecord, *, position: int
) -> TeachingLearnerObservation:
    observation = record.observation
    return TeachingLearnerObservation(
        position=position,
        kind=observation.kind,
        topic=observation.topic,
        description=observation.description,
        created_at=observation.created_at,
        evidence=[
            TeachingLearnerEvidence(
                content=evidence.content,
                created_at=evidence.created_at,
                question=(
                    TeachingLearnerQuestion(
                        kind=evidence.question.kind,
                        prompt=evidence.question.prompt,
                        response=evidence.question.response,
                    )
                    if evidence.question is not None
                    else None
                ),
            )
            for evidence in record.evidence
        ],
    )


def _plan_draft(plan: TeachingPlan | None) -> TeachingPlanDraft | None:
    """Project a current plan with prerequisite step positions."""
    if plan is None:
        return None
    step_positions = {
        step.uuid: position for position, step in enumerate(plan.steps)
    }
    prerequisites: dict[str, list[int]] = {step.uuid: [] for step in plan.steps}
    for dependency in plan.dependencies:
        prerequisites[dependency.step_uuid].append(
            step_positions[dependency.prerequisite_uuid]
        )
    return TeachingPlanDraft(
        summary=plan.summary,
        steps=[
            TeachingPlanStepDraft(
                title=step.title,
                objective=step.objective,
                prerequisite_positions=prerequisites[step.uuid],
            )
            for step in plan.steps
        ],
    )


def _pending_question_context(
    snapshot: TeachingSessionSnapshot,
    plan_positions: dict[str, int],
) -> TeachingQuestionContext | None:
    """Project the active private question and backend-derived step link."""
    if snapshot.pending_question_uuid is None:
        return None
    question_record = next(
        record
        for record in snapshot.turns
        if record.question is not None
        and record.question.uuid == snapshot.pending_question_uuid
    )
    return _question_context(
        question_record.question,
        question_record.question_step_uuid,
        plan_positions,
    )


def _question_context(
    question: TeachingQuestion,
    step_uuid: str | None,
    plan_positions: dict[str, int],
) -> TeachingQuestionContext:
    """Project a private question with its selected plan-step position."""
    return TeachingQuestionContext(
        kind=question.kind,
        prompt=question.prompt,
        response=question.response,
        step_position=(
            plan_positions.get(step_uuid) if step_uuid is not None else None
        ),
    )
