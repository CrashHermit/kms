"""Build priority-bounded teaching packets and run one teacher prediction."""

from dataclasses import dataclass
from typing import Protocol

import dspy

from kms.core.content import ContentImage, MultimodalContentBuilder
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.material import (
    ArgumentMaterial,
    MaterialContext,
    PropositionMaterial,
    SourceBlockMaterial,
    SourceGroupView,
    TurnMaterial,
)
from kms.core.model.teaching.acquisition import TeachingLocalOutcome
from kms.core.model.teaching.inference import (
    TeachingAction,
    TeachingAssessmentContext,
    TeachingCitationCandidate,
    TeachingContext,
    TeachingContextImage,
    TeachingDiscoveryCatalog,
    TeachingLearnerEvidence,
    TeachingLearnerObservation,
    TeachingLearnerQuestion,
    TeachingPlanDraft,
)
from kms.core.model.teaching.observation import LearnerObservationRecord
from kms.core.model.teaching.operation import TeachingPlanStepDraft
from kms.core.model.teaching.plan import TeachingPlan
from kms.core.model.teaching.question import (
    TeachingQuestion,
    TeachingQuestionContext,
)
from kms.core.model.teaching.session import TeachingSessionSnapshot
from kms.core.model.teaching.workspace import (
    BlockInspection,
    ImageInspection,
    PropositionInspection,
    TeachingEvidenceWorkspace,
)
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.langgraph.teaching.state import TeachingTurnState
from kms.node.teaching.evidence import (
    active_argument_positions,
    active_block_positions,
    active_image_positions,
    active_procedure_positions,
    active_proposition_positions,
    active_statement_positions,
)
from kms.service.content import load_content_images


class TeachingModule(Protocol):
    """Shared predictor API for independently configured teaching phases."""

    async def aforward(
        self,
        *,
        context: TeachingContext,
        content_images: list[dspy.Image],
    ) -> TeachingAction: ...


@dataclass(frozen=True, slots=True)
class TeachingPredictionPacket:
    """Model input and exact parent-position mappings for one prediction."""

    context: TeachingContext
    content_images: list[dspy.Image]
    visible_turn_indexes: list[int]
    visible_observation_indexes: list[int]
    included_block_positions: list[int]
    included_image_positions: list[int]
    citation_references: list[tuple[str, int]]


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

    async def run(self, state: TeachingTurnState) -> dict[str, object]:
        """Choose one typed action from the bounded interaction state."""
        packet = await self._build_context(state)
        action = await self._module.aforward(
            context=packet.context,
            content_images=packet.content_images,
        )
        receipt_positions = [
            self._confirm_inspection(state.workspace, kind, position)
            for kind, position in packet.citation_references
        ]
        return {
            'latest_context': packet.context,
            'latest_action': action,
            'visible_turn_indexes': packet.visible_turn_indexes,
            'visible_observation_indexes': packet.visible_observation_indexes,
            'visible_citation_receipt_indexes': receipt_positions,
        }

    async def _build_context(
        self,
        state: TeachingTurnState,
    ) -> TeachingPredictionPacket:
        """Fit pinned current state, ranked observations, then whole turns."""
        snapshot = state.request.snapshot
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
        context, image_payloads = await self._context(
            state,
            turn_indexes,
            visible_observation_indexes,
            history_truncated=pinned_start > 0,
            learner_history_truncated=False,
        )
        if self._count(context, image_payloads) > self._budget.token_limit:
            raise InputBudgetExceeded(
                'Pinned teaching state and active interaction exceed the input budget'
            )

        for (
            observation_position
        ) in state.workspace.active_learner_observation_positions:
            candidate_indexes = [
                *visible_observation_indexes,
                observation_position,
            ]
            candidate, candidate_images = await self._context(
                state,
                turn_indexes,
                candidate_indexes,
                history_truncated=pinned_start > 0,
                learner_history_truncated=False,
            )
            if (
                self._count(candidate, candidate_images)
                > self._budget.token_limit
            ):
                learner_history_truncated = True
                break
            visible_observation_indexes = candidate_indexes
        if len(visible_observation_indexes) < len(
            state.workspace.active_learner_observation_positions
        ):
            learner_history_truncated = True

        turn_start = pinned_start
        while turn_start > 0:
            candidate_start = turn_start - 1
            candidate_indexes = list(
                range(candidate_start, len(snapshot.turns))
            )
            candidate, candidate_images = await self._context(
                state,
                candidate_indexes,
                visible_observation_indexes,
                history_truncated=candidate_start > 0,
                learner_history_truncated=learner_history_truncated,
            )
            if (
                self._count(candidate, candidate_images)
                > self._budget.token_limit
            ):
                break
            turn_start = candidate_start
            turn_indexes = candidate_indexes

        context, image_payloads = await self._context(
            state,
            turn_indexes,
            visible_observation_indexes,
            history_truncated=turn_start > 0,
            learner_history_truncated=learner_history_truncated,
        )
        if self._count(context, image_payloads) > self._budget.token_limit:
            raise InputBudgetExceeded(
                'Pinned teaching state and active interaction exceed the input budget'
            )
        payloads = await load_content_images(
            image_payloads, budget=self._image_budget
        )
        return TeachingPredictionPacket(
            context=context,
            content_images=[dspy.Image(payload) for payload in payloads],
            visible_turn_indexes=turn_indexes,
            visible_observation_indexes=visible_observation_indexes,
            included_block_positions=[
                block.position for block in context.material.blocks
            ],
            included_image_positions=[
                image.position for image in context.images
            ],
            citation_references=[
                (candidate.kind, candidate.evidence_position)
                for candidate in context.citation_candidates
            ],
        )

    async def _context(
        self,
        state: TeachingTurnState,
        visible_turn_indexes: list[int],
        visible_observation_indexes: list[int],
        *,
        history_truncated: bool,
        learner_history_truncated: bool,
    ) -> tuple[TeachingContext, list[ContentImage]]:
        """Project active evidence with an isolated per-trial image catalog."""

        builder = MultimodalContentBuilder()
        request = state.request
        snapshot = request.snapshot
        workspace = state.workspace
        active_blocks = active_block_positions(workspace)
        active_images = active_image_positions(workspace)
        active_propositions = active_proposition_positions(workspace)
        active_arguments = set(active_argument_positions(workspace))
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
            TurnMaterial(
                position=position,
                role=snapshot.turns[turn_index].turn.role,
                material=builder.build(
                    text=snapshot.turns[turn_index].turn.content
                ),
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
            if image_position not in active_images:
                continue
            for block_position in image.block_positions:
                images_by_block.setdefault(block_position, []).append(
                    image_position
                )
        context_blocks = [
            SourceBlockMaterial(
                position=position,
                source_key=entry.source.key,
                page_indexes=entry.page_indexes,
                block_type=entry.block.block_type.value,
                material=builder.build(
                    text=(
                        entry.block.content
                        if position
                        in (
                            workspace.read_block_positions
                            | workspace.inspected_block_positions
                        )
                        else None
                    ),
                    images=[
                        ContentImage(
                            key=(
                                ('content', image_entry.asset.content_sha256)
                                if image_entry.asset.content_sha256 is not None
                                else ('path', image_entry.asset.path)
                            ),
                            payload=image_entry.data,
                        )
                        for image_position in images_by_block.get(position, [])
                        for image_entry in [
                            workspace.image_registry[image_position]
                        ]
                        if image_entry.data is not None
                    ],
                ),
                asset_positions=images_by_block.get(position, []),
                inspected=True,
            )
            for position, entry in enumerate(workspace.block_registry)
            if position in active_blocks or position in images_by_block
        ]
        context_images = [
            TeachingContextImage(
                position=position,
                block_positions=entry.block_positions,
                inspected=True,
            )
            for position, entry in enumerate(workspace.image_registry)
            if position in active_images
        ]
        context_propositions = [
            PropositionMaterial(
                position=position,
                material=builder.build(text=entry.proposition.content),
                inspected=True,
                supporting_block_positions=self._supporting_block_positions(
                    workspace, entry.uuid
                ),
                argument_positions=[
                    argument_position
                    for argument_position, argument in enumerate(
                        workspace.argument_registry
                    )
                    if argument_position in active_arguments
                    and argument.record is not None
                    and entry.uuid
                    in [
                        *argument.record.premise_uuids,
                        argument.record.conclusion_uuid,
                    ]
                ],
            )
            for position, entry in enumerate(workspace.proposition_registry)
            if position in active_propositions
            and position in workspace.inspected_proposition_positions
            and entry.proposition is not None
        ]
        context_arguments = [
            _argument_context(
                workspace, position, entry.record, set(active_blocks), builder
            )
            for position, entry in enumerate(workspace.argument_registry)
            if position in active_arguments
            and position in workspace.inspected_argument_positions
            and entry.record is not None
        ]
        context_groups = [
            SourceGroupView(
                position=position,
                kind='statement',
                material=builder.build(text=entry.statement.content),
                is_exercise=entry.statement.is_exercise,
                member_block_positions=self._member_block_positions(
                    workspace,
                    entry.source.uuid,
                    entry.source.revision,
                    entry.statement.member_block_uuids,
                ),
                governed_statement_positions=[],
            )
            for position, entry in enumerate(workspace.statement_registry)
            if position in active_statements
        ] + [
            SourceGroupView(
                position=position,
                kind='procedure',
                material=builder.build(text=entry.procedure.content),
                member_block_positions=self._member_block_positions(
                    workspace,
                    entry.source.uuid,
                    entry.source.revision,
                    entry.procedure.member_block_uuids,
                ),
                governed_statement_positions=[],
            )
            for position, entry in enumerate(workspace.procedure_registry)
            if position in active_procedures
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
                require_visual_inspection=(
                    outcome.request.require_visual_inspection
                ),
            )
            for position, outcome in enumerate(workspace.local_outcomes)
        ]
        learner_history = []
        for visible_position, registry_position in enumerate(
            visible_observation_indexes
        ):
            entry = workspace.learner_observation_registry[registry_position]
            assert entry.record is not None
            learner_history.append(
                _learner_observation(
                    entry.record,
                    position=visible_position,
                    registry_position=registry_position,
                )
            )

        context = TeachingContext(
            goal=snapshot.session.goal,
            phase=snapshot.session.phase,
            current_assessment=current_assessment,
            current_plan=current_plan,
            current_plan_accepted=current_plan_accepted,
            current_step_position=current_step_position,
            pending_question=pending_question,
            material=MaterialContext(
                turns=context_turns,
                blocks=context_blocks,
                propositions=context_propositions,
                arguments=context_arguments,
            ),
            learner_history=learner_history,
            learner_history_truncated=learner_history_truncated,
            images=context_images,
            groups=context_groups,
            image_limit=self._image_budget.max_images,
            observations=observations,
            tools_remaining=max(0, self._max_tool_calls - state.tool_calls),
            history_truncated=history_truncated,
            research_enabled=self._research_enabled,
            discovery=self._discovery(workspace),
            local_outcomes=local_outcomes,
            citation_candidates=self._citation_candidates(
                context_blocks, context_images, context_propositions
            ),
        )
        return context, builder.images

    def _count(
        self, context: TeachingContext, images: list[ContentImage]
    ) -> int:
        return self._budget.counter.estimate(
            text=context.model_dump_json(),
            images=images,
        )

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
            arguments=[
                {
                    'position': position,
                    'preview': (
                        workspace.argument_registry[position].preview[:limit]
                        if workspace.argument_registry[position].preview
                        is not None
                        else None
                    ),
                }
                for position in workspace.discovered_argument_positions
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
            learner_observations=[
                {
                    'position': position,
                    'kind': workspace.learner_observation_registry[
                        position
                    ].observation.kind,
                    'topic': workspace.learner_observation_registry[
                        position
                    ].observation.topic,
                    'preview': workspace.learner_observation_registry[
                        position
                    ].observation.description[:limit],
                }
                for position in workspace.discovered_learner_observation_positions
            ],
        )

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

    def _citation_candidates(
        self,
        blocks: list[SourceBlockMaterial],
        images: list[TeachingContextImage],
        propositions: list[PropositionMaterial],
    ) -> list[TeachingCitationCandidate]:
        references = [
            ('block', block.position)
            for block in blocks
            if block.material.text is not None
        ]
        references.extend(('image', image.position) for image in images)
        references.extend(
            ('proposition', proposition.position)
            for proposition in propositions
        )
        return [
            TeachingCitationCandidate(
                position=position,
                kind=kind,
                evidence_position=evidence_position,
            )
            for position, (kind, evidence_position) in enumerate(references)
        ]

    @staticmethod
    def _confirm_inspection(
        workspace: TeachingEvidenceWorkspace, kind: str, position: int
    ) -> int:
        if kind == 'block':
            entry = workspace.block_registry[position]
            workspace.inspected_block_positions.add(position)
            receipt = BlockInspection(
                source=entry.source,
                block=entry.block,
                page_indexes=entry.page_indexes,
            )
        elif kind == 'image':
            entry = workspace.image_registry[position]
            assert entry.data is not None
            workspace.inspected_image_positions.add(position)
            receipt = ImageInspection(
                source=entry.source,
                asset=entry.asset,
                block_uuids=[
                    workspace.block_registry[index].block.uuid
                    for index in entry.block_positions
                ],
                data=entry.data,
            )
        else:
            entry = workspace.proposition_registry[position]
            assert entry.record is not None
            workspace.inspected_proposition_positions.add(position)
            receipt = PropositionInspection(record=entry.record)
        for receipt_position, existing in enumerate(workspace.receipts):
            if _receipt_matches(workspace, kind, position, existing):
                return receipt_position
        workspace.receipts.append(receipt)
        return len(workspace.receipts) - 1


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


def _argument_context(
    workspace: TeachingEvidenceWorkspace,
    position: int,
    record: ArgumentRecord,
    active_blocks: set[int],
    builder: MultimodalContentBuilder,
) -> ArgumentMaterial:
    proposition_positions = {
        entry.uuid: index
        for index, entry in enumerate(workspace.proposition_registry)
    }
    return ArgumentMaterial(
        position=position,
        material=builder.build(text=record.argument.explanation),
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
    record: LearnerObservationRecord,
    *,
    position: int,
    registry_position: int,
) -> TeachingLearnerObservation:
    observation = record.observation
    return TeachingLearnerObservation(
        position=position,
        registry_position=registry_position,
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
