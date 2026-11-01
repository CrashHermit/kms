"""Backend record assignment and bounded knowledge read operations."""

import uuid
from datetime import UTC, datetime

from kms2.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeAssertionWrite,
    PropositionReference,
    ResolvedKnowledgeDraft,
    ResolvedPropositionMatchDraft,
    ResolvedPropositionMatchExisting,
)
from kms2.core.model.knowledge.proposition import (
    KnowledgeAssertion,
    Proposition,
    PropositionRecord,
    ReasoningStep,
)
from kms2.core.model.knowledge.reasoning import ReasoningStepRecord
from kms2.core.model.knowledge.retrieval import PropositionSearchCandidate
from kms2.database.knowledge.repository import KnowledgeRepository


class KnowledgeService:
    """Assign backend records and expose user-owned knowledge reads."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self._repository = repository

    async def read_revision(self, user_uuid: str) -> int:
        """Read the user's knowledge concurrency revision."""
        return await self._repository.read_revision(user_uuid)

    async def search_propositions(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
    ) -> list[PropositionSearchCandidate]:
        """Return bounded user-owned proposition candidates."""
        return await self._repository.search_propositions(
            user_uuid,
            query_embedding,
            candidate_limit=candidate_limit,
        )

    async def read_propositions(
        self,
        user_uuid: str,
        proposition_uuids: list[str],
    ) -> list[PropositionRecord]:
        """Read propositions and one-hop admission/argument references."""
        return await self._repository.read_propositions(
            user_uuid,
            proposition_uuids,
        )

    async def read_reasoning_steps(
        self,
        user_uuid: str,
        step_uuids: list[str],
    ) -> list[ReasoningStepRecord]:
        """Read selected user-owned reasoning steps."""
        return await self._repository.read_reasoning_steps(
            user_uuid, step_uuids
        )

    async def find_exact_proposition(
        self,
        user_uuid: str,
        content: str,
    ) -> Proposition | None:
        """Find an existing exact-content proposition."""
        return await self._repository.find_exact_proposition(user_uuid, content)

    async def find_reasoning_step(
        self,
        user_uuid: str,
        premise_uuids: list[str],
        conclusion_uuid: str,
        explanation: str,
        supporting_block_uuids: list[str],
    ) -> ReasoningStepRecord | None:
        """Find an exactly repeated owned argument."""
        return await self._repository.find_reasoning_step(
            user_uuid,
            premise_uuids,
            conclusion_uuid,
            explanation,
            supporting_block_uuids,
        )

    async def prepare_admission(
        self,
        user_uuid: str,
        expected_revision: int,
        draft: ResolvedKnowledgeDraft,
    ) -> KnowledgeAdmission | None:
        """Assign identities and exact turn provenance before persistence."""
        if not draft.propositions and not draft.reasoning_steps:
            return None

        now = datetime.now(UTC)
        proposition_uuids: list[str] = []
        new_propositions: list[Proposition] = []
        for resolved in draft.propositions:
            match = resolved.match
            if isinstance(match, ResolvedPropositionMatchExisting):
                proposition_uuid = match.uuid
            elif isinstance(match, ResolvedPropositionMatchDraft):
                proposition_uuid = proposition_uuids[match.position]
            else:
                proposition = Proposition(
                    uuid=str(uuid.uuid4()),
                    content=resolved.draft.content,
                    embedding=resolved.embedding,
                    created_at=now,
                )
                proposition_uuid = proposition.uuid
                new_propositions.append(proposition)
            proposition_uuids.append(proposition_uuid)

        def resolve_reference(reference: PropositionReference) -> str:
            if reference.kind == 'existing':
                return draft.existing_proposition_uuids[reference.position]
            return proposition_uuids[reference.position]

        def block_uuids(positions: list[int]) -> list[str]:
            return sorted(
                {
                    draft.supporting_block_uuids[position]
                    for position in positions
                }
            )

        used_block_positions: set[int] = set()
        assertion_supports: dict[str, set[str]] = {}
        assertion_steps: dict[str, set[str]] = {}
        for position, resolved in enumerate(draft.propositions):
            proposition_uuid = proposition_uuids[position]
            positions = set(resolved.draft.supporting_block_positions)
            used_block_positions.update(positions)
            assertion_supports.setdefault(proposition_uuid, set()).update(
                block_uuids(list(positions))
            )
            assertion_steps.setdefault(proposition_uuid, set())

        new_reasoning_steps: list[ReasoningStepRecord] = []
        existing_reasoning_step_uuids: set[str] = set()
        prepared_reasoning_steps: dict[
            tuple[tuple[str, ...], str, str, tuple[str, ...]], str
        ] = {}
        for step_draft in draft.reasoning_steps:
            premise_uuids = sorted(
                {
                    resolve_reference(reference)
                    for reference in step_draft.premises
                }
            )
            conclusion_uuid = resolve_reference(step_draft.conclusion)
            support_positions = set(step_draft.supporting_block_positions)
            used_block_positions.update(support_positions)
            support_uuids = block_uuids(list(support_positions))
            identity = (
                tuple(premise_uuids),
                conclusion_uuid,
                step_draft.explanation,
                tuple(support_uuids),
            )
            step_uuid = prepared_reasoning_steps.get(identity)
            if step_uuid is None:
                match = await self._repository.find_reasoning_step(
                    user_uuid,
                    premise_uuids,
                    conclusion_uuid,
                    step_draft.explanation,
                    support_uuids,
                )
                if match is None:
                    step = ReasoningStep(
                        uuid=str(uuid.uuid4()),
                        explanation=step_draft.explanation,
                        created_at=now,
                    )
                    step_uuid = step.uuid
                    new_reasoning_steps.append(
                        ReasoningStepRecord(
                            step=step,
                            premise_uuids=premise_uuids,
                            conclusion_uuid=conclusion_uuid,
                            supporting_block_uuids=support_uuids,
                        )
                    )
                else:
                    step_uuid = match.step.uuid
                    existing_reasoning_step_uuids.add(step_uuid)
                prepared_reasoning_steps[identity] = step_uuid
            assertion_steps.setdefault(conclusion_uuid, set()).add(step_uuid)

        assertion_writes = [
            KnowledgeAssertionWrite(
                assertion=KnowledgeAssertion(
                    uuid=str(uuid.uuid4()),
                    created_at=now,
                ),
                proposition_uuid=proposition_uuid,
                supporting_block_uuids=sorted(
                    assertion_supports.get(proposition_uuid, set())
                ),
                reasoning_step_uuids=sorted(
                    assertion_steps.get(proposition_uuid, set())
                ),
            )
            for proposition_uuid in sorted(assertion_steps)
        ]

        source_revisions = {
            (
                reference.source_uuid,
                reference.revision,
            ): reference
            for position, reference in enumerate(draft.block_source_revisions)
            if position in used_block_positions
        }
        return KnowledgeAdmission(
            expected_revision=expected_revision,
            source_revisions=[
                source_revisions[key] for key in sorted(source_revisions)
            ],
            new_propositions=new_propositions,
            proposition_uuids=proposition_uuids,
            assertions=assertion_writes,
            new_reasoning_steps=new_reasoning_steps,
            existing_reasoning_step_uuids=sorted(existing_reasoning_step_uuids),
        )
