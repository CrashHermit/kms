"""Backend record assignment and bounded knowledge read operations."""

import uuid
from datetime import UTC, datetime

from kms.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeAssertionWrite,
    PropositionReference,
    ResolvedKnowledgeDraft,
    ResolvedPropositionMatchDraft,
    ResolvedPropositionMatchExisting,
)
from kms.core.model.knowledge.argument import ArgumentRecord
from kms.core.model.knowledge.proposition import (
    Argument,
    KnowledgeAssertion,
    Proposition,
    PropositionRecord,
)
from kms.core.model.knowledge.retrieval import PropositionSearchCandidate
from kms.database.knowledge.repository import KnowledgeRepository


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

    async def read_arguments(
        self,
        user_uuid: str,
        argument_uuids: list[str],
    ) -> list[ArgumentRecord]:
        """Read selected user-owned arguments."""
        return await self._repository.read_arguments(user_uuid, argument_uuids)

    async def find_exact_proposition(
        self,
        user_uuid: str,
        content: str,
    ) -> Proposition | None:
        """Find an existing exact-content proposition."""
        return await self._repository.find_exact_proposition(user_uuid, content)

    async def find_argument(
        self,
        user_uuid: str,
        premise_uuids: list[str],
        conclusion_uuid: str,
        explanation: str,
        supporting_block_uuids: list[str],
    ) -> ArgumentRecord | None:
        """Find an exactly repeated owned argument."""
        return await self._repository.find_argument(
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
        if not draft.propositions and not draft.arguments:
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
        assertion_arguments: dict[str, set[str]] = {}
        for position, resolved in enumerate(draft.propositions):
            proposition_uuid = proposition_uuids[position]
            positions = set(resolved.draft.supporting_block_positions)
            used_block_positions.update(positions)
            assertion_supports.setdefault(proposition_uuid, set()).update(
                block_uuids(list(positions))
            )
            assertion_arguments.setdefault(proposition_uuid, set())

        new_arguments: list[ArgumentRecord] = []
        existing_argument_uuids: set[str] = set()
        prepared_arguments: dict[
            tuple[tuple[str, ...], str, str, tuple[str, ...]], str
        ] = {}
        for resolved_argument in draft.arguments:
            argument_draft = resolved_argument.draft
            premise_uuids = sorted(
                {
                    resolve_reference(reference)
                    for reference in argument_draft.premises
                }
            )
            conclusion_uuid = resolve_reference(argument_draft.conclusion)
            support_positions = set(argument_draft.supporting_block_positions)
            used_block_positions.update(support_positions)
            support_uuids = block_uuids(list(support_positions))
            identity = (
                tuple(premise_uuids),
                conclusion_uuid,
                argument_draft.explanation,
                tuple(support_uuids),
            )
            argument_uuid = prepared_arguments.get(identity)
            if argument_uuid is None:
                match = await self._repository.find_argument(
                    user_uuid,
                    premise_uuids,
                    conclusion_uuid,
                    argument_draft.explanation,
                    support_uuids,
                )
                if match is None:
                    argument = Argument(
                        uuid=str(uuid.uuid4()),
                        explanation=argument_draft.explanation,
                        embedding=resolved_argument.embedding,
                        created_at=now,
                    )
                    argument_uuid = argument.uuid
                    new_arguments.append(
                        ArgumentRecord(
                            argument=argument,
                            premise_uuids=premise_uuids,
                            conclusion_uuid=conclusion_uuid,
                            supporting_block_uuids=support_uuids,
                        )
                    )
                else:
                    argument_uuid = match.argument.uuid
                    existing_argument_uuids.add(argument_uuid)
                prepared_arguments[identity] = argument_uuid
            assertion_arguments.setdefault(conclusion_uuid, set()).add(
                argument_uuid
            )

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
                argument_uuids=sorted(
                    assertion_arguments.get(proposition_uuid, set())
                ),
            )
            for proposition_uuid in sorted(assertion_arguments)
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
            new_arguments=new_arguments,
            existing_argument_uuids=sorted(existing_argument_uuids),
        )
