"""Resolve draft proposition identity and prepare exact backend provenance."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from kms.core.embedding import Embedder
from kms.core.model.content import MultimodalContent
from kms.core.model.knowledge.admission import (
    KnowledgeAdmission,
    KnowledgeDraft,
    PropositionEquivalenceInput,
    PropositionIdentityCandidate,
    PropositionReference,
    ResolvedArgumentDraft,
    ResolvedKnowledgeDraft,
    ResolvedPropositionDraft,
    ResolvedPropositionMatchDraft,
    ResolvedPropositionMatchExisting,
)
from kms.core.model.knowledge.argument import argument_text
from kms.core.model.knowledge.proposition import Proposition
from kms.core.reranking import Reranker
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.windowing import pack_items
from kms.langgraph.knowledge.state import KnowledgeState
from kms.module.knowledge.equivalence import PropositionEquivalenceModule
from kms.service.knowledge import KnowledgeService


@dataclass(frozen=True, slots=True)
class _IdentityCandidate:
    """One existing or current-draft identity candidate."""

    kind: str
    uuid: str | None
    position: int | None
    content: str
    created_at: datetime | None


class KnowledgeIdentityNode:
    """Prepare proposition IDs before any database write transaction."""

    def __init__(
        self,
        knowledge: KnowledgeService,
        equivalence: PropositionEquivalenceModule,
        embedder: Embedder,
        reranker: Reranker,
        budget: TokenBudget,
        *,
        candidate_limit: int,
        result_limit: int,
    ) -> None:
        self._knowledge = knowledge
        self._equivalence = equivalence
        self._embedder = embedder
        self._reranker = reranker
        self._budget = budget
        self._candidate_limit = candidate_limit
        self._result_limit = result_limit

    async def run(
        self,
        state: KnowledgeState,
    ) -> dict[str, int | KnowledgeAdmission | None]:
        """Snapshot revision, resolve equivalent claims, and prepare admission."""
        draft = state.draft
        if draft is None or not draft.propositions and not draft.arguments:
            return {'admission': None}

        expected_revision = await self._knowledge.read_revision(state.user_uuid)
        parents = list(range(len(draft.propositions)))
        existing_matches: dict[int, list[Proposition]] = {}

        def find(position: int) -> int:
            while parents[position] != position:
                parents[position] = parents[parents[position]]
                position = parents[position]
            return position

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        exact_matches: list[Proposition | None] = []
        local_exact_positions: dict[int, int] = {}
        embeddings: dict[int, list[float]] = {}
        semantic_positions: list[int] = []
        for position, proposition_draft in enumerate(draft.propositions):
            exact = await self._knowledge.find_exact_proposition(
                state.user_uuid,
                proposition_draft.content,
            )
            exact_matches.append(exact)
            if exact is not None:
                embeddings[position] = exact.embedding
                continue
            previous_position = next(
                (
                    index
                    for index, previous in enumerate(
                        draft.propositions[:position]
                    )
                    if previous.content == proposition_draft.content
                ),
                None,
            )
            if previous_position is not None:
                local_exact_positions[position] = previous_position
                union(position, previous_position)
            else:
                semantic_positions.append(position)

        semantic_vectors = (
            await self._embedder.embed(
                [
                    MultimodalContent(text=draft.propositions[position].content)
                    for position in semantic_positions
                ],
                content_images=[],
                input_type='document',
            )
            if semantic_positions
            else []
        )
        for position, embedding in zip(
            semantic_positions,
            semantic_vectors,
            strict=True,
        ):
            embeddings[position] = embedding
        for position, previous_position in local_exact_positions.items():
            embeddings[position] = embeddings[previous_position]

        for position, proposition_draft in enumerate(draft.propositions):
            exact = exact_matches[position]
            if exact is None and position in local_exact_positions:
                continue
            candidates = await self._candidate_pool(
                state.user_uuid,
                proposition_draft.content,
                embeddings[position],
                draft,
                position,
                exact,
            )
            if exact is not None:
                existing_matches.setdefault(position, []).append(exact)
            equivalents = await self._judge_candidates(
                proposition_draft.content,
                candidates,
            )
            for candidate in equivalents:
                if candidate.kind == 'existing':
                    if all(
                        match.uuid != candidate.uuid
                        for match in existing_matches.setdefault(position, [])
                    ):
                        existing = await self._knowledge.read_propositions(
                            state.user_uuid,
                            [candidate.uuid],
                        )
                        existing_matches[position].append(
                            existing[0].proposition
                        )
                else:
                    union(position, candidate.position)
        class_existing: dict[int, list[Proposition]] = {}
        for position, matches in existing_matches.items():
            class_existing.setdefault(find(position), []).extend(matches)
        representatives: dict[int, int] = {}
        for position in range(len(draft.propositions)):
            root = find(position)
            representatives[root] = min(
                position, representatives.get(root, position)
            )

        resolved: list[ResolvedPropositionDraft] = []
        for position, proposition_draft in enumerate(draft.propositions):
            root = find(position)
            existing = class_existing.get(root, [])
            if existing:
                canonical = min(
                    existing,
                    key=lambda proposition: (
                        proposition.created_at,
                        proposition.uuid,
                    ),
                )
                match = ResolvedPropositionMatchExisting(
                    kind='existing',
                    uuid=canonical.uuid,
                )
            else:
                representative = representatives[root]
                match = (
                    None
                    if position == representative
                    else ResolvedPropositionMatchDraft(
                        kind='draft',
                        position=representative,
                    )
                )
            resolved.append(
                ResolvedPropositionDraft(
                    draft=proposition_draft,
                    embedding=embeddings[position],
                    match=match,
                )
            )
        canonical_existing_uuids: dict[tuple[str, int], str] = {}
        canonical_draft_positions: dict[tuple[str, int], int] = {}

        def resolve_reference(reference: PropositionReference) -> None:
            key = (reference.kind, reference.position)
            if (
                key in canonical_existing_uuids
                or key in canonical_draft_positions
            ):
                return
            if reference.kind == 'existing':
                canonical_existing_uuids[key] = (
                    state.existing_proposition_uuids[reference.position]
                )
                return

            position = reference.position
            match = resolved[position].match
            while isinstance(match, ResolvedPropositionMatchDraft):
                position = match.position
                match = resolved[position].match
            if isinstance(match, ResolvedPropositionMatchExisting):
                canonical_existing_uuids[key] = match.uuid
            else:
                canonical_draft_positions[key] = position

        for argument_draft in draft.arguments:
            for reference in (
                *argument_draft.premises,
                argument_draft.conclusion,
            ):
                resolve_reference(reference)

        existing_uuids = list(dict.fromkeys(canonical_existing_uuids.values()))
        existing_records = (
            await self._knowledge.read_propositions(
                state.user_uuid,
                existing_uuids,
            )
            if existing_uuids
            else []
        )
        existing_contents = {
            record.proposition.uuid: record.proposition.content
            for record in existing_records
        }

        def reference_content(reference: PropositionReference) -> str:
            key = (reference.kind, reference.position)
            if key in canonical_existing_uuids:
                return existing_contents[canonical_existing_uuids[key]]
            return draft.propositions[canonical_draft_positions[key]].content

        argument_texts = [
            argument_text(
                [reference_content(reference) for reference in step.premises],
                reference_content(step.conclusion),
                step.explanation,
            )
            for step in draft.arguments
        ]
        argument_embeddings = (
            await self._embedder.embed(
                [MultimodalContent(text=text) for text in argument_texts],
                content_images=[],
                input_type='document',
            )
            if argument_texts
            else []
        )
        resolved_arguments = [
            ResolvedArgumentDraft(draft=step, embedding=embedding)
            for step, embedding in zip(
                draft.arguments,
                argument_embeddings,
                strict=True,
            )
        ]

        prepared = ResolvedKnowledgeDraft(
            propositions=resolved,
            arguments=resolved_arguments,
            existing_proposition_uuids=state.existing_proposition_uuids,
            supporting_block_uuids=state.supporting_block_uuids,
            block_source_revisions=state.block_source_revisions,
        )
        admission = await self._knowledge.prepare_admission(
            state.user_uuid,
            expected_revision,
            prepared,
        )
        return {
            'expected_revision': expected_revision,
            'admission': admission,
        }

    async def _candidate_pool(
        self,
        user_uuid: str,
        target: str,
        embedding: list[float],
        draft: KnowledgeDraft,
        position: int,
        exact: Proposition | None,
    ) -> list[_IdentityCandidate]:
        """Rerank owned semantic candidates alongside earlier draft claims."""
        if exact is None:
            owned = await self._knowledge.search_propositions(
                user_uuid,
                embedding,
                candidate_limit=self._candidate_limit,
            )
        else:
            owned = []
        pool = [
            _IdentityCandidate(
                kind='existing',
                uuid=candidate.proposition.uuid,
                position=None,
                content=candidate.proposition.content,
                created_at=candidate.proposition.created_at,
            )
            for candidate in owned
        ]
        if exact is not None:
            pool.append(
                _IdentityCandidate(
                    kind='existing',
                    uuid=exact.uuid,
                    position=None,
                    content=exact.content,
                    created_at=exact.created_at,
                )
            )
        pool.extend(
            _IdentityCandidate(
                kind='draft',
                uuid=None,
                position=previous_position,
                content=previous.content,
                created_at=None,
            )
            for previous_position, previous in enumerate(
                draft.propositions[:position]
            )
        )
        unique: dict[tuple[str, str | int], _IdentityCandidate] = {}
        for candidate in pool:
            key: tuple[str, str | int] = (
                ('existing', candidate.uuid)
                if candidate.kind == 'existing'
                else ('draft', candidate.position)
            )
            unique[key] = candidate
        pool = list(unique.values())
        if not pool:
            return []
        ranked = await self._reranker.rerank(
            MultimodalContent(text=target),
            [MultimodalContent(text=candidate.content) for candidate in pool],
            content_images=[],
            top_n=min(self._result_limit, len(pool)),
        )
        return [pool[result['index']] for result in ranked]

    async def _judge_candidates(
        self,
        target: str,
        candidates: Sequence[_IdentityCandidate],
    ) -> list[_IdentityCandidate]:
        """Compare complete candidate claims in budgeted repeated envelopes."""
        direct_matches = [
            candidate for candidate in candidates if candidate.content == target
        ]
        candidates = [
            candidate for candidate in candidates if candidate.content != target
        ]
        if not candidates:
            return direct_matches
        base_request = PropositionEquivalenceInput(
            proposition_content=target,
            candidates=[],
        )
        base_cost = self._budget.counter.count_texts(
            [base_request.model_dump_json()]
        )[0]
        if base_cost > self._budget.token_limit:
            raise InputBudgetExceeded(
                'Proposition equivalence target exceeds the configured input budget'
            )
        positioned = [
            PropositionIdentityCandidate(position=index, content=item.content)
            for index, item in enumerate(candidates)
        ]
        item_costs = self._budget.counter.count_texts(
            [candidate.model_dump_json() for candidate in positioned]
        )
        comma_cost = self._budget.counter.count_texts([','])[0]
        batches = pack_items(
            positioned,
            token_counts=[cost + comma_cost for cost in item_costs],
            token_budget=self._budget.token_limit - base_cost,
        )
        matches: list[_IdentityCandidate] = []
        for batch in batches:
            batch_candidates = [
                candidates[candidate.position] for candidate in batch
            ]
            request = PropositionEquivalenceInput(
                proposition_content=target,
                candidates=[
                    PropositionIdentityCandidate(
                        position=index,
                        content=candidate.content,
                    )
                    for index, candidate in enumerate(batch_candidates)
                ],
            )
            decisions = await self._equivalence.acall(request=request)
            matches.extend(
                batch_candidates[decision.position]
                for decision in decisions
                if decision.is_equivalent
            )
        return direct_matches + matches
