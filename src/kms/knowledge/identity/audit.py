"""Resolve proposition candidates to owner-scoped canonical vertices."""

from functools import partial

from kms.concurrency import run_blocking_work
from kms.core.embedding import Embedder, EmbeddingRequest
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.inference.generation import fits_generation_request
from kms.knowledge.candidates import NewProposition
from kms.knowledge.identity.inference import IdentityModule
from kms.knowledge.identity.models import AuditResult, IdentityRequest
from kms.knowledge.models import Proposition
from kms.knowledge.storage import KnowledgeRepository
from kms.media.models import Material


class IdentityAuditor:
    """Audit candidate equivalence before any proposition is committed."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        embedder: Embedder,
        embedding_model: str,
        module: IdentityModule,
        budget: TokenBudget,
        *,
        candidate_limit: int,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._embedding_model = embedding_model
        self._module = module
        self._budget = budget
        self._candidate_limit = candidate_limit

    async def audit(
        self, user_uuid: str, candidates: list[NewProposition]
    ) -> AuditResult:
        """Return canonical choices while preserving local candidate order."""
        canonical: dict[str, Proposition] = {}
        new_propositions: list[Proposition] = []

        for candidate in candidates:
            embedding_request = EmbeddingRequest(
                materials=[Material.from_text(candidate.statement)],
                input_type='query',
            )
            [embedding] = await self._embedder.embed(embedding_request)
            search_matches = await self._repository.search(
                user_uuid,
                embedding,
                limit=self._candidate_limit,
                embedding_model=self._embedding_model,
            )
            matches = await _matching_propositions(
                candidate,
                list(canonical.values()),
                search_matches,
                budget=self._budget,
            )

            if matches:
                decision = await self._module.acall(
                    request=IdentityRequest(
                        candidate=candidate,
                        matches=matches,
                    )
                )
                if decision.match_uuid is not None:
                    canonical[candidate.id] = next(
                        proposition
                        for proposition in matches
                        if proposition.uuid == decision.match_uuid
                    )
                    continue

            proposition = Proposition(
                name=candidate.name,
                statement=candidate.statement,
            )
            canonical[candidate.id] = proposition
            new_propositions.append(proposition)

        return AuditResult(
            canonical=canonical,
            new_propositions=new_propositions,
        )


async def _matching_propositions(
    candidate: NewProposition,
    earlier: list[Proposition],
    search_matches: list[Proposition],
    *,
    budget: TokenBudget,
) -> list[Proposition]:
    """Pack earlier local choices first, then search hits fitting full prompts."""
    matches: list[Proposition] = []
    included: set[str] = set()
    for proposition in earlier:
        if proposition.uuid in included:
            continue
        request = IdentityRequest(
            candidate=candidate,
            matches=[*matches, proposition],
        )
        if not await _fits(request, budget):
            raise InputBudgetExceeded(
                'Earlier batch proposition does not fit the identity request'
            )
        matches.append(proposition)
        included.add(proposition.uuid)

    for proposition in search_matches:
        if proposition.uuid in included:
            continue
        request = IdentityRequest(
            candidate=candidate,
            matches=[*matches, proposition],
        )
        if await _fits(request, budget):
            matches.append(proposition)
            included.add(proposition.uuid)
        elif not matches:
            raise InputBudgetExceeded(
                'First identity candidate does not fit the complete request'
            )
        else:
            break
    return matches


async def _fits(request: IdentityRequest, budget: TokenBudget) -> bool:
    return await run_blocking_work(
        partial(
            fits_generation_request,
            {'request': request},
            budget=budget,
            image_budget=None,
        )
    )
