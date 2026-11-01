"""Behavioral tests for identity-resolved knowledge admission."""

import asyncio
from datetime import UTC, datetime

from kms2.core.model.knowledge.admission import (
    KnowledgeDraft,
    KnowledgeExtractionInput,
    PropositionDraft,
)
from kms2.core.model.knowledge.proposition import Proposition
from kms2.core.model.knowledge.retrieval import PropositionSearchCandidate
from kms2.core.model.source import SourceRevisionReference
from kms2.core.windowing import TokenBudget
from kms2.langgraph.knowledge.state import KnowledgeState
from kms2.node.knowledge.identity import KnowledgeIdentityNode
from kms2.service.knowledge import KnowledgeService


class _TokenCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [max(1, len(text) // 4) for text in texts]


class _EmbeddingClient:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]


class _Reranker:
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, float | int]]:
        return [
            {'index': index, 'relevance_score': 1.0 - index / 10}
            for index in range(min(len(documents), top_n or len(documents)))
        ]


class _Equivalence:
    async def acall(self, *, request):
        from kms2.core.model.knowledge.admission import (
            PropositionEquivalenceDecision,
        )

        decisions = []
        for candidate in request.candidates:
            if request.proposition_content == 'early paraphrase':
                equivalent = False
            else:
                equivalent = candidate.content in {
                    'Canonical claim with its complete conditions.',
                    'early paraphrase',
                }
            decisions.append(
                PropositionEquivalenceDecision(
                    position=candidate.position,
                    is_equivalent=equivalent,
                )
            )
        return decisions


class _Repository:
    def __init__(self, existing: Proposition) -> None:
        self.existing = existing
        self.created_step_match = None

    async def read_revision(self, user_uuid: str) -> int:
        return 7

    async def search_propositions(
        self,
        user_uuid: str,
        query_embedding: list[float],
        *,
        candidate_limit: int,
    ) -> list[PropositionSearchCandidate]:
        return [
            PropositionSearchCandidate(
                proposition=self.existing,
                similarity=0.92,
            )
        ]

    async def find_exact_proposition(self, user_uuid: str, content: str):
        return self.existing if content == self.existing.content else None

    async def read_propositions(
        self, user_uuid: str, proposition_uuids: list[str]
    ):
        from kms2.core.model.knowledge.proposition import PropositionRecord

        return [
            PropositionRecord(
                proposition=self.existing,
                assertions=[],
                reasoning_step_uuids=[],
            )
        ]

    async def find_reasoning_step(self, *args):
        return self.created_step_match


async def _exercise_late_canonical_match_redirects_whole_draft_equivalence_class():
    existing = Proposition(
        uuid='canonical-claim',
        content='Canonical claim with its complete conditions.',
        embedding=[1.0, 0.0],
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    repository = _Repository(existing)
    node = KnowledgeIdentityNode(
        KnowledgeService(repository),
        _Equivalence(),
        _EmbeddingClient(),
        _Reranker(),
        TokenBudget(counter=_TokenCounter(), token_limit=8192),
        candidate_limit=32,
        result_limit=8,
    )
    state = KnowledgeState(
        user_uuid='user-1',
        request=KnowledgeExtractionInput(
            response_content='The response teaches one proposition.',
            blocks=[],
            existing_propositions=[],
        ),
        existing_proposition_uuids=[],
        supporting_block_uuids=[],
        block_source_revisions=[],
        evidence_image_bytes=[],
        draft=KnowledgeDraft(
            propositions=[
                PropositionDraft(
                    content='early paraphrase',
                    supporting_block_positions=[],
                ),
                PropositionDraft(
                    content='later form of same claim',
                    supporting_block_positions=[],
                ),
            ],
            reasoning_steps=[],
        ),
    )

    update = await node.run(state)

    admission = update['admission']
    assert admission.expected_revision == 7
    assert admission.proposition_uuids == ['canonical-claim', 'canonical-claim']
    assert admission.new_propositions == []
    assert len(admission.assertions) == 1
    assert admission.assertions[0].proposition_uuid == 'canonical-claim'


def test_late_canonical_match_redirects_whole_draft_equivalence_class():
    asyncio.run(
        _exercise_late_canonical_match_redirects_whole_draft_equivalence_class()
    )


def test_repeated_exact_new_claim_reuses_draft_identity():
    existing = Proposition(
        uuid='unrelated-claim',
        content='A different proposition.',
        embedding=[1.0, 0.0],
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    node = KnowledgeIdentityNode(
        KnowledgeService(_Repository(existing)),
        _Equivalence(),
        _EmbeddingClient(),
        _Reranker(),
        TokenBudget(counter=_TokenCounter(), token_limit=8192),
        candidate_limit=32,
        result_limit=8,
    )
    state = KnowledgeState(
        user_uuid='user-1',
        request=KnowledgeExtractionInput(
            response_content='The response teaches one proposition.',
            blocks=[],
            existing_propositions=[],
        ),
        existing_proposition_uuids=[],
        supporting_block_uuids=[],
        block_source_revisions=[],
        evidence_image_bytes=[],
        draft=KnowledgeDraft(
            propositions=[
                PropositionDraft(
                    content='A newly stated proposition.',
                    supporting_block_positions=[],
                ),
                PropositionDraft(
                    content='A newly stated proposition.',
                    supporting_block_positions=[],
                ),
            ],
            reasoning_steps=[],
        ),
    )

    admission = asyncio.run(node.run(state))['admission']

    assert admission.proposition_uuids[0] == admission.proposition_uuids[1]
    assert len(admission.new_propositions) == 1


async def _exercise_repeated_argument_is_resolved_once_per_admission():
    from kms2.core.model.knowledge.admission import (
        ExistingPropositionReference,
        ReasoningStepDraft,
        ResolvedKnowledgeDraft,
    )

    class Repository:
        def __init__(self):
            self.calls = 0

        async def find_reasoning_step(self, *args):
            self.calls += 1
            return None

    repository = Repository()
    reference = ExistingPropositionReference(kind='existing', position=0)
    repeated = ReasoningStepDraft(
        explanation='The premise entails the conclusion.',
        premises=[reference],
        conclusion=reference,
        supporting_block_positions=[],
    )
    admission = await KnowledgeService(repository).prepare_admission(
        'user-1',
        2,
        ResolvedKnowledgeDraft(
            propositions=[],
            reasoning_steps=[repeated, repeated],
            existing_proposition_uuids=['proposition-existing'],
            supporting_block_uuids=[],
            block_source_revisions=[],
        ),
    )

    assert repository.calls == 1
    assert len(admission.new_reasoning_steps) == 1
    assert admission.assertions[0].reasoning_step_uuids == [
        admission.new_reasoning_steps[0].step.uuid
    ]


def test_repeated_argument_is_resolved_once_per_admission():
    asyncio.run(_exercise_repeated_argument_is_resolved_once_per_admission())


async def _exercise_preparation_keeps_per_turn_support_and_deduplicates_block_tokens():
    from kms2.core.model.knowledge.admission import (
        DraftPropositionReference,
        ReasoningStepDraft,
        ResolvedKnowledgeDraft,
        ResolvedPropositionDraft,
    )

    class Repository:
        async def find_reasoning_step(self, *args):
            return None

    service = KnowledgeService(Repository())
    direct = PropositionDraft(
        content='A claim with a scoped condition.',
        supporting_block_positions=[1],
    )
    explanation = ReasoningStepDraft(
        explanation='The source definition justifies the claim.',
        premises=[DraftPropositionReference(kind='draft', position=0)],
        conclusion=DraftPropositionReference(kind='draft', position=0),
        supporting_block_positions=[1],
    )
    admission = await service.prepare_admission(
        'user-1',
        11,
        ResolvedKnowledgeDraft(
            propositions=[
                ResolvedPropositionDraft(
                    draft=direct,
                    embedding=[0.5, 0.5],
                    match=None,
                )
            ],
            reasoning_steps=[explanation],
            existing_proposition_uuids=[],
            supporting_block_uuids=['block-unrelated', 'block-1'],
            block_source_revisions=[
                SourceRevisionReference(
                    source_uuid='unused-source', revision=2
                ),
                SourceRevisionReference(source_uuid='source-1', revision=4),
            ],
        ),
    )

    assert admission is not None
    assert len(admission.new_propositions) == 1
    assert admission.assertions[0].supporting_block_uuids == ['block-1']
    assert admission.assertions[0].reasoning_step_uuids == [
        admission.new_reasoning_steps[0].step.uuid
    ]
    assert admission.new_reasoning_steps[0].supporting_block_uuids == [
        'block-1'
    ]
    assert admission.source_revisions == [
        SourceRevisionReference(source_uuid='source-1', revision=4)
    ]


def test_preparation_keeps_per_turn_support_and_deduplicates_block_tokens():
    asyncio.run(
        _exercise_preparation_keeps_per_turn_support_and_deduplicates_block_tokens()
    )


async def _exercise_existing_zero_premise_reasoning_step_is_reused_exactly():
    from kms2.core.model.knowledge.admission import (
        ExistingPropositionReference,
        ReasoningStepDraft,
        ResolvedKnowledgeDraft,
    )
    from kms2.core.model.knowledge.proposition import ReasoningStep
    from kms2.core.model.knowledge.reasoning import ReasoningStepRecord

    step = ReasoningStepRecord(
        step=ReasoningStep(
            uuid='step-existing',
            explanation='A direct definition establishes the result.',
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        premise_uuids=[],
        conclusion_uuid='proposition-existing',
        supporting_block_uuids=[],
    )

    class Repository:
        async def find_reasoning_step(self, *args):
            assert args[1] == []
            return step

    admission = await KnowledgeService(Repository()).prepare_admission(
        'user-1',
        2,
        ResolvedKnowledgeDraft(
            propositions=[],
            reasoning_steps=[
                ReasoningStepDraft(
                    explanation=step.step.explanation,
                    premises=[],
                    conclusion=ExistingPropositionReference(
                        kind='existing',
                        position=0,
                    ),
                    supporting_block_positions=[],
                )
            ],
            existing_proposition_uuids=['proposition-existing'],
            supporting_block_uuids=[],
            block_source_revisions=[],
        ),
    )

    assert admission is not None
    assert admission.new_reasoning_steps == []
    assert admission.existing_reasoning_step_uuids == ['step-existing']
    assert admission.assertions[0].proposition_uuid == 'proposition-existing'
    assert admission.assertions[0].reasoning_step_uuids == ['step-existing']


def test_existing_zero_premise_reasoning_step_is_reused_exactly():
    asyncio.run(
        _exercise_existing_zero_premise_reasoning_step_is_reused_exactly()
    )
