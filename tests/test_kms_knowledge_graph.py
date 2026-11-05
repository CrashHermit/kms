"""Behavioral tests for identity-resolved knowledge admission."""

import asyncio
from datetime import UTC, datetime
from io import BytesIO

import pytest
from PIL import Image

from kms.core.model.content import MultimodalContent
from kms.core.model.knowledge.admission import (
    KnowledgeDraft,
    KnowledgeExtractionInput,
    KnowledgeResponseSegment,
    PropositionDraft,
)
from kms.core.model.knowledge.proposition import Proposition
from kms.core.model.knowledge.retrieval import PropositionSearchCandidate
from kms.core.model.source import SourceRevisionReference
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.langgraph.knowledge.state import KnowledgeState
from kms.node.knowledge.extraction import KnowledgeExtractionNode
from kms.node.knowledge.identity import KnowledgeIdentityNode
from kms.service.knowledge import KnowledgeService


class _TokenCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [max(1, len(text) // 4) for text in texts]


def test_knowledge_extraction_counts_each_native_image_before_dispatch():
    class Extractor:
        def __init__(self):
            self.images = None

        async def acall(self, *, request, content_images):
            self.images = content_images
            return KnowledgeDraft(propositions=[], arguments=[])

    image_buffer = BytesIO()
    Image.new('RGB', (2, 2), 'green').save(image_buffer, format='PNG')
    image_bytes = image_buffer.getvalue()
    request = KnowledgeExtractionInput(
        segments=[],
        blocks=[],
        existing_propositions=[],
    )
    state = KnowledgeState(
        user_uuid='user-1',
        request=request,
        existing_proposition_uuids=[],
        supporting_block_uuids=[],
        block_source_revisions=[],
        content_image_bytes=[image_bytes, image_bytes],
    )
    counter = MaterialTokenEstimator(_TokenCounter())
    payload_tokens = counter.count_texts([request.model_dump_json()])[0]

    rejected_extractor = Extractor()
    rejected = KnowledgeExtractionNode(
        rejected_extractor,
        TokenBudget(counter=counter, token_limit=payload_tokens + 3),
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(rejected.run(state))
    assert rejected_extractor.images is None

    accepted_extractor = Extractor()
    accepted = KnowledgeExtractionNode(
        accepted_extractor,
        TokenBudget(counter=counter, token_limit=payload_tokens + 4),
    )
    asyncio.run(accepted.run(state))
    assert len(accepted_extractor.images) == 2


class _FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(
        self, materials, *, content_images, input_type
    ) -> list[list[float]]:
        assert content_images == []
        assert input_type == 'document'
        self.calls.append([material.text for material in materials])
        return [[float(index + 1), 1.0] for index, _ in enumerate(materials)]


class _Reranker:
    async def rerank(
        self,
        query,
        documents,
        *,
        content_images,
        top_n: int | None = None,
    ) -> list[dict[str, float | int]]:
        assert content_images == []
        return [
            {'index': index, 'relevance_score': 1.0 - index / 10}
            for index in range(min(len(documents), top_n or len(documents)))
        ]


class _Equivalence:
    async def acall(self, *, request):
        from kms.core.model.knowledge.admission import (
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
        self.created_argument_match = None

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
        from kms.core.model.knowledge.proposition import PropositionRecord

        return [
            PropositionRecord(
                proposition=self.existing,
                assertions=[],
                argument_uuids=[],
            )
        ]

    async def find_argument(self, *args):
        return self.created_argument_match


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
        _FakeEmbedder(),
        _Reranker(),
        TokenBudget(
            counter=MaterialTokenEstimator(_TokenCounter()), token_limit=8192
        ),
        candidate_limit=32,
        result_limit=8,
    )
    state = KnowledgeState(
        user_uuid='user-1',
        request=KnowledgeExtractionInput(
            segments=[
                KnowledgeResponseSegment(
                    material=MultimodalContent(
                        text='The response teaches one proposition.'
                    ),
                    block_positions=[],
                    image_positions=[],
                    proposition_positions=[],
                )
            ],
            blocks=[],
            existing_propositions=[],
        ),
        existing_proposition_uuids=[],
        supporting_block_uuids=[],
        block_source_revisions=[],
        content_image_bytes=[],
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
            arguments=[],
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
    embedder = _FakeEmbedder()
    node = KnowledgeIdentityNode(
        KnowledgeService(_Repository(existing)),
        _Equivalence(),
        embedder,
        _Reranker(),
        TokenBudget(
            counter=MaterialTokenEstimator(_TokenCounter()), token_limit=8192
        ),
        candidate_limit=32,
        result_limit=8,
    )
    state = KnowledgeState(
        user_uuid='user-1',
        request=KnowledgeExtractionInput(
            segments=[
                KnowledgeResponseSegment(
                    material=MultimodalContent(
                        text='The response teaches one proposition.'
                    ),
                    block_positions=[],
                    image_positions=[],
                    proposition_positions=[],
                )
            ],
            blocks=[],
            existing_propositions=[],
        ),
        existing_proposition_uuids=[],
        supporting_block_uuids=[],
        block_source_revisions=[],
        content_image_bytes=[],
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
            arguments=[],
        ),
    )

    admission = asyncio.run(node.run(state))['admission']

    assert admission.proposition_uuids[0] == admission.proposition_uuids[1]
    assert len(admission.new_propositions) == 1
    assert embedder.calls == [['A newly stated proposition.']]


async def _exercise_repeated_argument_is_resolved_once_per_admission():
    from kms.core.model.knowledge.admission import (
        ArgumentDraft,
        ExistingPropositionReference,
        ResolvedArgumentDraft,
        ResolvedKnowledgeDraft,
    )

    class Repository:
        def __init__(self):
            self.calls = 0

        async def find_argument(self, *args):
            self.calls += 1
            return None

    repository = Repository()
    reference = ExistingPropositionReference(kind='existing', position=0)
    repeated = ArgumentDraft(
        explanation='The premise entails the conclusion.',
        premises=[reference],
        conclusion=reference,
        supporting_block_positions=[],
    )
    resolved_repeated = ResolvedArgumentDraft(
        draft=repeated,
        embedding=[0.5, 0.5],
    )
    admission = await KnowledgeService(repository).prepare_admission(
        'user-1',
        2,
        ResolvedKnowledgeDraft(
            propositions=[],
            arguments=[resolved_repeated, resolved_repeated],
            existing_proposition_uuids=['proposition-existing'],
            supporting_block_uuids=[],
            block_source_revisions=[],
        ),
    )

    assert repository.calls == 1
    assert len(admission.new_arguments) == 1
    assert admission.new_arguments[0].argument.embedding == [0.5, 0.5]
    assert admission.assertions[0].argument_uuids == [
        admission.new_arguments[0].argument.uuid
    ]


def test_repeated_argument_is_resolved_once_per_admission():
    asyncio.run(_exercise_repeated_argument_is_resolved_once_per_admission())


async def _exercise_preparation_keeps_per_turn_support_and_deduplicates_block_tokens():
    from kms.core.model.knowledge.admission import (
        ArgumentDraft,
        DraftPropositionReference,
        ResolvedArgumentDraft,
        ResolvedKnowledgeDraft,
        ResolvedPropositionDraft,
    )

    class Repository:
        async def find_argument(self, *args):
            return None

    service = KnowledgeService(Repository())
    direct = PropositionDraft(
        content='A claim with a scoped condition.',
        supporting_block_positions=[1],
    )
    explanation = ArgumentDraft(
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
            arguments=[
                ResolvedArgumentDraft(
                    draft=explanation,
                    embedding=[0.25, 0.75],
                )
            ],
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
    assert admission.assertions[0].argument_uuids == [
        admission.new_arguments[0].argument.uuid
    ]
    assert admission.new_arguments[0].supporting_block_uuids == ['block-1']
    assert admission.new_arguments[0].argument.embedding == [0.25, 0.75]
    assert admission.source_revisions == [
        SourceRevisionReference(source_uuid='source-1', revision=4)
    ]


def test_preparation_keeps_per_turn_support_and_deduplicates_block_tokens():
    asyncio.run(
        _exercise_preparation_keeps_per_turn_support_and_deduplicates_block_tokens()
    )


async def _exercise_existing_zero_premise_argument_is_reused_exactly():
    from kms.core.model.knowledge.admission import (
        ArgumentDraft,
        ExistingPropositionReference,
        ResolvedArgumentDraft,
        ResolvedKnowledgeDraft,
    )
    from kms.core.model.knowledge.argument import ArgumentRecord
    from kms.core.model.knowledge.proposition import Argument

    argument = ArgumentRecord(
        argument=Argument(
            uuid='argument-existing',
            explanation='A direct definition establishes the result.',
            embedding=[0.6, 0.8],
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        premise_uuids=[],
        conclusion_uuid='proposition-existing',
        supporting_block_uuids=[],
    )

    class Repository:
        async def find_argument(self, *args):
            assert args[1] == []
            return argument

    admission = await KnowledgeService(Repository()).prepare_admission(
        'user-1',
        2,
        ResolvedKnowledgeDraft(
            propositions=[],
            arguments=[
                ResolvedArgumentDraft(
                    draft=ArgumentDraft(
                        explanation=argument.argument.explanation,
                        premises=[],
                        conclusion=ExistingPropositionReference(
                            kind='existing',
                            position=0,
                        ),
                        supporting_block_positions=[],
                    ),
                    embedding=[0.1, 0.9],
                )
            ],
            existing_proposition_uuids=['proposition-existing'],
            supporting_block_uuids=[],
            block_source_revisions=[],
        ),
    )

    assert admission is not None
    assert admission.new_arguments == []
    assert admission.existing_argument_uuids == ['argument-existing']
    assert admission.assertions[0].proposition_uuid == 'proposition-existing'
    assert admission.assertions[0].argument_uuids == ['argument-existing']
    assert argument.argument.embedding == [0.6, 0.8]


def test_existing_zero_premise_argument_is_reused_exactly():
    asyncio.run(_exercise_existing_zero_premise_argument_is_reused_exactly())


async def _exercise_identity_embeds_complete_canonical_arguments():
    from kms.core.model.knowledge.admission import (
        ArgumentDraft,
        DraftPropositionReference,
        ExistingPropositionReference,
        KnowledgeExistingProposition,
    )
    from kms.core.model.knowledge.argument import argument_text
    from kms.core.model.knowledge.proposition import PropositionRecord

    created_at = datetime(2025, 1, 1, tzinfo=UTC)
    existing = {
        'existing-premise': Proposition(
            uuid='existing-premise',
            content='The first premise.',
            embedding=[1.0, 0.0],
            created_at=created_at,
        ),
        'existing-conclusion': Proposition(
            uuid='existing-conclusion',
            content='The conclusion.',
            embedding=[0.0, 1.0],
            created_at=created_at,
        ),
    }

    class Repository:
        def __init__(self):
            self.read_calls: list[list[str]] = []

        async def read_revision(self, user_uuid: str) -> int:
            return 7

        async def find_exact_proposition(self, user_uuid: str, content: str):
            return None

        async def search_propositions(
            self,
            user_uuid: str,
            query_embedding: list[float],
            *,
            candidate_limit: int,
        ) -> list[PropositionSearchCandidate]:
            return []

        async def read_propositions(
            self,
            user_uuid: str,
            proposition_uuids: list[str],
        ) -> list[PropositionRecord]:
            self.read_calls.append(proposition_uuids)
            return [
                PropositionRecord(
                    proposition=existing[proposition_uuid],
                    assertions=[],
                    argument_uuids=[],
                )
                for proposition_uuid in proposition_uuids
            ]

        async def find_argument(self, *args):
            return None

    class FakeEmbedder:
        def __init__(self):
            self.calls: list[list[str]] = []

        async def embed(
            self, materials, *, content_images, input_type
        ) -> list[list[float]]:
            assert content_images == []
            assert input_type == 'document'
            texts = [material.text for material in materials]
            self.calls.append(texts)
            return [[float(index + 1), 0.5] for index in range(len(materials))]

    repository = Repository()
    embedder = FakeEmbedder()
    node = KnowledgeIdentityNode(
        KnowledgeService(repository),
        _Equivalence(),
        embedder,
        _Reranker(),
        TokenBudget(
            counter=MaterialTokenEstimator(_TokenCounter()), token_limit=8192
        ),
        candidate_limit=32,
        result_limit=8,
    )
    existing_premise = ExistingPropositionReference(kind='existing', position=0)
    existing_conclusion = ExistingPropositionReference(
        kind='existing', position=1
    )
    draft_zero = DraftPropositionReference(kind='draft', position=0)
    draft_one = DraftPropositionReference(kind='draft', position=1)
    first_argument = ArgumentDraft(
        explanation='Combines the existing and new claims.',
        premises=[existing_premise, draft_zero, draft_one, draft_zero],
        conclusion=existing_conclusion,
        supporting_block_positions=[],
    )
    zero_premise_argument = ArgumentDraft(
        explanation='A direct claim has no premises.',
        premises=[],
        conclusion=draft_one,
        supporting_block_positions=[],
    )
    state = KnowledgeState(
        user_uuid='user-1',
        request=KnowledgeExtractionInput(
            segments=[
                KnowledgeResponseSegment(
                    material=MultimodalContent(text='response'),
                    block_positions=[],
                    image_positions=[],
                    proposition_positions=[],
                )
            ],
            blocks=[],
            existing_propositions=[
                KnowledgeExistingProposition(
                    position=0,
                    material=MultimodalContent(text='The first premise.'),
                ),
                KnowledgeExistingProposition(
                    position=1,
                    material=MultimodalContent(text='The conclusion.'),
                ),
            ],
        ),
        existing_proposition_uuids=[
            'existing-premise',
            'existing-conclusion',
        ],
        supporting_block_uuids=[],
        block_source_revisions=[],
        content_image_bytes=[],
        draft=KnowledgeDraft(
            propositions=[
                PropositionDraft(
                    content='A newly resolved claim.',
                    supporting_block_positions=[],
                ),
                PropositionDraft(
                    content='A newly resolved claim.',
                    supporting_block_positions=[],
                ),
            ],
            arguments=[first_argument, zero_premise_argument],
        ),
    )

    admission = (await node.run(state))['admission']
    first_argument = argument_text(
        ['The first premise.', 'A newly resolved claim.'],
        'The conclusion.',
        first_argument.explanation,
    )
    zero_premise_argument = argument_text(
        [],
        'A newly resolved claim.',
        zero_premise_argument.explanation,
    )

    assert embedder.calls == [
        ['A newly resolved claim.'],
        [first_argument, zero_premise_argument],
    ]
    assert repository.read_calls == [
        ['existing-premise', 'existing-conclusion']
    ]
    assert admission.proposition_uuids[0] == admission.proposition_uuids[1]
    assert set(admission.new_arguments[0].premise_uuids) == {
        admission.new_propositions[0].uuid,
        'existing-premise',
    }
    assert admission.new_arguments[1].premise_uuids == []
    assert [
        record.argument.embedding for record in admission.new_arguments
    ] == [[1.0, 0.5], [2.0, 0.5]]


def test_argument_text_sorts_and_deduplicates_premises():
    from kms.core.model.knowledge.argument import argument_text

    assert argument_text(['second', 'first', 'first'], 'result', 'proof') == (
        '{"conclusion":"result","explanation":"proof",'
        '"premises":["first","second"]}'
    )


def test_identity_embeds_complete_canonical_arguments():
    asyncio.run(_exercise_identity_embeds_complete_canonical_arguments())
