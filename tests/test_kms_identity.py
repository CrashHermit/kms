"""Behavioral tests for owner-scoped proposition identity auditing."""

import asyncio

import dspy
import pytest
from dspy.utils import DummyLM

from kms.config.inference import TextInferenceSettings
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import NativePredictor, generation_signature
from kms.knowledge.candidates import NewProposition
from kms.knowledge.identity.audit import IdentityAuditor
from kms.knowledge.identity.inference import IdentityModule, IdentitySignature
from kms.knowledge.identity.models import IdentityDecision, IdentityRequest
from kms.knowledge.identity.settings import IdentitySettings
from kms.knowledge.models import Proposition


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Embedder:
    def __init__(self) -> None:
        self.requests = []

    async def embed(self, request):
        self.requests.append(request)
        return [[float(len(request.materials[0].text or ''))]]


class _Repository:
    def __init__(self, matches: list[Proposition]) -> None:
        self.matches = matches
        self.searches = []

    async def search(self, user_uuid, embedding, *, limit, embedding_model):
        self.searches.append((user_uuid, embedding, limit, embedding_model))
        return self.matches


class _Decisions(dspy.Module):
    def __init__(self, decisions: list[str | None]) -> None:
        super().__init__()
        self.decisions = iter(decisions)
        self.requests: list[IdentityRequest] = []

    async def aforward(self, *, request: IdentityRequest) -> dspy.Prediction:
        match_uuid = next(self.decisions)
        if match_uuid == 'earlier':
            match_uuid = request.matches[0].uuid
        return dspy.Prediction(decision=IdentityDecision(match_uuid=match_uuid))


def _auditor(
    repository: _Repository,
    module: _Decisions,
    *,
    token_limit: int = 100_000,
    candidate_limit: int = 16,
) -> tuple[IdentityAuditor, _Embedder]:
    embedder = _Embedder()
    budget = TokenBudget(
        MaterialTokenEstimator(_Counter()),
        token_limit,
    )
    return (
        IdentityAuditor(
            repository,
            embedder,
            'embedding-v1',
            IdentityModule(module),
            budget,
            candidate_limit=candidate_limit,
        ),
        embedder,
    )


def test_auditor_reuses_owner_scoped_existing_match() -> None:
    existing = Proposition(uuid='stored', name='Claim', statement='Same claim')
    repository = _Repository([existing])
    module = _Decisions(['stored'])
    auditor, embedder = _auditor(repository, module, candidate_limit=3)
    candidate = NewProposition(id='local', name='Claim', statement='Same claim')

    result = asyncio.run(auditor.audit('owner', [candidate]))

    assert result.canonical == {'local': existing}
    assert result.new_propositions == []
    assert repository.searches == [('owner', [10.0], 3, 'embedding-v1')]
    assert embedder.requests[0].input_type == 'query'


def test_auditor_reuses_earlier_batch_canonical_candidate() -> None:
    repository = _Repository([])
    module = _Decisions(['earlier'])
    auditor, _ = _auditor(repository, module)
    candidates = [
        NewProposition(id='first', name='First', statement='A claim'),
        NewProposition(id='second', name='Second', statement='A claim'),
    ]

    result = asyncio.run(auditor.audit('owner', candidates))

    assert result.canonical['second'] is result.canonical['first']
    assert result.new_propositions == [result.canonical['first']]


def test_auditor_creates_non_equivalent_candidate_and_returns_structure() -> (
    None
):
    existing = Proposition(
        uuid='related',
        name='Related',
        statement='A related but narrower claim',
    )
    repository = _Repository([existing])
    module = _Decisions([None])
    auditor, _ = _auditor(repository, module)
    candidate = NewProposition(
        id='new', name='New', statement='A distinct claim'
    )

    result = asyncio.run(auditor.audit('owner', [candidate]))

    assert result.canonical['new'].name == 'New'
    assert result.canonical['new'].statement == 'A distinct claim'
    assert result.canonical['new'] is result.new_propositions[0]
    assert result.canonical['new'].uuid != existing.uuid


def test_auditor_rejects_match_that_cannot_fit_complete_request() -> None:
    existing = Proposition(uuid='stored', name='Claim', statement='Same claim')
    repository = _Repository([existing])
    module = _Decisions([])
    auditor, _ = _auditor(repository, module, token_limit=1)
    candidate = NewProposition(id='local', name='Claim', statement='Same claim')

    with pytest.raises(InputBudgetExceeded):
        asyncio.run(auditor.audit('owner', [candidate]))


def test_identity_signature_parses_a_native_structured_decision():
    existing = Proposition(
        uuid='existing', name='Claim', statement='Positive speeds only.'
    )
    candidate = NewProposition(
        id='new', name='Claim', statement='For positive speeds.'
    )
    lm = DummyLM([{'decision': {'match_uuid': 'existing'}}])
    backend = dspy.Predict(generation_signature(IdentitySignature))
    backend.set_lm(lm)
    module = IdentityModule(
        NativePredictor(
            backend,
            IdentitySignature,
            budget=TokenBudget(MaterialTokenEstimator(_Counter()), 100000),
            image_budget=None,
            supports_images=False,
        )
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        decision = asyncio.run(
            module.acall(
                request=IdentityRequest(candidate=candidate, matches=[existing])
            )
        )
    assert decision.match_uuid == existing.uuid


def test_identity_contract_and_default_settings() -> None:
    assert IdentityDecision(match_uuid=None).match_uuid is None
    assert IdentitySettings().inference == TextInferenceSettings(num_retries=0)
    assert IdentitySettings().candidate_limit == 16
