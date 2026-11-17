"""Demand-driven reasoning over complete proposition and argument context."""

import asyncio

import dspy
import pytest
from dspy.utils import DummyLM

from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import NativePredictor, generation_signature
from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewArgument,
    NewProposition,
    NewPropositionReference,
)
from kms.knowledge.commit import KnowledgeCommitNode
from kms.knowledge.models import (
    Argument,
    KnowledgeContext,
    Proposition,
    StoredKnowledge,
)
from kms.knowledge.reasoning.graph import KnowledgeReasoningGraph
from kms.knowledge.reasoning.inference import (
    KnowledgeReasoningModule,
    KnowledgeReasoningSignature,
)
from kms.knowledge.reasoning.models import ReasoningCandidates, ReasoningRequest
from kms.knowledge.reasoning.node import KnowledgeReasoningNode
from kms.knowledge.retrieval.node import KnowledgeSearchNode


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _CapturingModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[ReasoningRequest] = []

    async def aforward(self, *, request: ReasoningRequest) -> dspy.Prediction:
        self.requests.append(request)
        return dspy.Prediction(
            candidates=ReasoningCandidates(
                new_propositions=[], new_arguments=[]
            )
        )


def _context(
    propositions: list[Proposition],
    *,
    matched: list[Proposition] | None = None,
    arguments: list[Argument] | None = None,
) -> KnowledgeContext:
    return KnowledgeContext(
        matched=propositions[:1] if matched is None else matched,
        propositions=propositions,
        arguments=arguments or [],
    )


def _reasoner(
    module: dspy.Module, *, candidate_limit: int = 16
) -> KnowledgeReasoningNode:
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100000)
    return KnowledgeReasoningNode(
        KnowledgeReasoningModule(module),
        budget,
        candidate_limit=candidate_limit,
        argument_limit=16,
    )


def _run(
    node: KnowledgeReasoningNode,
    context: KnowledgeContext,
    understanding: str = 'Understand the result',
) -> dict:
    return asyncio.run(
        node.run(
            {
                'understanding': understanding,
                'context': context,
            }
        )
    )


def test_empty_context_returns_empty_without_prediction() -> None:
    module = _CapturingModule()
    result = _run(_reasoner(module), _context([]))

    assert result['pending'].new_propositions == []
    assert result['pending'].new_arguments == []
    assert result['has_reasoned'] is True
    assert module.requests == []


def test_context_budget_never_sends_argument_with_dangling_propositions() -> (
    None
):
    first = Proposition(uuid='p1', name='First', statement='a' * 20)
    second = Proposition(uuid='p2', name='Second', statement='b' * 20)
    third = Proposition(uuid='p3', name='Third', statement='c' * 20)
    argument = Argument(
        uuid='a1',
        premise_uuids=['p1', 'p2'],
        conclusion_uuid='p3',
    )
    module = _CapturingModule()

    _run(
        _reasoner(module, candidate_limit=2),
        _context([first, second, third], arguments=[argument]),
    )

    request = module.requests[0]
    included = {
        proposition.uuid for proposition in request.context.propositions
    }
    assert [
        proposition.uuid for proposition in request.context.propositions
    ] == [
        'p1',
        'p2',
    ]
    assert request.context.arguments == []
    assert {'p1', 'p2', 'p3'} - included


def test_fitting_endpoints_are_kept_when_complete_argument_exceeds_budget():
    conclusion = Proposition(uuid='p1', name='Conclusion', statement='c' * 20)
    premise = Proposition(uuid='p2', name='Premise', statement='p' * 20)
    argument = Argument(uuid='a1', premise_uuids=['p2'], conclusion_uuid='p1')
    context = _context(
        [conclusion, premise],
        matched=[conclusion],
        arguments=[argument],
    )
    estimator = MaterialTokenEstimator(_Counter())
    endpoint_request = ReasoningRequest(
        understanding='Understand',
        context=KnowledgeContext(
            matched=[conclusion],
            propositions=[conclusion, premise],
            arguments=[],
        ),
    )
    budget = TokenBudget(
        estimator,
        estimator.estimate(
            text=f'{{"request":{endpoint_request.model_dump_json()}}}'
        ),
    )
    module = _CapturingModule()
    node = KnowledgeReasoningNode(
        KnowledgeReasoningModule(module),
        budget,
        candidate_limit=16,
        argument_limit=16,
    )

    asyncio.run(node.run({'understanding': 'Understand', 'context': context}))

    request = module.requests[0]
    assert request.context.propositions == [conclusion, premise]
    assert request.context.arguments == []


def test_connected_argument_and_endpoints_take_priority_over_unrelated_claims() -> (
    None
):
    matched = Proposition(uuid='p1', name='Matched', statement='Matched claim')
    unrelated = Proposition(uuid='p2', name='Other', statement='Other claim')
    premise = Proposition(uuid='p3', name='Premise', statement='Premise claim')
    argument = Argument(
        uuid='a1',
        premise_uuids=['p3'],
        conclusion_uuid='p1',
    )
    module = _CapturingModule()

    _run(
        _reasoner(module, candidate_limit=2),
        _context([matched, unrelated, premise], arguments=[argument]),
    )
    request = module.requests[0]
    assert [item.uuid for item in request.context.propositions] == ['p1', 'p3']
    assert request.context.arguments == [argument]


def test_oversized_matched_claim_raises_instead_of_returning_no_knowledge() -> (
    None
):
    proposition = Proposition(uuid='p1', name='Claim', statement='A claim')
    reasoner = KnowledgeReasoningNode(
        KnowledgeReasoningModule(_CapturingModule()),
        TokenBudget(MaterialTokenEstimator(_Counter()), 1),
        candidate_limit=16,
        argument_limit=16,
    )
    with pytest.raises(InputBudgetExceeded):
        _run(reasoner, _context([proposition]), 'Understand')


def test_dspy_prediction_returns_new_and_existing_references() -> None:
    existing = Proposition(
        uuid='existing', name='Slope', statement='Slope is rise over run.'
    )
    premise = Proposition(
        uuid='premise', name='Rise', statement='Rise is vertical change.'
    )
    argument = Argument(
        uuid='existing-argument',
        premise_uuids=[premise.uuid],
        conclusion_uuid=existing.uuid,
    )
    response = {
        'new_propositions': [
            {
                'id': 'new-slope',
                'name': 'Derivative',
                'statement': 'A derivative gives an instantaneous slope.',
            }
        ],
        'new_arguments': [
            {
                'premises': [{'kind': 'existing', 'uuid': 'existing'}],
                'conclusion': {'kind': 'new', 'id': 'new-slope'},
            }
        ],
    }
    backend = dspy.Predict(generation_signature(KnowledgeReasoningSignature))
    lm = DummyLM([{'candidates': response}])
    backend.set_lm(lm)
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100000)
    predictor = NativePredictor(
        backend,
        KnowledgeReasoningSignature,
        budget=budget,
        image_budget=None,
        supports_images=False,
    )
    module = KnowledgeReasoningModule(predictor)

    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        result = _run(
            KnowledgeReasoningNode(
                module,
                budget,
                candidate_limit=16,
                argument_limit=16,
            ),
            _context([existing, premise], arguments=[argument]),
            'Understand instantaneous slope',
        )['pending']
    assert result.new_propositions == [
        NewProposition(
            id='new-slope',
            name='Derivative',
            statement='A derivative gives an instantaneous slope.',
        )
    ]
    assert result.new_arguments == [
        NewArgument(
            premises=[{'kind': 'existing', 'uuid': 'existing'}],
            conclusion={'kind': 'new', 'id': 'new-slope'},
        )
    ]
    [entry] = lm.history
    message = [item for item in entry['messages'] if item['role'] == 'user'][-1]
    assert 'existing-argument' in str(message['content'])
    assert '"origin"' not in str(message['content'])


def test_reasoning_graph_returns_original_search_context_and_canonical_storage():
    premise = Proposition(
        uuid='premise', name='Premise', statement='A stored premise.'
    )
    retrieved = _context([premise])
    conclusion = Proposition(
        uuid='canonical-conclusion',
        name='Conclusion',
        statement='A new conclusion.',
    )
    argument = Argument(
        uuid='stored-argument',
        premise_uuids=[premise.uuid],
        conclusion_uuid=conclusion.uuid,
    )

    class _Search:
        async def search(self, user_uuid, query):
            assert user_uuid == 'owner'
            return retrieved

    class _Predictor(dspy.Module):
        async def aforward(self, *, request):
            return dspy.Prediction(
                candidates=ReasoningCandidates(
                    new_propositions=[
                        NewProposition(
                            id='local-conclusion',
                            name='Conclusion',
                            statement='A new conclusion.',
                        )
                    ],
                    new_arguments=[
                        NewArgument(
                            premises=[
                                ExistingPropositionReference(uuid=premise.uuid)
                            ],
                            conclusion=NewPropositionReference(
                                id='local-conclusion'
                            ),
                        )
                    ],
                )
            )

    class _Committer:
        async def commit(self, user_uuid, candidates):
            assert user_uuid == 'owner'
            assert candidates.new_propositions[0].id == 'local-conclusion'
            return StoredKnowledge(
                propositions=[conclusion], arguments=[argument]
            )

    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100000)
    reasoning = KnowledgeReasoningNode(
        KnowledgeReasoningModule(_Predictor()),
        budget,
        candidate_limit=16,
        argument_limit=16,
    )
    graph = KnowledgeReasoningGraph(
        KnowledgeSearchNode(_Search()),
        reasoning,
        KnowledgeCommitNode(_Committer()),
    )

    result = asyncio.run(graph.reason('owner', 'Understand the conclusion'))

    assert result.context == retrieved
    assert [item.uuid for item in result.context.propositions] == ['premise']
    assert [item.uuid for item in result.stored.propositions] == [
        'canonical-conclusion'
    ]
    [stored_argument] = result.stored.arguments
    assert stored_argument.premise_uuids == ['premise']
    assert stored_argument.conclusion_uuid == 'canonical-conclusion'
