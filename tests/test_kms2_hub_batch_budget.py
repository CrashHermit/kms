"""Complete-request judge budgets and ordered community-hub collection."""

import asyncio
from importlib import import_module

import pytest
from kms2_token_helpers import direct_synthesis
from langgraph.graph import END, START, StateGraph

from kms2.core.windowing import InputBudgetExceeded, TokenBudget
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.langgraph.source_semantic.state import SourceSemanticState

CASES = [
    ('source', 'Entity'),
    ('source', 'Event'),
    ('source', 'Predicate'),
    ('source', 'Statement'),
    ('source', 'Procedure'),
    ('global', 'Entity'),
    ('global', 'Event'),
    ('global', 'Predicate'),
    ('global', 'Statement'),
    ('global', 'Procedure'),
]


class _Counter:
    def __init__(self, cost):
        self.cost = cost

    def count_texts(self, texts):
        return [self.cost] * len(texts)


class _Reranker:
    def __init__(self):
        self.documents = []

    async def rerank(self, query, documents, top_n=None):
        self.documents.append(documents)
        await asyncio.sleep(0.01 if len(documents) == 2 else 0)
        return [
            {'index': index, 'relevance_score': 0.7}
            for index in range(len(documents))
        ]


class _Judge:
    def __init__(self, decision):
        self.decision = decision
        self.requests = []

    async def acall(self, *, requests):
        self.requests.append(requests)
        await asyncio.sleep(0.01 if len(requests) == 2 else 0)
        return [
            self.decision(index=index, belongs_in_same_hub=True)
            for index in range(len(requests))
        ]


def _fixture(scope, kind, *, threshold=8, batch_size=10):
    stem = f'{scope.title()}{kind}Hub'
    basename = f'{scope}_{kind.lower()}_hub'
    models = import_module(f'kms2.core.model.{scope}_semantic.{basename}')
    nodes = import_module(f'kms2.node.{scope}_semantic.{basename}')
    config = import_module(f'kms2.config.{scope}_semantic')
    candidate_type = getattr(models, f'{stem}Candidate')
    candidates = []
    for index in range(3):
        values = {
            field: f'{field}-{index}'
            for field in candidate_type.model_fields
            if field != 'score'
        }
        values.update(left_uuid='left', right_uuid=f'right-{index}', score=0.9)
        candidates.append(candidate_type(**values))
    reranker = _Reranker()
    judge = _Judge(getattr(models, f'{stem}JudgeDecision'))
    budget = TokenBudget(_Counter(4), threshold)
    settings = getattr(config, f'{stem}Settings')(
        reranker_token_budget=8,
        judge_token_budget=threshold,
        judge_batch_size=batch_size,
    )
    node = getattr(nodes, f'{stem}Node')(
        None,
        None,
        judge,
        reranker,
        None,
        settings,
        reranker_token_counter=_Counter(2),
        judge_budget=budget,
        **direct_synthesis(),
    )
    return basename, node, candidates, reranker, judge, budget


@pytest.mark.parametrize(('scope', 'kind'), CASES)
@pytest.mark.parametrize('batch_size', [1, 10])
def test_complete_judge_budgets_preserve_candidate_order_through_collectors(
    scope, kind, batch_size
):
    asyncio.run(_exercise(scope, kind, batch_size))


async def _exercise(scope, kind, batch_size):
    basename, node, candidates, reranker, judge, budget = _fixture(
        scope, kind, batch_size=batch_size
    )
    state_type = (
        SourceSemanticState if scope == 'source' else GlobalSemanticState
    )
    graph = StateGraph(state_type)
    graph.add_node('route', lambda state: {})
    graph.add_node(f'{basename}_rerank_worker', node.rerank_worker)
    graph.add_node(f'{basename}_rerank_collect', node.collect_rerank)
    graph.add_node(f'{basename}_judge_worker', node.judge_worker)
    graph.add_node(f'{basename}_judge_collect', node.collect_judge)
    graph.add_edge(START, 'route')
    graph.add_conditional_edges('route', node.dispatch_rerank)
    graph.add_edge(f'{basename}_rerank_worker', f'{basename}_rerank_collect')
    graph.add_conditional_edges(
        f'{basename}_rerank_collect', node.dispatch_judge
    )
    graph.add_edge(f'{basename}_judge_worker', f'{basename}_judge_collect')
    graph.add_edge(f'{basename}_judge_collect', END)
    result = await graph.compile().ainvoke(
        {'source_uuid': 'source', f'{basename}_candidates': candidates}
        if scope == 'source'
        else {f'{basename}_candidates': candidates}
    )
    assert [len(documents) for documents in reranker.documents] == [2, 1]
    assert [len(requests) for requests in judge.requests] == (
        [1, 1, 1] if batch_size == 1 else [2, 1]
    )
    for requests in judge.requests:
        assert [request.index for request in requests] == list(
            range(len(requests))
        )
        assert 4 * len(requests) <= budget.token_limit
    assert [
        pair.right_uuid for pair in result[f'{basename}_accepted_pairs']
    ] == ['right-0', 'right-1', 'right-2']


@pytest.mark.parametrize(('scope', 'kind'), CASES)
def test_oversized_judge_singleton_is_not_dispatched(scope, kind):
    async def exercise():
        basename, node, candidates, _, judge, _ = _fixture(
            scope, kind, threshold=3
        )
        state_type = (
            SourceSemanticState if scope == 'source' else GlobalSemanticState
        )
        values = {f'{basename}_borderline_pairs': candidates}
        if scope == 'source':
            values['source_uuid'] = 'source'
        state = state_type(**values)
        with pytest.raises(InputBudgetExceeded, match='singleton'):
            await node.dispatch_judge(state)
        assert judge.requests == []

    asyncio.run(exercise())
