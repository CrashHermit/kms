import asyncio
import logging
from types import SimpleNamespace

from kms.core import clustering, reranker


def _record(
    name: str,
    embedding: list[float],
    description: str = '',
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, embedding=embedding, description=description
    )


class _FakeReranker:
    """Deterministic reranker stand-in.

    Returns results in ``index_order`` (filtered to the document count) by
    default, or the reversed document order when ``reverse`` is set. Every call
    is recorded so tests can assert the query, documents, and ``top_n`` the
    caller passed.
    """

    def __init__(self, index_order=None, reverse=False):
        self.index_order = index_order
        self.reverse = reverse
        self.calls = []

    async def rerank(self, query, documents, top_n=None):
        self.calls.append((query, list(documents), top_n))
        documents = list(documents)
        count = len(documents)
        if self.reverse:
            order = list(range(count - 1, -1, -1))
        else:
            order = [i for i in self.index_order if i < count]
        return [{'index': i, 'relevance_score': 1.0} for i in order]


def _patch_reranker(monkeypatch, configured, fake):
    """Wires the shared reranker module to a fake client or a disabled flag."""
    if configured:
        monkeypatch.setattr(reranker, 'is_configured', lambda: True)
        monkeypatch.setattr(reranker, 'reranker', lambda: fake)
    else:
        monkeypatch.setattr(reranker, 'is_configured', lambda: False)


def test_embedding_thresholds_skip_boundary_judgment():
    calls = []

    async def adjudicate(left, right):
        calls.append((left.name, right.name))
        return True

    records = [
        _record('pivot', [1.0, 0.0]),
        _record('merge', [0.99, 0.01]),
        _record('separate', [-1.0, 0.0]),
    ]
    groups = asyncio.run(
        clustering.adjudicated_groups(
            records,
            recall_threshold=-1.0,
            merge_above=0.9,
            separate_below=0.1,
            adjudicate=adjudicate,
            max_concurrency=2,
        )
    )

    assert calls == []
    assert {record.name for record in groups[0]} == {'pivot', 'merge'}
    assert {record.name for record in groups[1]} == {'separate'}


def test_boundary_judgments_for_one_pivot_run_concurrently():
    active = 0
    maximum_active = 0

    async def adjudicate(_left, right):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return right.name == 'merge'

    records = [
        _record('pivot', [1.0, 0.0]),
        _record('merge', [0.8, 0.6]),
        _record('separate', [0.8, -0.6]),
    ]
    groups = asyncio.run(
        clustering.adjudicated_groups(
            records,
            recall_threshold=0.0,
            merge_above=0.95,
            separate_below=0.0,
            adjudicate=adjudicate,
            max_concurrency=2,
        )
    )

    assert maximum_active == 2
    assert {record.name for record in groups[0]} == {'pivot', 'merge'}
    assert {record.name for record in groups[1]} == {'separate'}


def test_select_reranked_candidates_returns_only_selected(monkeypatch):
    # The reranker returns only two of four indices; the helper must return
    # only those two in reranker relevance order and drop the rest entirely.
    fake = _FakeReranker(index_order=[3, 1])
    _patch_reranker(monkeypatch, configured=True, fake=fake)

    selected = asyncio.run(
        clustering.select_reranked_candidates(
            'query', ['a', 'b', 'c', 'd'], str, top_n=2
        )
    )

    assert selected == ['d', 'b']
    query, documents, top_n = fake.calls[0]
    assert query == 'query'
    assert documents == ['a', 'b', 'c', 'd']
    assert top_n == 2


def test_select_reranked_candidates_unchanged_when_disabled(monkeypatch):
    fake = _FakeReranker(index_order=[3, 1])
    _patch_reranker(monkeypatch, configured=False, fake=fake)

    selected = asyncio.run(
        clustering.select_reranked_candidates(
            'query', ['a', 'b', 'c', 'd'], str, top_n=2
        )
    )

    # An unconfigured reranker must be a transparent no-op: no client use and
    # the original order preserved with all candidates retained.
    assert selected == ['a', 'b', 'c', 'd']
    assert fake.calls == []


def test_select_reranked_candidates_empty_input_skips_client(monkeypatch):
    fake = _FakeReranker(index_order=[0])
    _patch_reranker(monkeypatch, configured=True, fake=fake)

    selected = asyncio.run(
        clustering.select_reranked_candidates('query', [], str, top_n=2)
    )

    assert selected == []
    assert fake.calls == []


def test_adjudicated_groups_adjudicates_in_reranked_order(monkeypatch):
    # Pivot P is the most central mention; Alpha, Beta, and Gamma all fall in the
    # boundary cosine band, so every candidate reaches the adjudicator. The fake
    # reranker reverses the candidate order, which differs from the raw order,
    # and adjudication must follow that reranked order. Each record's group
    # membership then follows the judge's yes/no response.
    pivot = _record('P', [1.0, 0.0], 'P')
    alpha = _record('A', [0.5, 0.8660254], 'A')
    beta = _record('B', [0.5, -0.8660254], 'B')
    gamma = _record('C', [0.7071068, 0.7071068], 'C')
    records = [pivot, alpha, beta, gamma]

    adjudicator_calls = []

    async def adjudicate(_pivot, record):
        adjudicator_calls.append(record.name)
        return record.name in {'B', 'C'}

    fake = _FakeReranker(reverse=True)
    _patch_reranker(monkeypatch, configured=True, fake=fake)

    groups = asyncio.run(
        clustering.adjudicated_groups(
            records,
            recall_threshold=-1.0,
            merge_above=0.9,
            separate_below=0.2,
            adjudicate=adjudicate,
            max_concurrency=8,
            rerank_top_n=3,
        )
    )

    input_documents = fake.calls[0][1]
    # The reranker reversed the raw candidate order, so the adjudicator must be
    # invoked in that reranked order rather than the raw order.
    assert input_documents != list(reversed(input_documents))
    assert adjudicator_calls == list(reversed(input_documents))
    assert {*adjudicator_calls} == {'A', 'B', 'C'}
    # Membership follows the judge: Beta and Gamma merge with the pivot, Alpha
    # is separated despite adjacency.
    assert {record.name for record in groups[0]} == {'P', 'B', 'C'}
    assert {record.name for record in groups[1]} == {'A'}


def test_adjudicated_groups_requeues_omitted_candidates(monkeypatch):
    # Pivot P is the most central record. The reranker selects only one of
    # the two boundary candidates, omitting the other. The selected candidate
    # is adjudicated and merged with P; the omitted candidate is requeued to
    # unassigned and forms its own singleton group in the next pivot pass
    # rather than being silently dropped.
    pivot = _record('P', [1.0, 0.0, 0.0], 'P')
    alpha = _record('A', [0.8, 0.5, 0.0], 'A')
    beta = _record('B', [0.8, 0.0, 0.5], 'B')
    records = [pivot, alpha, beta]

    adjudicator_calls = []

    async def adjudicate(_pivot, record):
        adjudicator_calls.append(record.name)
        return True

    # The reranker selects only index 0 from the candidates that
    # coarse_groups presents, omitting the remaining candidate.
    fake = _FakeReranker(index_order=[0])
    _patch_reranker(monkeypatch, configured=True, fake=fake)

    groups = asyncio.run(
        clustering.adjudicated_groups(
            records,
            recall_threshold=-1.0,
            merge_above=0.9,
            separate_below=0.1,
            adjudicate=adjudicate,
            max_concurrency=8,
            rerank_top_n=1,
        )
    )

    # The reranker's top_n is forwarded to the client.
    assert fake.calls[0][2] == 1
    # Only one candidate reaches the adjudicator; the omitted one does not.
    assert len(adjudicator_calls) == 1
    # Two groups: pivot + selected, and the omitted singleton.
    assert len(groups) == 2
    # All records survive across the groups (no data loss from pruning).
    all_names = {record.name for group in groups for record in group}
    assert all_names == {'P', 'A', 'B'}
    # The selected candidate is grouped with the pivot.
    selected = adjudicator_calls[0]
    pivot_group = next(group for group in groups if pivot in group)
    assert {record.name for record in pivot_group} == {'P', selected}
    # The omitted candidate forms its own singleton.
    omitted = {'A', 'B'} - {selected}
    singleton = next(group for group in groups if len(group) == 1)
    assert {record.name for record in singleton} == omitted


def test_adjudicated_groups_logs_embedding_comparisons(caplog):
    async def adjudicate(_left, _right):
        return True

    records = [
        _record('pivot', [1.0, 0.0], 'pivot description'),
        _record('merge', [0.99, 0.01], 'merge description'),
        _record('separate', [-1.0, 0.0], 'separate description'),
    ]
    with caplog.at_level(logging.DEBUG, logger='kms.core.clustering'):
        asyncio.run(
            clustering.adjudicated_groups(
                records,
                recall_threshold=-1.0,
                merge_above=0.9,
                separate_below=0.1,
                adjudicate=adjudicate,
                max_concurrency=2,
            )
        )

    messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == 'kms.core.clustering'
    ]
    embedding_logs = [m for m in messages if 'clustering embedding:' in m]
    # Two candidates compared against the pivot (one auto-merged, one separated).
    assert len(embedding_logs) >= 2
    assert any('classification=merge' in m for m in embedding_logs)
    assert any('classification=separate' in m for m in embedding_logs)
    assert any('cosine=' in m for m in embedding_logs)


def test_adjudicated_groups_logs_adjudication_decisions(caplog):
    async def adjudicate(_left, right):
        return right.name == 'merge'

    records = [
        _record('pivot', [1.0, 0.0], 'pivot'),
        _record('merge', [0.8, 0.6], 'merge'),
        _record('separate', [0.8, -0.6], 'separate'),
    ]
    with caplog.at_level(logging.DEBUG, logger='kms.core.clustering'):
        asyncio.run(
            clustering.adjudicated_groups(
                records,
                recall_threshold=0.0,
                merge_above=0.95,
                separate_below=0.0,
                adjudicate=adjudicate,
                max_concurrency=2,
            )
        )

    messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == 'kms.core.clustering'
    ]
    adjudication_logs = [m for m in messages if 'clustering adjudication:' in m]
    assert len(adjudication_logs) >= 2
    assert any('decision=merge' in m for m in adjudication_logs)
    assert any('decision=separate' in m for m in adjudication_logs)
    assert any('pivot=pivot' in m for m in adjudication_logs)
    assert any('candidate=merge' in m for m in adjudication_logs)
    assert any('candidate=separate' in m for m in adjudication_logs)
