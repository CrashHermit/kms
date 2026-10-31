"""Behavioral contracts for conservative proposition hub grouping."""

import asyncio
from types import SimpleNamespace

from kms2.core.model.source_semantic.source_proposition_hub import (
    SourcePropositionHubCandidate,
    SourcePropositionHubJudgeDecision,
    SourcePropositionHubMember,
)
from kms2.node.source_semantic.source_proposition_hub import (
    SourcePropositionHubNode,
    _complete_link_groups,
)


class _Counter:
    def count_texts(self, texts):
        return [1 for _ in texts]


class _Reranker:
    async def rerank(self, query, documents, top_n=None):
        assert top_n is None
        return [
            {'index': index, 'relevance_score': score}
            for index, score in enumerate((0.99, 0.60, 0.59))
        ]


def test_every_reranker_survivor_is_dispatched_to_judgment():
    candidates = [
        _candidate('a', 'b', 'Claim A', 'Claim B', 0.95),
        _candidate('a', 'c', 'Claim A', 'Claim C', 0.94),
        _candidate('a', 'd', 'Claim A', 'Claim D', 0.93),
    ]
    node = SourcePropositionHubNode(
        None,
        None,
        _Reranker(),
        SimpleNamespace(reranker_rejection_threshold=0.60, judge_batch_size=10),
        reranker_budget=SimpleNamespace(counter=_Counter(), token_limit=20),
        judge_budget=SimpleNamespace(counter=_Counter(), token_limit=20),
    )

    result = asyncio.run(
        node.rerank_worker(
            {
                'source_proposition_hub_rerank_ordinal': 0,
                'source_proposition_hub_left_text': 'Claim A',
                'source_proposition_hub_batch': candidates,
            }
        )
    )
    survivors = result['source_proposition_hub_rerank_results'][0].candidates
    sends = node.dispatch_judge(
        SimpleNamespace(source_proposition_hub_reranked_candidates=survivors)
    )

    assert [candidate.right_uuid for candidate in survivors] == ['b', 'c']
    assert [
        candidate.right_uuid
        for send in sends
        for candidate in send.arg['source_proposition_hub_batch']
    ] == ['b', 'c']

    class _RejectJudge:
        async def acall(self, *, requests):
            return [
                SourcePropositionHubJudgeDecision(
                    index=request.index, equivalent=False
                )
                for request in requests
            ]

    node._judge_module = _RejectJudge()
    judged = asyncio.run(node.judge_worker(sends[0].arg))
    accepted = judged['source_proposition_hub_judge_results'][0].accepted
    groups = _complete_link_groups(
        [
            _member('a', 'Claim A'),
            _member('b', 'Claim B'),
        ],
        accepted,
    )
    assert [[member.uuid for member in group.members] for group in groups] == [
        ['a'],
        ['b'],
    ]


def _member(uuid: str, text: str) -> SourcePropositionHubMember:
    return SourcePropositionHubMember(uuid=uuid, text=text)


def _candidate(
    left_uuid: str,
    right_uuid: str,
    left_text: str,
    right_text: str,
    score: float,
) -> SourcePropositionHubCandidate:
    return SourcePropositionHubCandidate(
        left_uuid=left_uuid,
        right_uuid=right_uuid,
        left_text=left_text,
        right_text=right_text,
        score=score,
    )


def test_positive_paraphrase_and_alpha_rename_group_in_source_order():
    members = [
        _member('claim-b', 'For every real x, x² ≥ 0.'),
        _member('claim-a', 'For each real t, t squared is nonnegative.'),
        _member('claim-c', 'Every real number has nonnegative square.'),
        _member('unrelated', 'Every integer is rational.'),
    ]
    groups = _complete_link_groups(
        members,
        [
            _candidate(
                'claim-a',
                'claim-b',
                members[1].text,
                members[0].text,
                0.94,
            ),
            _candidate(
                'claim-a',
                'claim-c',
                members[1].text,
                members[2].text,
                0.89,
            ),
            _candidate(
                'claim-b',
                'claim-c',
                members[0].text,
                members[2].text,
                0.91,
            ),
        ],
    )

    assert [[member.uuid for member in group.members] for group in groups] == [
        ['claim-b', 'claim-a', 'claim-c'],
        ['unrelated'],
    ]


def test_missing_or_rejected_cross_pair_prevents_similarity_bridge_merge():
    members = [
        _member('a', 'Claim A'),
        _member('b', 'Claim B'),
        _member('c', 'Claim C'),
    ]
    groups = _complete_link_groups(
        members,
        [
            _candidate('a', 'b', 'Claim A', 'Claim B', 0.99),
            _candidate('b', 'c', 'Claim B', 'Claim C', 0.98),
        ],
    )

    assert [[member.uuid for member in group.members] for group in groups] == [
        ['a', 'b'],
        ['c'],
    ]


def test_high_reranker_score_does_not_merge_without_positive_judgment():
    members = [
        _member('early', 'Original claim'),
        _member('later', 'Related claim'),
    ]
    groups = _complete_link_groups(members, [])

    assert [[member.uuid for member in group.members] for group in groups] == [
        ['early'],
        ['later'],
    ]


def test_all_occurrences_survive_as_ordered_singletons():
    members = [
        _member('third', 'Third original'),
        _member('first', 'First original'),
        _member('second', 'Second original'),
    ]
    groups = _complete_link_groups(members, [])

    assert [(group.ordinal, group.members[0].uuid) for group in groups] == [
        (0, 'third'),
        (1, 'first'),
        (2, 'second'),
    ]
