"""Behavioral tests for owner-scoped knowledge retrieval."""

import asyncio

from kms.knowledge.models import Argument, KnowledgeContext, Proposition
from kms.knowledge.reasoning.settings import ReasoningSettings
from kms.knowledge.retrieval.search import KnowledgeSearch
from kms.knowledge.retrieval.settings import KnowledgeSearchSettings
from kms.media.models import Material
from kms.retrieval.selection.models import SearchMaterialKind


class _Embedder:
    def __init__(self) -> None:
        self.calls = []

    async def embed(self, request):
        self.calls.append(request)
        return [[1.0, 0.0]]


class _Repository:
    def __init__(self, candidates: list[Proposition]) -> None:
        self.candidates = candidates
        self.search_calls = []
        self.context_calls = []

    async def search(self, user_uuid, embedding, *, limit, embedding_model):
        self.search_calls.append((user_uuid, embedding, limit, embedding_model))
        return self.candidates

    async def context(self, user_uuid, matched, *, argument_limit):
        self.context_calls.append((user_uuid, matched, argument_limit))
        selected = {item.uuid for item in matched}
        arguments = (
            [
                Argument(
                    uuid='argument',
                    premise_uuids=['p1'],
                    conclusion_uuid='p2',
                )
            ]
            if 'p2' in selected
            else []
        )
        endpoint_uuids = {
            uuid
            for argument in arguments
            for uuid in [*argument.premise_uuids, argument.conclusion_uuid]
        }
        by_uuid = {item.uuid: item for item in self.candidates}
        propositions = [
            by_uuid[uuid]
            for uuid in by_uuid
            if uuid in selected or uuid in endpoint_uuids
        ]
        return KnowledgeContext(
            matched=matched,
            propositions=propositions,
            arguments=arguments,
        )


class _Selector:
    def __init__(self, positions: list[int]) -> None:
        self.positions = positions
        self.calls = []

    async def select(self, *, query, material_kind, candidates):
        self.calls.append((query, material_kind, candidates))
        return self.positions


def _propositions() -> list[Proposition]:
    return [
        Proposition(
            uuid=f'p{index}',
            name=f'Claim {index}',
            statement=f'Statement {index}',
        )
        for index in range(3)
    ]


def test_search_uses_independent_limits_and_loads_complete_connected_arguments():
    candidates = _propositions()
    repository = _Repository(candidates)
    embedder = _Embedder()
    selector = _Selector([2, 0])
    search = KnowledgeSearch(
        repository,
        embedder,
        'embedding-model',
        selection=selector,
        settings=KnowledgeSearchSettings(
            candidate_limit=3, result_limit=1, argument_limit=1
        ),
    )
    reasoning_settings = ReasoningSettings(candidate_limit=1, argument_limit=1)

    result = asyncio.run(search.search('owner', Material.from_text('query')))

    assert reasoning_settings.candidate_limit == 1
    assert reasoning_settings.argument_limit == 1
    assert repository.search_calls == [
        ('owner', [1.0, 0.0], 3, 'embedding-model')
    ]
    assert repository.context_calls == [('owner', [candidates[2]], 1)]
    assert selector.calls[0][1] is SearchMaterialKind.PROPOSITION
    assert [candidate.position for candidate in selector.calls[0][2]] == [
        0,
        1,
        2,
    ]
    assert result.matched == [candidates[2]]
    assert [item.uuid for item in result.propositions] == ['p1', 'p2']
    [argument] = result.arguments
    assert argument.uuid == 'argument'
    assert argument.premise_uuids == ['p1']
    assert argument.conclusion_uuid == 'p2'

    expanded = KnowledgeSearch(
        repository,
        embedder,
        'embedding-model',
        selection=selector,
        settings=KnowledgeSearchSettings(
            candidate_limit=3, result_limit=2, argument_limit=1
        ),
    )
    expanded_result = asyncio.run(
        expanded.search('owner', Material.from_text('query'))
    )
    assert [item.uuid for item in expanded_result.matched] == ['p2', 'p0']
    assert repository.context_calls[-1] == (
        'owner',
        [candidates[2], candidates[0]],
        1,
    )


def test_empty_retrieval_and_selection_skip_context_loading():
    candidates = _propositions()
    settings = KnowledgeSearchSettings(
        candidate_limit=3, result_limit=1, argument_limit=1
    )
    empty_repository = _Repository([])
    empty_selector = _Selector([0])
    search = KnowledgeSearch(
        empty_repository,
        _Embedder(),
        'embedding-model',
        selection=empty_selector,
        settings=settings,
    )
    empty = asyncio.run(search.search('owner', Material.from_text('query')))
    assert empty.matched == []
    assert empty.propositions == []
    assert empty.arguments == []
    assert empty_selector.calls == []
    assert empty_repository.context_calls == []

    selected_repository = _Repository(candidates)
    no_selection = _Selector([])
    search = KnowledgeSearch(
        selected_repository,
        _Embedder(),
        'embedding-model',
        selection=no_selection,
        settings=settings,
    )
    empty = asyncio.run(search.search('owner', Material.from_text('query')))
    assert empty.matched == []
    assert empty.propositions == []
    assert empty.arguments == []
    assert len(no_selection.calls) == 1
    assert selected_repository.context_calls == []
