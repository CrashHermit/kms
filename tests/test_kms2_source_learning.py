import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from kms2 import application, tui
from kms2.config.settings import Settings
from kms2.core.model.source import Source
from kms2.core.model.source_learning.atomic import (
    SourceAtomicFlashcardInput,
    SourceAtomicFlashcardRequest,
)
from kms2.core.model.source_learning.coherent import (
    SourceCoherentFlashcardCandidate,
)
from kms2.core.model.source_learning.context import (
    SourceLearningEvidence,
    SourceLearningHubContext,
)
from kms2.core.model.source_learning.flashcard import (
    SourceFlashcardCandidate,
    SourceFlashcardOccurrence,
)
from kms2.core.model.source_semantic.source_triplet_hub import (
    SourceTripletHubRole,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_learning.graph import SourceLearningGraph
from kms2.langgraph.source_learning.state import (
    SourceLearningState,
)
from kms2.node.source_learning.atomic import SourceAtomicFlashcardNode
from kms2.node.source_learning.atomic_persistence import (
    SourceAtomicFlashcardPersistenceNode,
)
from kms2.node.source_learning.coherent import SourceCoherentFlashcardNode
from kms2.node.source_learning.coherent_persistence import (
    SourceCoherentFlashcardPersistenceNode,
)


def test_source_learning_application_rejects_unowned_source(monkeypatch):
    async def list_sources(settings, user_uuid):
        return []

    monkeypatch.setattr(application, 'list_sources', list_sources)

    with pytest.raises(ValueError, match='not owned by user'):
        asyncio.run(
            application.run_source_learning_stage(
                Settings(), 'user-1', 'foreign-source'
            )
        )


def test_tui_exposes_owned_source_learning_action(monkeypatch):
    from kms2.core.model.user import User

    user = User(uuid='user-1', name='Alex')
    source = Source(uuid='source-1', key='book.pdf')
    prompts = iter([tui.SOURCE_LEARNING_OPTION, source, True])
    selections = []
    calls = []

    class Prompt:
        def execute(self):
            return next(prompts)

    async def list_users(settings):
        return [user]

    async def list_sources(settings, user_uuid):
        return [source]

    async def list_unowned_sources(settings):
        return []

    async def run_learning(settings, user_uuid, source_uuid):
        calls.append((user_uuid, source_uuid))
        return SimpleNamespace(
            atomic_flashcard_count=2,
            coherent_flashcard_count=2,
        )

    monkeypatch.setattr(tui, 'list_users', list_users)
    monkeypatch.setattr(tui, 'list_sources', list_sources)
    monkeypatch.setattr(tui, 'list_unowned_sources', list_unowned_sources)
    monkeypatch.setattr(tui, 'run_source_learning_stage', run_learning)
    monkeypatch.setattr(
        tui,
        'inquirer',
        SimpleNamespace(
            select=lambda **kwargs: selections.append(kwargs) or Prompt(),
            confirm=lambda **kwargs: Prompt(),
        ),
    )

    tui._run_tui()

    assert [choice for choice in selections[0]['choices']] == [
        tui.NEW_SOURCE_OPTION,
        tui.CREATE_DECK_OPTION,
        tui.ADD_CARD_OPTION,
        tui.REMOVE_CARD_OPTION,
        tui.REVIEW_CARDS_OPTION,
        tui.REVIEW_HISTORY_OPTION,
        tui.EXISTING_SOURCE_OPTION,
        tui.SOURCE_LEARNING_OPTION,
        tui.GLOBAL_SEMANTIC_OPTION,
    ]
    assert calls == [('user-1', 'source-1')]


def test_source_learning_application_runs_owned_source(monkeypatch):
    source = Source(uuid='source-1', key='book.pdf')
    resources = SimpleNamespace(
        local_models=object(),
        tokenizers=object(),
        database=object(),
        recorder=None,
    )
    calls = []

    async def list_sources(settings, user_uuid):
        return [source]

    class Graph:
        async def ainvoke(self, initial_state):
            calls.append(initial_state)
            return {
                'atomic_flashcard_count': 2,
                'coherent_flashcard_count': 4,
            }

    @asynccontextmanager
    async def application_resources(settings):
        yield resources

    monkeypatch.setattr(application, 'list_sources', list_sources)
    monkeypatch.setattr(
        application, '_application_resources', application_resources
    )
    monkeypatch.setattr(
        application,
        'build_source_learning_graph',
        lambda *args, **kwargs: SimpleNamespace(build_graph=lambda: Graph()),
    )

    result = asyncio.run(
        application.run_source_learning_stage(Settings(), 'user-1', source.uuid)
    )

    assert result.atomic_flashcard_count == 2
    assert result.coherent_flashcard_count == 4
    assert calls == [{'source_uuid': source.uuid}]


class _Counter:
    def count_texts(self, texts):
        return [1 for _ in texts]


class _Repository:
    def __init__(self, requests):
        self.requests = requests

    async def load_atomic_flashcard_requests(self, source_uuid):
        return self.requests


class _AtomicModule:
    async def acall(self, *, request: SourceAtomicFlashcardInput):
        return [
            SourceFlashcardCandidate(
                question=f'Question: {request.evidence.subject}',
                answer=request.evidence.source_fact_text,
            )
        ]


class _CoherentModule:
    def __init__(self):
        self.inputs = []

    async def acall(self, *, request):
        self.inputs.append(request)
        return [
            SourceCoherentFlashcardCandidate(
                question='How are the claims connected?',
                answer='They share the supplied source context.',
                card_indexes=[1, 2],
            )
        ]


def _request(triplet_uuid, source_fact_uuid):
    context = SourceLearningHubContext(
        triplet_hub=SourceTripletHubRole(name='Velocity', description='Rate'),
        subject_hub=SourceTripletHubRole(name='Position', description='s(t)'),
        predicate_hub=SourceTripletHubRole(
            name='Has rate', description='relates'
        ),
        object_hub=SourceTripletHubRole(
            name='Average velocity', description='g(x)'
        ),
    )
    return SourceAtomicFlashcardRequest(
        hub_uuid='hub-1',
        triplet_uuid=triplet_uuid,
        source_fact_uuid=source_fact_uuid,
        input=SourceAtomicFlashcardInput(
            context=context,
            evidence=SourceLearningEvidence(
                subject='s(t)',
                predicate='has average velocity',
                object='g(x)',
                source_fact_text='The source defines the average velocity.',
            ),
        ),
    )


def test_two_pass_generation_retains_context_and_uses_local_indexes():
    requests = [
        _request('triplet-1', 'fact-1'),
        _request('triplet-2', 'fact-2'),
    ]
    repository = _Repository(requests)
    budget = TokenBudget(counter=_Counter(), token_limit=100)
    atomic = SourceAtomicFlashcardNode(
        repository,
        _AtomicModule(),
        budget=budget,
    )
    state = SourceLearningState(
        source_uuid='source-1', atomic_requests=requests
    )
    sends = atomic.dispatch(state)
    assert len(sends) == 2
    first = asyncio.run(atomic.worker(sends[0].arg))
    second = asyncio.run(atomic.worker(sends[1].arg))
    atomic_state = state.model_copy(
        update={
            'atomic_results': [
                *second['atomic_results'],
                *first['atomic_results'],
            ]
        }
    )
    packet_state = atomic.collect(atomic_state)
    coherent_module = _CoherentModule()
    coherent = SourceCoherentFlashcardNode(
        coherent_module,
        budget=budget,
    )
    coherent_state = atomic_state.model_copy(update=packet_state)
    prepared = coherent.prepare(coherent_state)
    request = prepared['coherent_requests'][0]
    model_input = request.model_input()
    serialized = model_input.model_dump_json()
    assert 'hub-1' not in serialized
    assert 'triplet-1' not in serialized
    assert [card.index for card in model_input.packets[0].cards] == [1]
    assert [card.index for card in model_input.packets[1].cards] == [2]
    coherent_state = coherent_state.model_copy(update=prepared)
    result = asyncio.run(coherent.worker({'ordinal': 0, 'request': request}))
    collected = coherent.collect(
        coherent_state.model_copy(
            update={'coherent_results': result['coherent_results']}
        )
    )
    occurrence: SourceFlashcardOccurrence = collected['coherent_occurrences'][0]
    assert occurrence.triplet_uuids == ['triplet-1', 'triplet-2']
    assert occurrence.source_fact_uuids == ['fact-1', 'fact-2']
    assert len(occurrence.derived_card_uuids) == 2
    assert coherent_module.inputs[0].packets[0].evidence.source_fact_text


@pytest.mark.parametrize('has_evidence', [True, False])
def test_learning_graph_persists_selected_provenance_after_atomic_parents(
    has_evidence,
):
    requests = (
        [_request('triplet-1', 'fact-1'), _request('triplet-2', 'fact-2')]
        if has_evidence
        else []
    )

    class Repository(_Repository):
        def __init__(self):
            super().__init__(requests)
            self.cards = {'old-card': None}

        async def clear_source_learning(self, source_uuid):
            self.cards.clear()

        async def persist_flashcards(self, source_uuid, occurrences):
            for occurrence in occurrences:
                for parent_uuid in occurrence.derived_card_uuids:
                    parent = self.cards[parent_uuid]
                    assert parent.hub_uuid == occurrence.hub_uuid
                self.cards[occurrence.card.uuid] = occurrence
            return len(occurrences)

    repository = Repository()
    budget = TokenBudget(counter=_Counter(), token_limit=100)
    graph = SourceLearningGraph(
        repository,
        SourceAtomicFlashcardNode(repository, _AtomicModule(), budget=budget),
        SourceAtomicFlashcardPersistenceNode(repository),
        SourceCoherentFlashcardNode(_CoherentModule(), budget=budget),
        SourceCoherentFlashcardPersistenceNode(repository),
    ).build_graph()

    final_state = asyncio.run(graph.ainvoke({'source_uuid': 'source-1'}))

    assert 'old-card' not in repository.cards
    assert final_state['atomic_flashcard_count'] == (2 if has_evidence else 0)
    assert final_state['coherent_flashcard_count'] == (1 if has_evidence else 0)
    if not has_evidence:
        assert repository.cards == {}
        return

    coherent = final_state['coherent_occurrences'][0]
    atomic = [
        occurrence
        for packet in final_state['atomic_packets']
        for occurrence in packet.cards
    ]
    assert set(repository.cards) == {
        coherent.card.uuid,
        *(occurrence.card.uuid for occurrence in atomic),
    }
    assert coherent.derived_card_uuids == [
        occurrence.card.uuid for occurrence in atomic
    ]
    assert coherent.triplet_uuids == ['triplet-1', 'triplet-2']
    assert coherent.source_fact_uuids == ['fact-1', 'fact-2']
