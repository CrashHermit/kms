import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from kms2 import application, tui
from kms2.config.settings import Settings
from kms2.core.model.source import Source
from kms2.core.model.source_learning.entity import (
    SourceEntityFlashcardInput,
    SourceEntityFlashcardResult,
    SourceEntityLearningFactEvidence,
    SourceEntityLearningFactInput,
    SourceEntityLearningFactResult,
)
from kms2.core.model.source_learning.event import (
    SourceEventFlashcardInput,
    SourceEventFlashcardResult,
    SourceEventLearningFactEvidence,
    SourceEventLearningFactInput,
    SourceEventLearningFactResult,
)
from kms2.core.model.source_learning.predicate import (
    SourcePredicateFlashcardInput,
    SourcePredicateFlashcardResult,
    SourcePredicateLearningFactEvidence,
    SourcePredicateLearningFactInput,
    SourcePredicateLearningFactResult,
)
from kms2.core.model.source_learning.triplet import (
    SourceTripletFlashcardInput,
    SourceTripletFlashcardResult,
    SourceTripletLearningFactEvidence,
    SourceTripletLearningFactInput,
    SourceTripletLearningFactResult,
)
from kms2.core.model.user import User
from kms2.database.schema import SCHEMA_STATEMENTS
from kms2.database.source_learning.queries import (
    _CLEAR_SOURCE_LEARNING,
    PERSIST_SOURCE_FLASHCARDS,
    READ_SOURCE_ENTITY_LEARNING_INPUTS,
    READ_SOURCE_EVENT_LEARNING_INPUTS,
    READ_SOURCE_PREDICATE_LEARNING_INPUTS,
    READ_SOURCE_TRIPLET_LEARNING_INPUTS,
)
from kms2.database.source_learning.repository import SourceLearningRepository
from kms2.module.source_learning.entity import (
    SourceEntityFlashcardModule,
    SourceEntityLearningFactModule,
)
from kms2.module.source_learning.event import (
    SourceEventFlashcardModule,
    SourceEventLearningFactModule,
)
from kms2.module.source_learning.predicate import (
    SourcePredicateFlashcardModule,
    SourcePredicateLearningFactModule,
)
from kms2.module.source_learning.triplet import (
    SourceTripletFlashcardModule,
    SourceTripletLearningFactModule,
)


class _Predictor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(result=self.result)

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(result=self.result)


def _evidence(model):
    return model(
        source_fact_uuid='fact-1',
        source_fact_text='Alice works for Acme.',
        triplet_uuid='triplet-1',
        subject='Alice',
        predicate='works for',
        object='Acme',
        member_role='subject',
    )


def test_all_typed_modules_forward_sync_and_async():
    cases = (
        (
            SourceEntityLearningFactModule,
            SourceEntityFlashcardModule,
            SourceEntityLearningFactInput,
            SourceEntityLearningFactResult,
            SourceEntityFlashcardInput,
            SourceEntityFlashcardResult,
            SourceEntityLearningFactEvidence,
        ),
        (
            SourceEventLearningFactModule,
            SourceEventFlashcardModule,
            SourceEventLearningFactInput,
            SourceEventLearningFactResult,
            SourceEventFlashcardInput,
            SourceEventFlashcardResult,
            SourceEventLearningFactEvidence,
        ),
        (
            SourcePredicateLearningFactModule,
            SourcePredicateFlashcardModule,
            SourcePredicateLearningFactInput,
            SourcePredicateLearningFactResult,
            SourcePredicateFlashcardInput,
            SourcePredicateFlashcardResult,
            SourcePredicateLearningFactEvidence,
        ),
        (
            SourceTripletLearningFactModule,
            SourceTripletFlashcardModule,
            SourceTripletLearningFactInput,
            SourceTripletLearningFactResult,
            SourceTripletFlashcardInput,
            SourceTripletFlashcardResult,
            SourceTripletLearningFactEvidence,
        ),
    )
    for (
        fact_module,
        card_module,
        input_model,
        fact_result,
        card_input_model,
        card_result,
        evidence_model,
    ) in cases:
        evidence = _evidence(evidence_model)
        request = input_model(
            hub_uuid='hub-1',
            hub_name='Hub',
            hub_description='Description',
            evidence=[evidence],
        )
        facts = fact_result(
            facts=[
                {
                    'text': 'Alice works for Acme.',
                    'source_fact_uuids': ['fact-1'],
                    'triplet_uuids': ['triplet-1'],
                }
            ]
        )
        fact_predictor = _Predictor(facts)
        fact = fact_module(fact_predictor)
        assert fact(request=request) is facts
        assert asyncio.run(fact.aforward(request=request)) is facts
        learning_fact_model = card_input_model.model_fields[
            'learning_fact'
        ].annotation
        learning_fact = learning_fact_model(
            uuid='learning-fact-1',
            text=facts.facts[0].text,
        )
        card_request = card_input_model(
            learning_fact=learning_fact,
            learning_fact_text=learning_fact.text,
            hub_uuid='hub-1',
            hub_name='Hub',
            hub_description='Description',
            evidence=[evidence],
        )
        card = card_result(
            question='What works?', answer='Alice works for Acme.'
        )
        card_predictor = _Predictor(card)
        creator = card_module(card_predictor)
        assert creator(request=card_request) is card
        assert asyncio.run(creator.aforward(request=card_request)) is card
        assert len(fact_predictor.calls) == 2
        assert len(card_predictor.calls) == 2


def test_source_learning_settings_have_eight_independent_profiles():
    settings = Settings().source_learning
    assert list(type(settings).model_fields) == [
        'source_entity_learning_fact',
        'source_entity_flashcard',
        'source_event_learning_fact',
        'source_event_flashcard',
        'source_predicate_learning_fact',
        'source_predicate_flashcard',
        'source_triplet_learning_fact',
        'source_triplet_flashcard',
    ]


def test_source_learning_queries_preserve_directed_context_and_roles():
    for query in (
        READ_SOURCE_ENTITY_LEARNING_INPUTS,
        READ_SOURCE_EVENT_LEARNING_INPUTS,
        READ_SOURCE_PREDICATE_LEARNING_INPUTS,
        READ_SOURCE_TRIPLET_LEARNING_INPUTS,
    ):
        assert 'source_fact_text: fact.text' in query
        assert 'subject: subject.name' in query
        assert 'predicate: predicate.predicate' in query
        assert 'object: object.name' in query
        assert 'member_role' in query
        assert 'WITH DISTINCT' in query


def test_flashcards_use_learning_fact_for_hub_provenance():
    assert (
        'CREATE (card)-[:DERIVED_FROM]->(learning_fact)'
        in PERSIST_SOURCE_FLASHCARDS
    )
    assert 'CREATE (card)-[:ABOUT_HUB]' not in PERSIST_SOURCE_FLASHCARDS


def test_source_learning_uuid_constraints_are_registered():
    schema = '\n'.join(SCHEMA_STATEMENTS)
    for label in (
        'SourceEntityLearningFact',
        'SourceEventLearningFact',
        'SourcePredicateLearningFact',
        'SourceTripletLearningFact',
        'SourceFlashcard',
    ):
        assert (
            f'FOR ({"fact" if label.endswith("Fact") else "card"}:{label}) '
            'REQUIRE '
            f'{"fact" if label.endswith("Fact") else "card"}.uuid IS UNIQUE'
        ) in schema


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
            entity_learning_fact_count=2,
            entity_flashcard_count=2,
            event_learning_fact_count=0,
            event_flashcard_count=0,
            predicate_learning_fact_count=0,
            predicate_flashcard_count=0,
            triplet_learning_fact_count=0,
            triplet_flashcard_count=0,
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
                'entity_learning_fact_count': 2,
                'entity_flashcard_count': 2,
                'event_learning_fact_count': 1,
                'event_flashcard_count': 1,
                'predicate_learning_fact_count': 3,
                'predicate_flashcard_count': 3,
                'triplet_learning_fact_count': 4,
                'triplet_flashcard_count': 4,
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

    assert result.entity_learning_fact_count == 2
    assert result.triplet_flashcard_count == 4
    assert calls == [{'source_uuid': source.uuid}]


class _QueryResult:
    def __init__(self, rows=None, persisted=0):
        self.rows = rows or []
        self.persisted = persisted

    async def data(self):
        return self.rows

    async def single(self):
        return {'persisted': self.persisted}

    async def consume(self):
        return None


class _QuerySession:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def run(self, statement, **parameters):
        self.calls.append((statement, parameters))
        if statement == READ_SOURCE_ENTITY_LEARNING_INPUTS:
            return _QueryResult(self.rows)
        if 'HAS_LEARNING_FACT]->(learning_fact:' in statement:
            return _QueryResult(persisted=1)
        if statement == PERSIST_SOURCE_FLASHCARDS:
            return _QueryResult(persisted=1)
        return _QueryResult()


def test_source_learning_repository_preserves_typed_evidence_and_provenance():
    evidence = _evidence(SourceEntityLearningFactEvidence)
    session = _QuerySession(
        [
            {
                'hub_uuid': 'hub-1',
                'hub_name': 'Alice',
                'hub_description': 'A person.',
                'evidence': [evidence.model_dump()],
            }
        ]
    )
    repository = SourceLearningRepository(lambda: session)
    inputs = asyncio.run(repository.load_entity_learning_inputs('source-1'))
    assert inputs[0].hub_uuid == 'hub-1'
    assert inputs[0].evidence[0].source_fact_uuid == 'fact-1'

    result = asyncio.run(
        repository.replace_entity_learning_facts(
            'source-1',
            inputs,
            [
                SourceEntityLearningFactResult(
                    facts=[
                        {
                            'text': 'Alice works for Acme.',
                            'source_fact_uuids': ['fact-1'],
                            'triplet_uuids': ['triplet-1'],
                        }
                    ]
                )
            ],
        )
    )
    assert result == 1
    _, parameters = session.calls[-1]
    assert parameters['source_uuid'] == 'source-1'
    assert parameters['rows'][0]['hub_uuid'] == 'hub-1'
    assert parameters['rows'][0]['source_fact_uuids'] == ['fact-1']
    assert parameters['rows'][0]['triplet_uuids'] == ['triplet-1']


def test_source_learning_repository_cleanup_is_source_scoped():
    session = _QuerySession()
    repository = SourceLearningRepository(lambda: session)
    asyncio.run(repository.clear_source_learning('source-1'))
    statement, parameters = session.calls[-1]
    assert statement == _CLEAR_SOURCE_LEARNING
    assert parameters == {'source_uuid': 'source-1'}
