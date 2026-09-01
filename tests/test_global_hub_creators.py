import asyncio
from importlib import import_module
from types import SimpleNamespace

from kms import config
from kms.core import embeddings

global_entity = import_module('kms.postprocessing.global.entity_hubs')
global_event = import_module('kms.postprocessing.global.event_hubs')
global_predicate = import_module('kms.postprocessing.global.predicate_hubs')


class _GlobalEmbedder:
    async def embed(self, texts):
        return [[0.7, 0.3] for _ in texts]


class _GlobalSynthesizer:
    async def aforward(self, surface_forms, descriptions, scope):
        return ('Global concept', 'A cross-source concept.')


class _GlobalAdjudicator:
    async def aforward(self, left, right, scope):
        return True


def _settings():
    stage = SimpleNamespace(
        merge_above=0.9,
        separate_below=0.2,
        recall_threshold=0.2,
        rerank_top_n=0,
        rerank_candidate_limit=32,
        comparison_token_budget=4096,
    )
    return SimpleNamespace(
        stages=SimpleNamespace(
            entity_hubs=stage, predicate_hubs=stage, event_hubs=stage
        ),
        concurrency=SimpleNamespace(max_concurrent_calls=16),
    )


def _record(uuid, source='book-a', vector=(1.0, 0.0)):
    return {
        'uuid': uuid,
        'name': uuid,
        'aliases': [],
        'description': 'description',
        'embedding': list(vector),
        'source': source,
    }


def _entity_spec():
    return global_entity.GlobalEntityHubSpec(
        graph=None,
        stage_name='entity_hubs',
        hub_id_factory=lambda records, definition: (
            'entity-global:'
            + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: None,
        synthesis_context='scope',
        all_components=None,
        clear_hubs=None,
        persist_hubs=None,
        rebuild_names=None,
        rebuild_triplets=None,
        all_source_hubs=None,
        qualified_global_hub_uuids=None,
        index_name='index',
        attach_global_hubs=None,
        clear_invalid_global_hubs=None,
        rebuild_global_names=None,
    )


def _predicate_spec():
    return global_predicate.GlobalPredicateHubSpec(
        graph=None,
        stage_name='predicate_hubs',
        hub_id_factory=lambda records, definition: (
            'predicate-global:'
            + '|'.join(sorted(record['uuid'] for record in records))
        ),
        source_resolver=lambda records: None,
        adjudication_context='compare',
        synthesis_context='scope',
        all_components=None,
        clear_hubs=None,
        persist_hubs=None,
        rebuild_names=None,
        rebuild_triplets=None,
        all_source_hubs=None,
        qualified_global_hub_uuids=None,
        index_name='index',
        attach_global_hubs=None,
        clear_invalid_global_hubs=None,
        rebuild_global_names=None,
    )


def test_global_builders_do_not_depend_on_construction_modules(monkeypatch):
    monkeypatch.setattr(config, 'get_settings', _settings)
    monkeypatch.setattr(embeddings, 'embedder', _GlobalEmbedder)
    records = [_record('b', 'book-b'), _record('a', 'book-a')]

    entity_result = asyncio.run(
        global_entity._build_global_entity_hubs(
            records,
            spec=_entity_spec(),
            max_concurrency=None,
            synthesizer=_GlobalSynthesizer(),
        )
    )
    predicate_result = asyncio.run(
        global_predicate._build_global_predicate_hubs(
            records,
            spec=_predicate_spec(),
            max_concurrency=None,
            adjudicator=_GlobalAdjudicator(),
            synthesizer=_GlobalSynthesizer(),
        )
    )

    assert entity_result['hubs'][0]['members'] == ['a', 'b']
    assert predicate_result['hubs'][0]['members'] == ['b', 'a']
    assert not any(
        name.startswith('local_hubs') for name in global_entity.__dict__
    )
    assert not any(
        name.startswith('local_hubs') for name in global_predicate.__dict__
    )
    assert 'local_hubs' not in global_event.__dict__


def test_global_event_builder_keeps_event_boundary_adjudication(monkeypatch):
    monkeypatch.setattr(config, 'get_settings', _settings)
    monkeypatch.setattr(embeddings, 'embedder', _GlobalEmbedder)
    calls = []

    class _RejectRelated:
        async def aforward(self, left, right, scope):
            calls.append(scope)
            return False

    result = asyncio.run(
        global_event._build_global_event_hubs(
            [
                _record('a', 'book-a'),
                _record('b', 'book-b', vector=(0.5, 0.866)),
            ],
            adjudicator=_RejectRelated(),
            synthesizer=_GlobalSynthesizer(),
            max_concurrency=None,
        )
    )
    assert len(result['hubs']) == 2
    assert calls == [
        'Compare event mentions by event meaning, trigger, and roles.'
    ]
