import asyncio

from kms.core import identity, models
from kms.graph import projectors


async def _noop(*args, **kwargs):
    return None


def _patch_writers(monkeypatch):
    monkeypatch.setattr(projectors.schema, 'ensure_schema', _noop)
    for name in (
        'persist_nodes',
        'persist_statements',
        'persist_procedures',
        'persist_statement_procedure_links',
        'persist_instructions',
        'persist_assertions',
        'persist_chain',
        'persist_entity_hubs',
        'persist_predicate_hubs',
        'attach_entity_components',
        'attach_predicate_components',
        'clear_triplet_hubs',
        'persist_triplet_hubs',
        'persist_statement_enrichment',
        'persist_procedure_enrichment',
        'clear_statement_hubs',
        'persist_statement_hubs',
        'clear_procedure_hubs',
        'persist_procedure_hubs',
    ):
        monkeypatch.setattr(projectors.writer, name, _noop)


def test_final_projector_is_noop_without_graph():
    result = asyncio.run(
        projectors.FinalProjectorNode(
            session_factory=None, neo4j_configured=False
        ).run({'source_key': 'book'})
    )
    assert result == {'projected': False}


def test_final_projector_writes_complete_state(monkeypatch):
    calls = []

    async def ensure_schema(session_factory):
        calls.append('schema')

    async def record(name, *args, **kwargs):
        calls.append(name)

    monkeypatch.setattr(projectors.schema, 'ensure_schema', ensure_schema)
    for name in (
        'persist_nodes',
        'persist_statements',
        'persist_procedures',
        'persist_statement_procedure_links',
        'persist_instructions',
        'persist_assertions',
        'persist_chain',
        'persist_entity_hubs',
        'persist_predicate_hubs',
        'attach_entity_components',
        'attach_predicate_components',
        'persist_statement_enrichment',        'persist_procedure_enrichment',
        'clear_statement_hubs',
        'persist_statement_hubs',
        'clear_procedure_hubs',
        'persist_procedure_hubs',
    ):
        monkeypatch.setattr(
            projectors.writer,
            name,
            lambda *args, _name=name, **kwargs: record(_name, *args, **kwargs),
        )
    state = {
        'source_key': 'book',
        'nodes': [],
        'statements': [],
        'procedures': [],
        'instructions': [],
        'triplets': [],
        'statement_enrichments': [],
        'procedure_enrichments': [],
        'statement_hubs': [],
        'procedure_hubs': [],
    }
    result = asyncio.run(
        projectors.FinalProjectorNode(object(), True).run(state)
    )
    assert result == {'projected': True}
    assert calls[0] == 'schema'
    assert 'persist_assertions' in calls


def test_final_projector_persists_procedure_updates(monkeypatch):
    _patch_writers(monkeypatch)
    result = asyncio.run(
        projectors.FinalProjectorNode(object(), True).run(
            {
                'source_key': 'book',
                'generated_procedures': [],
                'procedure_links': [],
            }
        )
    )
    assert result == {'projected': True}


def test_final_projector_forwards_assertions(monkeypatch):
    calls = []
    _patch_writers(monkeypatch)

    async def persist(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(projectors.writer, 'persist_assertions', persist)
    triplet = models.Triplet(
        subject='entity',
        predicate='relates',
        object='name',
        evidence_positions=[0],
    )
    identity.assign_triplet_ids([triplet], 'book')
    state = {
        'source_key': 'book',
        'nodes': [models.Node(uuid='node-0', content='fact')],
        'triplets': [triplet],
        'entity_descriptions': {0: {'entity': 'description'}},
        'predicate_descriptions': {0: {'predicate': 'description'}},
        'entity_embeddings': {0: {'entity': [1.0]}},
        'predicate_embeddings': {0: {'predicate': [1.0]}},
    }
    result = asyncio.run(
        projectors.FinalProjectorNode(object(), True).run(state)
    )
    assert result == {'projected': True}
    assert calls[0][0][:3] == (
        state['triplets'],
        'book',
        state['nodes'],
    )
