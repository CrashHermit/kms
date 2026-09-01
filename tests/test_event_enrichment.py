import asyncio

from kms.construction import entity_enrichment, event_enrichment
from kms.core import identity, models


def _mixed_triplet() -> models.Triplet:
    return models.Triplet(
        subject='analyst',
        predicate='scheduled',
        object='appointment',
        object_kind=models.NodeKind.EVENT,
        evidence_positions=[0],
    )


def _nodes() -> list[models.SourceNode]:
    return [
        models.SourceNode(
            uuid='node-0', content='The analyst scheduled an appointment.'
        )
    ]


def test_enrichment_splits_entity_and_event_endpoints(monkeypatch):
    captured: dict[str, dict] = {}

    async def describe(_nodes, terms_by_position, _enricher, *budgets):
        captured['terms'] = terms_by_position
        return {
            0: {
                term: f'{term} description'
                for term in sorted(next(iter(terms_by_position.values())))
            }
        }

    async def embed(descriptions):
        return {
            position: {
                term: [float(index + 1)] for index, term in enumerate(terms)
            }
            for position, terms in descriptions.items()
        }

    monkeypatch.setattr(entity_enrichment.semantic, 'describe_terms', describe)
    monkeypatch.setattr(event_enrichment.semantic, 'describe_terms', describe)
    monkeypatch.setattr(entity_enrichment.semantic, 'embed_descriptions', embed)
    monkeypatch.setattr(event_enrichment.semantic, 'embed_descriptions', embed)

    triplet = _mixed_triplet()
    entity_descriptions = asyncio.run(
        entity_enrichment.enrich(_nodes(), [triplet], object())
    )
    event_descriptions = asyncio.run(
        event_enrichment.enrich(_nodes(), [triplet], object())
    )

    assert entity_descriptions == {0: {'analyst': 'analyst description'}}
    assert event_descriptions == {0: {'appointment': 'appointment description'}}

    entity_bundle = models.ConstructionBundle(
        source=models.Source(key='book'), nodes=_nodes(), triplets=[triplet]
    )
    event_bundle = models.ConstructionBundle(
        source=models.Source(key='book'), nodes=_nodes(), triplets=[triplet]
    )
    entity_result = asyncio.run(
        entity_enrichment.EntityEnrichmentNode(object()).run(
            {'construction_bundle': entity_bundle}
        )
    )
    event_result = asyncio.run(
        event_enrichment.EventEnrichmentNode(object()).run(
            {'construction_bundle': event_bundle}
        )
    )

    assert [
        component.name for component in entity_result['entity_hub_components']
    ] == ['analyst']
    assert [
        component.name for component in event_result['event_hub_components']
    ] == ['appointment']
    assert event_result['event_hub_components'][0].uuid == identity.event_uuid(
        'book', 0, 'appointment'
    )
    assert event_result['event_descriptions'] == event_descriptions
    assert event_result['event_embeddings'] == {0: {'appointment': [1.0]}}
    assert set(captured['terms'][0]) == {'appointment'}


def test_event_enrichment_skips_semantic_calls_without_event_endpoints(
    monkeypatch,
):
    triplet = models.Triplet(
        subject='analyst',
        predicate='knows',
        object='method',
        evidence_positions=[0],
    )

    async def fail(*args, **kwargs):
        raise AssertionError('event semantic call should not run')

    monkeypatch.setattr(event_enrichment.semantic, 'describe_terms', fail)
    result = asyncio.run(event_enrichment.enrich(_nodes(), [triplet], object()))
    assert result == {}

    async def empty_enrich(*args, **kwargs):
        return {}

    async def fail_embed(*args, **kwargs):
        raise AssertionError('event embedding should not run')

    monkeypatch.setattr(event_enrichment, 'enrich', empty_enrich)
    monkeypatch.setattr(
        event_enrichment.semantic, 'embed_descriptions', fail_embed
    )
    bundle = models.ConstructionBundle(
        source=models.Source(key='book'), nodes=_nodes(), triplets=[triplet]
    )
    result = asyncio.run(
        event_enrichment.EventEnrichmentNode(object()).run(
            {'construction_bundle': bundle}
        )
    )
    assert result['event_descriptions'] == {}
    assert result['event_embeddings'] == {}
    assert result['event_hub_components'] == []
