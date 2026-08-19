import asyncio

import pytest

from kms.graph import hubs, queries, schema, writer


def test_hub_labels_distinguish_source_and_meta_tiers():
    assert hubs.hub_label('entity', tier='source') == 'EntityHub'
    assert hubs.hub_label('predicate', tier='source') == 'PredicateHub'
    assert hubs.meta_hub_label('entity') == 'MetaEntityHub'
    assert hubs.meta_hub_label('predicate') == 'MetaPredicateHub'


def test_hub_label_rejects_unknown_tier():
    with pytest.raises(ValueError, match='unknown hub tier'):
        hubs.hub_label('entity', tier='unknown')


def test_component_label_rejects_unknown_kind():
    assert hubs.component_label('entity') == 'Entity'
    assert hubs.component_label('predicate') == 'Predicate'

    with pytest.raises(ValueError, match='unknown component kind'):
        hubs.component_label('unknown')


def test_alignment_query_uses_distinct_relationship_and_labels():
    cypher = queries.merge_alignment_query(
        hubs.ENTITY_HUB_LABEL,
        hubs.META_ENTITY_HUB_LABEL,
    )

    assert '(s:EntityHub {uuid: pair.source_hub})' in cypher
    assert '(m:MetaEntityHub {uuid: pair.meta_hub})' in cypher
    assert 'MERGE (s)-[r:ALIGNS_TO]->(m)' in cypher
    assert 'r.score = pair.score' in cypher
    assert 'r.decision = pair.decision' in cypher


def test_schema_contains_meta_hub_constraints_and_indexes():
    statements = schema.schema_statements()
    combined = '\n'.join(statements)

    assert 'meta_entity_hub_uuid' in combined
    assert 'meta_predicate_hub_uuid' in combined
    assert 'meta_entity_hub_embedding' in combined
    assert 'meta_predicate_hub_embedding' in combined
    assert '`vector.dimensions`: 1024' in combined


def test_meta_hub_properties_require_stable_id_and_omit_source():
    properties = hubs.hub_properties(
        kind='entity',
        source=None,
        canonical_name='graph',
        aliases=['graph'],
        description='A mathematical graph.',
        tier='meta',
        hub_id='meta-graph',
    )

    assert properties['uuid'] == 'meta-graph'
    assert 'source' not in properties

    with pytest.raises(ValueError, match='explicit hub_id'):
        hubs.hub_properties(
            kind='entity',
            source=None,
            canonical_name='graph',
            aliases=[],
            description='A graph.',
            tier='meta',
        )


def test_meta_hub_uuid_is_stable_for_identity():
    assert hubs.meta_hub_uuid('entity', 'source-hub-a') == hubs.meta_hub_uuid(
        'entity', 'source-hub-a'
    )
    assert hubs.meta_hub_uuid('entity', 'source-hub-a') != hubs.meta_hub_uuid(
        'entity', 'source-hub-b'
    )


def test_attach_source_components_merges_membership_and_aliases():
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))

    async def scenario():
        await writer.attach_entity_components(
            [{'component': 'entity-a', 'hub': 'hub-a'}],
            aliases=[{'hub': 'hub-a', 'aliases': ['graph', '$G']}],
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert len(captured) == 2
    assert 'MATCH (c:Entity {uuid: pair.component})' in captured[0][0]
    assert 'coalesce(h.aliases, [])' in captured[1][0]
    assert captured[1][1]['rows'] == [
        {'hub': 'hub-a', 'aliases': ['graph', '$G']}
    ]


def test_attach_meta_hubs_replaces_alignment_and_updates_aliases(
    monkeypatch,
):
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))

    async def fake_source_hubs(session_factory, hub_uuids=None):
        return [{'uuid': 'source-hub-a', 'source': 'book-a'}]

    async def fake_qualified_meta_hubs(session_factory):
        return {'meta-hub-a'}

    monkeypatch.setattr(writer.queries, 'all_entity_source_hubs', fake_source_hubs)
    monkeypatch.setattr(
        writer.queries,
        'qualified_entity_meta_hub_uuids',
        fake_qualified_meta_hubs,
    )

    async def scenario():
        await writer.attach_entity_meta_hubs(
            [
                {
                    'source_hub': 'source-hub-a',
                    'meta_hub': 'meta-hub-a',
                    'score': 0.9,
                    'decision': 'Merge',
                }
            ],
            aliases=[{'hub': 'meta-hub-a', 'aliases': ['graph']}],
            subsumption_edges=[],
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert len(captured) == 3
    assert 'DELETE r' in captured[0][0]
    assert 'MERGE (s)-[r:ALIGNS_TO]->(m)' in captured[1][0]
    assert captured[1][1]['pairs'][0]['decision'] == 'Merge'
    assert 'MetaEntityHub' in captured[2][0]


def test_attach_meta_hubs_rejects_unqualified_new_meta_hubs(monkeypatch):
    async def fake_source_hubs(session_factory, hub_uuids=None):
        return [{'uuid': 'source-hub-a', 'source': 'book-a'}]

    async def fake_qualified_meta_hubs(session_factory):
        return set()

    monkeypatch.setattr(writer.queries, 'all_entity_source_hubs', fake_source_hubs)
    monkeypatch.setattr(
        writer.queries,
        'qualified_entity_meta_hub_uuids',
        fake_qualified_meta_hubs,
    )

    with pytest.raises(ValueError, match='requires two distinct sources'):
        asyncio.run(
            writer.attach_entity_meta_hubs(
                [
                    {
                        'source_hub': 'source-hub-a',
                        'meta_hub': 'meta-hub-a',
                        'score': 0.9,
                        'decision': 'Merge',
                    }
                ],
                aliases=[],
                subsumption_edges=[],
                session_factory=lambda: None,
            )
        )


def test_clear_source_hubs_targets_one_source_only():
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))

    async def scenario():
        await writer.clear_entity_hubs(
            'book-a',
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert captured == [
        (
            'MATCH (src:Source {key: $source}), '
            '(h:EntityHub {source: src.uuid}) DETACH DELETE h',
            {'source': 'book-a'},
        )
    ]


def test_clear_invalid_meta_hubs_targets_underqualified_meta_nodes():
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append(cypher)

    async def scenario():
        await writer.clear_invalid_entity_meta_hubs(
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert captured == [
        'MATCH (h:MetaEntityHub) '
        'OPTIONAL MATCH (h)<-[:ALIGNS_TO]-(s) '
        'WITH h, count(DISTINCT s.source) AS source_count '
        'WHERE source_count < 2 DETACH DELETE h'
    ]


def test_clear_meta_hubs_targets_only_meta_label():
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append(cypher)

    async def scenario():
        await writer.clear_entity_meta_hubs(
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert captured == ['MATCH (h:MetaEntityHub) DETACH DELETE h']


def test_persist_hubs_rejects_singleton_meta_hubs():
    with pytest.raises(ValueError, match='at least two source-hub members'):
        asyncio.run(
            writer.persist_entity_hubs(
                [
                    {
                        'uuid': 'meta-graph',
                        'canonical_name': 'graph',
                        'aliases': ['graph'],
                        'description': 'A graph.',
                        'members': ['source-hub-a'],
                    }
                ],
                tier='meta',
                session_factory=lambda: None,
            )
        )


def test_persist_hubs_rejects_same_source_meta_hubs(monkeypatch):
    async def fake_source_hubs(session_factory, hub_uuids=None):
        return [
            {'uuid': 'source-hub-a', 'source': 'book-a'},
            {'uuid': 'source-hub-b', 'source': 'book-a'},
        ]

    monkeypatch.setattr(writer.queries, 'all_entity_source_hubs', fake_source_hubs)

    with pytest.raises(ValueError, match='two distinct source supports'):
        asyncio.run(
            writer.persist_entity_hubs(
                [
                    {
                        'uuid': 'meta-graph',
                        'canonical_name': 'graph',
                        'aliases': ['graph'],
                        'description': 'A graph.',
                        'members': ['source-hub-a', 'source-hub-b'],
                    }
                ],
                tier='meta',
                session_factory=lambda: None,
            )
        )


def test_persist_hubs_uses_alignment_for_meta_members(monkeypatch):
    captured = []

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))

    async def fake_source_hubs(session_factory, hub_uuids=None):
        return [
            {'uuid': 'source-hub-a', 'source': 'book-a'},
            {'uuid': 'source-hub-b', 'source': 'book-b'},
        ]

    monkeypatch.setattr(writer.queries, 'all_entity_source_hubs', fake_source_hubs)

    async def scenario():
        await writer.persist_entity_hubs(
            [
                {
                    'uuid': 'meta-graph',
                    'source': None,
                    'canonical_name': 'graph',
                    'aliases': ['graph'],
                    'description': 'A graph.',
                    'members': ['source-hub-a', 'source-hub-b'],
                }
            ],
            tier='meta',
            session_factory=lambda: _Session(),
        )

    asyncio.run(scenario())

    assert len(captured) == 2
    assert 'MERGE (h:MetaEntityHub' in captured[0][0]
    assert 'MERGE (s)-[r:ALIGNS_TO]->(m)' in captured[1][0]
    assert captured[1][1]['pairs'] == [
        {'source_hub': 'source-hub-a', 'meta_hub': 'meta-graph'},
        {'source_hub': 'source-hub-b', 'meta_hub': 'meta-graph'},
    ]
