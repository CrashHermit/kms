import asyncio
import os

import pytest

from kms.construction import local_entity_hubs, local_predicate_hubs, name_hubs
from kms.core import models
from kms.graph import db, nodes, schema, writer

pytestmark = pytest.mark.skipif(
    not os.environ.get('KMS_NEO4J_IT'),
    reason='set KMS_NEO4J_IT=1 (with KMS_DATABASE__URI/USERNAME/PASSWORD) '
    'to run the Neo4j integration test',
)


def test_connectivity_round_trip_and_idempotent_schema():
    async def scenario():
        def _session_factory():
            return db.session()

        try:
            await db.verify_connectivity()
            async with db.session() as session:
                result = await session.run('RETURN 1 AS n')
                record = await result.single()
                assert record['n'] == 1
            await schema.ensure_schema(_session_factory)
            await schema.ensure_schema(_session_factory)
        finally:
            await db.close_driver()

    asyncio.run(scenario())


def test_persist_nodes_upserts_labels_and_next_chain():
    source = 'integration-test-book'
    stream = [
        models.SourceNode(type='header', content='§1', id=0, document_index=0),
        models.SourceNode(type='paragraph', content='a', id=1, document_index=0),
        models.SourceNode(type='math', content='$x$', id=2, document_index=0),
    ]

    async def one(session, query):
        return await (await session.run(query)).single()

    async def scenario():
        def _session_factory():
            return db.session()

        try:
            meta = {'title': 'Test Book', 'author': 'A. Mathematician'}
            await schema.ensure_schema(_session_factory)
            await writer.persist_nodes(
                stream,
                source,
                session_factory=_session_factory,
                metadata=meta,
            )
            await writer.persist_nodes(
                stream,
                source,
                session_factory=_session_factory,
                metadata=meta,
            )
            await writer.persist_chain(
                stream, source, session_factory=_session_factory
            )
            await writer.persist_chain(
                stream, source, session_factory=_session_factory
            )
            async with db.session() as session:
                math = await one(
                    session,
                    "MATCH (s:Source {title: 'Test Book', "
                    "author: 'A. Mathematician'}) "
                    "MATCH (n:Math:Node {content: '$x$'}) "
                    'WHERE n.source = s.uuid RETURN count(n) AS c',
                )
                chain = await one(
                    session,
                    "MATCH (s:Source {title: 'Test Book', "
                    "author: 'A. Mathematician'})-[:HEAD]->(head:Node) "
                    'MATCH p=(head)-[:NEXT*]->(:Node) '
                    'RETURN max(length(p)) AS longest',
                )
                head = await one(
                    session,
                    "MATCH (s:Source {title: 'Test Book', "
                    "author: 'A. Mathematician'})"
                    '-[:HEAD]->(n:Node) RETURN n.content AS c',
                )
                assert math['c'] == 1
                assert chain['longest'] == 2
                assert head['c'] == '§1'
        finally:
            async with db.session() as session:
                await session.run(
                    'MATCH (n) WHERE n.source = $source_uuid DETACH DELETE n',
                    source_uuid=nodes.source_uuid('integration-test-book'),
                )
                await session.run(
                    'MATCH (s:Source {title: $title, author: $author}) '
                    'DETACH DELETE s',
                    title='Test Book',
                    author='A. Mathematician',
                )
            await db.close_driver()

    asyncio.run(scenario())


def test_meta_rebuild_preserves_durable_components_and_source_hubs(monkeypatch):
    sources = {
        'integration-meta-book-a': nodes.source_uuid('integration-meta-book-a'),
        'integration-meta-book-b': nodes.source_uuid('integration-meta-book-b'),
    }
    entity_components = ['integration-entity-a', 'integration-entity-b']
    predicate_components = [
        'integration-predicate-a',
        'integration-predicate-b',
    ]
    source_hubs = {
        'entity': ['integration-entity-hub-a', 'integration-entity-hub-b'],
        'predicate': [
            'integration-predicate-hub-a',
            'integration-predicate-hub-b',
        ],
    }
    meta_hubs = {
        'entity': 'integration-meta-entity-hub',
        'predicate': 'integration-meta-predicate-hub',
    }

    async def fake_build_hubs(*args, **kwargs):
        return {
            'clusters': [],
            'hubs': [
                {
                    'uuid': meta_hubs['entity'],
                    'canonical_name': 'entity',
                    'aliases': ['entity'],
                    'description': 'entity meta hub',
                    'members': source_hubs['entity'],
                }
            ],
        }

    async def fake_build_hubs_predicate(*args, **kwargs):
        return {
            'clusters': [],
            'hubs': [
                {
                    'uuid': meta_hubs['predicate'],
                    'canonical_name': 'predicate',
                    'aliases': ['predicate'],
                    'description': 'predicate meta hub',
                    'members': source_hubs['predicate'],
                }
            ],
        }

    monkeypatch.setattr(local_entity_hubs, 'build_hubs', fake_build_hubs)
    monkeypatch.setattr(local_predicate_hubs, 'build_hubs', fake_build_hubs_predicate)
    monkeypatch.setattr(
        name_hubs,
        'rebuild_global',
        lambda *args, **kwargs: asyncio.sleep(
            0, result={'name_hubs': 0, 'source_name_hubs': 0}
        ),
    )

    async def scenario():
        def _session_factory():
            return db.session()

        try:
            await schema.ensure_schema(_session_factory)
            async with db.session() as session:
                await session.run(
                    'UNWIND $sources AS row '
                    'MERGE (s:Source {key: row.key}) '
                    'SET s.uuid = row.uuid',
                    sources=[
                        {'key': key, 'uuid': uuid}
                        for key, uuid in sources.items()
                    ],
                )
                await session.run(
                    'UNWIND $rows AS row '
                    'MERGE (e:Entity {uuid: row.uuid}) '
                    'SET e.source = row.source, e.node_id = 1, '
                    'e.name = row.name',
                    rows=[
                        {
                            'uuid': entity_components[0],
                            'source': sources['integration-meta-book-a'],
                            'name': 'entity-a',
                        },
                        {
                            'uuid': entity_components[1],
                            'source': sources['integration-meta-book-b'],
                            'name': 'entity-b',
                        },
                    ],
                )
                await session.run(
                    'UNWIND $rows AS row '
                    'MERGE (p:Predicate {uuid: row.uuid}) '
                    'SET p.source = row.source, p.node_id = 1, '
                    'p.predicate = row.predicate',
                    rows=[
                        {
                            'uuid': predicate_components[0],
                            'source': sources['integration-meta-book-a'],
                            'predicate': 'relates-a',
                        },
                        {
                            'uuid': predicate_components[1],
                            'source': sources['integration-meta-book-b'],
                            'predicate': 'relates-b',
                        },
                    ],
                )

            for kind, component_ids in {
                'entity': entity_components,
                'predicate': predicate_components,
            }.items():
                for source, component_id, hub_id in zip(
                    sources,
                    component_ids,
                    source_hubs[kind],
                    strict=True,
                ):
                    hub = {
                        'uuid': hub_id,
                        'source': source,
                        'canonical_name': kind,
                        'aliases': [kind],
                        'description': f'{kind} source hub',
                        'members': [component_id],
                    }
                    persist = (
                        writer.persist_entity_hubs
                        if kind == 'entity'
                        else writer.persist_predicate_hubs
                    )
                    await persist(
                        [hub],
                        tier='source',
                        session_factory=_session_factory,
                    )

            for kind in ('entity', 'predicate'):
                await (
                    local_entity_hubs.rebuild_global
                    if kind == 'entity'
                    else local_predicate_hubs.rebuild_global
                )(
                    language_model=object(),
                    adjudicator=object(),
                    synthesizer=object(),
                    session_factory=_session_factory,
                )

            async with db.session() as session:
                result = await (
                    await session.run(
                        'MATCH (e:Entity) WHERE e.uuid IN $entities '
                        'WITH count(e) AS entities '
                        'MATCH (p:Predicate) '
                        'WHERE p.uuid IN $predicates '
                        'WITH entities, count(p) AS predicates '
                        'MATCH (eh:EntityHub) '
                        'WHERE eh.uuid IN $entity_hubs '
                        'WITH entities, predicates, count(eh) AS entity_hubs '
                        'MATCH (ph:PredicateHub) '
                        'WHERE ph.uuid IN $predicate_hubs '
                        'WITH entities, predicates, entity_hubs, '
                        'count(ph) AS predicate_hubs '
                        'MATCH (meh:MetaEntityHub) '
                        'WHERE meh.uuid = $meta_entity '
                        'WITH entities, predicates, entity_hubs, '
                        'predicate_hubs, count(meh) AS meta_entities '
                        'MATCH (mph:MetaPredicateHub) '
                        'WHERE mph.uuid = $meta_predicate '
                        'RETURN entities, predicates, entity_hubs, '
                        'predicate_hubs, meta_entities, count(mph) '
                        'AS meta_predicates',
                        entities=entity_components,
                        predicates=predicate_components,
                        local_entity_hubs=source_hubs['entity'],
                        local_predicate_hubs=source_hubs['predicate'],
                        meta_entity=meta_hubs['entity'],
                        meta_predicate=meta_hubs['predicate'],
                    )
                ).single()
                assert result['entities'] == 2
                assert result['predicates'] == 2
                assert result['entity_hubs'] == 2
                assert result['predicate_hubs'] == 2
                assert result['meta_entities'] == 1
                assert result['meta_predicates'] == 1
        finally:
            async with db.session() as session:
                await session.run(
                    'MATCH (n) WHERE n.uuid IN $uuids DETACH DELETE n',
                    uuids=(
                        entity_components
                        + predicate_components
                        + source_hubs['entity']
                        + source_hubs['predicate']
                        + list(meta_hubs.values())
                    ),
                )
                await session.run(
                    'MATCH (s:Source) WHERE s.uuid IN $sources DETACH DELETE s',
                    sources=list(sources.values()),
                )
            await db.close_driver()

    asyncio.run(scenario())
