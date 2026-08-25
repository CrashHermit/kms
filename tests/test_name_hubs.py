import asyncio

from kms.construction import name_hubs
from kms.graph import names, queries, schema


def _row(uuid: str, text: str) -> dict:
    return {
        'uuid': uuid,
        'text': text,
        'normalized_text': names.normalize_text(text),
        'component_uuid': f'component-{uuid}',
        'source': 'book-a',
    }


def test_name_normalization_preserves_case_order_and_phrase_content():
    assert names.normalize_text('  Is   a subset  of  ') == 'Is a subset of'
    assert names.normalize_text('G') != names.normalize_text('g')
    assert names.normalize_text('is greater than') != names.normalize_text(
        'is not greater than'
    )


def test_lexical_groups_use_phrase_similarity_not_semantic_similarity():
    rows = [
        _row('a', 'color'),
        _row('b', 'colour'),
        _row('c', 'car'),
        _row('d', 'automobile'),
        _row('e', 'is a subset of'),
        _row('f', 'is not a subset of'),
    ]

    groups = names.lexical_groups(rows, threshold=0.8)
    groups_by_ids = {
        frozenset(row['uuid'] for row in group) for group in groups
    }

    assert frozenset({'a', 'b'}) in groups_by_ids
    assert frozenset({'c'}) in groups_by_ids
    assert frozenset({'d'}) in groups_by_ids
    assert frozenset({'e'}) in groups_by_ids
    assert frozenset({'f'}) in groups_by_ids


def test_entity_and_predicate_name_hubs_have_parallel_labels():
    assert names.name_label('entity') == 'EntityName'
    assert names.name_label('predicate') == 'PredicateName'
    assert names.name_hub_label('entity') == 'LocalEntityNameHub'
    assert names.name_hub_label('predicate') == 'LocalPredicateNameHub'
    assert names.name_hub_label('entity', tier='meta') == 'GlobalEntityNameHub'
    assert names.name_hub_label('predicate', tier='meta') == (
        'GlobalPredicateNameHub'
    )


def test_name_hub_ids_are_source_and_membership_stable():
    first = names.name_hub_uuid('entity', 'book-a', ['name-a', 'name-b'])
    second = names.name_hub_uuid('entity', 'book-a', ['name-b', 'name-a'])
    assert first == second
    assert first != names.name_hub_uuid(
        'entity', 'book-b', ['name-a', 'name-b']
    )
    assert names.meta_name_hub_uuid(
        'entity', ['source-name-a', 'source-name-b']
    ) == names.meta_name_hub_uuid('entity', ['source-name-b', 'source-name-a'])


def test_assertion_query_contains_lexical_occurrence_edges():
    assert 'MATCH (c:Entity' in queries.MERGE_HAS_ENTITY_NAME
    assert 'MERGE (c)-[:HAS_NAME]->(n)' in queries.MERGE_HAS_ENTITY_NAME
    assert 'MATCH (c:Predicate' in queries.MERGE_HAS_PREDICATE_NAME
    assert 'MERGE (c)-[:HAS_NAME]->(n)' in queries.MERGE_HAS_PREDICATE_NAME


def test_lexical_name_hub_query_uses_exact_normalized_phrase():
    captured = []

    class _Result:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def run(self, cypher, **kwargs):
            captured.append((cypher, kwargs))
            return _Result()

    async def scenario():
        return await queries.lexical_name_hubs(
            lambda: _Session(),
            'predicate',
            '  is   a subset of ',
            source='book-a',
        )

    assert asyncio.run(scenario()) == []
    assert 'PredicateNameHub' in captured[0][0]
    assert 'h.normalized_form = $normalized' in captured[0][0]
    assert captured[0][1] == {
        'normalized': 'is a subset of',
        'source': 'book-a',
    }


def test_meta_name_hubs_use_alignment_edges():
    merge = queries.merge_meta_name_hubs_query('entity')
    align = queries.merge_name_hub_alignments_query('predicate')
    delete = queries.delete_meta_name_hubs_query('entity')

    assert 'GlobalEntityNameHub' in merge
    assert 'LocalPredicateNameHub' in align
    assert 'GlobalPredicateNameHub' in align
    assert 'MERGE (s)-[r:ALIGNS_TO]->(m)' in align
    assert 'DETACH DELETE h' in delete


def test_semantic_name_projection_uses_occurrence_membership_path():
    cypher = queries.merge_semantic_name_links_query('predicate')

    assert 'LocalPredicateHub' in cypher
    assert '<-[:CANONICAL]-(c:Predicate)' in cypher
    assert '-[:HAS_NAME]->(n:PredicateName)' in cypher
    assert '-[:LEXICAL_CANONICAL]->(nh:LocalPredicateNameHub)' in cypher
    assert 'MERGE (h)-[:HAS_NAME_HUB]->(nh)' in cypher


def test_meta_name_hubs_rebuild_from_qualified_source_hubs(monkeypatch):
    rows = [
        {
            'uuid': 'source-name-a',
            'text': 'color',
            'normalized_text': 'color',
            'aliases': ['color'],
            'source': 'book-a',
        },
        {
            'uuid': 'source-name-b',
            'text': 'colour',
            'normalized_text': 'colour',
            'aliases': ['colour'],
            'source': 'book-b',
        },
        {
            'uuid': 'source-name-c',
            'text': 'automobile',
            'normalized_text': 'automobile',
            'aliases': ['automobile'],
            'source': 'book-b',
        },
    ]
    calls = []

    async def fake_all_name_hubs(session_factory, kind):
        return rows

    async def fake_clear(kind, *, session_factory):
        calls.append(('clear', kind))

    async def fake_persist(kind, hubs, *, session_factory):
        calls.append(('persist', kind, hubs))

    class _Synth:
        async def aforward(self, **kwargs):
            calls.append(('llm', kwargs['surface_forms']))
            return 0

    monkeypatch.setattr(name_hubs.queries, 'all_name_hubs', fake_all_name_hubs)
    monkeypatch.setattr(name_hubs.writer, 'clear_meta_name_hubs', fake_clear)
    monkeypatch.setattr(
        name_hubs.writer, 'persist_meta_name_hubs', fake_persist
    )
    monkeypatch.setattr(
        name_hubs, '_LexicalDefinitionSynthesizer', lambda lm: _Synth()
    )
    monkeypatch.setattr(
        name_hubs.llm, 'gate', lambda max_concurrency: asyncio.Semaphore(1)
    )

    async def fake_judged_groups(rows, kind, **kwargs):
        return names.lexical_groups(
            rows, threshold=kwargs['similarity_threshold']
        )

    monkeypatch.setattr(name_hubs, '_judged_groups', fake_judged_groups)

    result = asyncio.run(
        name_hubs.rebuild_meta(
            'entity',
            language_model=object(),
            session_factory=object(),
            similarity_threshold=0.8,
        )
    )

    assert result == {'name_hubs': 1, 'source_name_hubs': 3}
    assert calls[0] == ('clear', 'entity')
    assert ('llm', ['color', 'colour']) in calls
    persisted = next(call[2] for call in calls if call[0] == 'persist')
    assert persisted[0]['members'] == ['source-name-a', 'source-name-b']
    assert persisted[0]['sources'] == ['book-a', 'book-b']
    assert persisted[0]['aliases'] == ['color', 'colour']


def test_schema_contains_lexical_constraints_and_text_indexes():
    combined = '\n'.join(schema.schema_statements())

    assert 'entity_name_uuid' in combined
    assert 'predicate_name_uuid' in combined
    assert 'entity_name_hub_uuid' in combined
    assert 'predicate_name_hub_uuid' in combined
    assert 'entity_name_text' in combined
    assert 'predicate_name_text' in combined
    assert 'entity_name_hub_form' in combined
    assert 'predicate_name_hub_form' in combined
    assert 'entity_name_hub_normalized' in combined
    assert 'predicate_name_hub_normalized' in combined
    assert 'meta_entity_name_hub_uuid' in combined
    assert 'meta_predicate_name_hub_uuid' in combined
    assert 'meta_entity_name_hub_normalized' in combined
    assert 'meta_predicate_name_hub_normalized' in combined


def test_exact_lexical_duplicates_merge_without_judgment(monkeypatch):
    rows = [
        _row('a', 'means that'),
        _row('b', 'means that'),
        _row('c', 'colour'),
    ]
    calls = []

    class _Judge:
        async def aforward(self, **kwargs):
            calls.append(kwargs)
            return 'Separate'

    monkeypatch.setattr(
        name_hubs, '_LexicalMembershipJudge', lambda lm: _Judge()
    )

    async def scenario():
        return await name_hubs._judged_groups(
            rows,
            'predicate',
            language_model=object(),
            similarity_threshold=0.8,
            gate=asyncio.Semaphore(1),
        )

    groups = asyncio.run(scenario())

    assert {frozenset(row['uuid'] for row in group) for group in groups} == {
        frozenset({'a', 'b'}),
        frozenset({'c'}),
    }
    assert calls == []


def test_lexical_judge_rejects_a_deterministic_candidate_before_naming(
    monkeypatch,
):
    rows = [
        _row('a', 'color'),
        _row('b', 'colour'),
        _row('c', 'automobile'),
    ]
    calls = []

    class _Judge:
        async def aforward(self, **kwargs):
            calls.append(kwargs)
            return 'Separate'

    monkeypatch.setattr(
        name_hubs, '_LexicalMembershipJudge', lambda lm: _Judge()
    )

    async def scenario():
        return await name_hubs._judged_groups(
            rows,
            'entity',
            language_model=object(),
            similarity_threshold=0.8,
            gate=asyncio.Semaphore(1),
        )

    groups = asyncio.run(scenario())

    assert [row['text'] for row in groups[0]] == ['color']
    assert all(len(group) == 1 for group in groups)
    assert calls == [{'left': 'color', 'right': 'colour', 'kind': 'entity'}]


def test_name_hub_rebuild_uses_no_embedding_search(monkeypatch):
    calls = []

    async def fake_components(session_factory, source):
        return [
            {
                'uuid': 'component-a',
                'name': 'is a subset of',
                'node_id': 1,
                'source': source,
            },
            {
                'uuid': 'component-b',
                'name': 'is a subset of',
                'node_id': 2,
                'source': source,
            },
        ]

    async def fake_names(session_factory, kind, source):
        return [
            _row(
                names.name_uuid('predicate', 'component-a'),
                'is a subset of',
            ),
            _row(
                names.name_uuid('predicate', 'component-b'),
                'is a subset of',
            ),
        ]

    async def fake_persist_occurrences(*args, **kwargs):
        calls.append('occurrences')

    async def fake_clear(*args, **kwargs):
        calls.append('clear')

    async def fake_persist(*args, **kwargs):
        calls.append(('persist', args[0], kwargs['source']))

    class _Synth:
        async def aforward(self, **kwargs):
            calls.append(('llm', kwargs['surface_forms']))
            return 0

    monkeypatch.setattr(
        name_hubs.queries, 'all_predicate_components', fake_components
    )
    monkeypatch.setattr(name_hubs.queries, 'all_name_occurrences', fake_names)
    monkeypatch.setattr(
        name_hubs.writer,
        'persist_name_occurrences',
        fake_persist_occurrences,
    )
    monkeypatch.setattr(name_hubs.writer, 'clear_name_hubs', fake_clear)
    monkeypatch.setattr(name_hubs.writer, 'persist_name_hubs', fake_persist)
    monkeypatch.setattr(
        name_hubs, '_LexicalDefinitionSynthesizer', lambda lm: _Synth()
    )
    monkeypatch.setattr(
        name_hubs.llm, 'gate', lambda max_concurrency: asyncio.Semaphore(1)
    )

    async def fake_judged_groups(rows, kind, **kwargs):
        return names.lexical_groups(
            rows, threshold=kwargs['similarity_threshold']
        )

    monkeypatch.setattr(name_hubs, '_judged_groups', fake_judged_groups)

    result = asyncio.run(
        name_hubs.rebuild(
            'predicate',
            'book-a',
            language_model=object(),
            session_factory=object(),
        )
    )

    assert result == {'name_hubs': 1, 'names': 2}
    assert calls[0] == 'occurrences'
    assert ('llm', ['is a subset of']) in calls
    assert calls[-1][0] == 'persist'
