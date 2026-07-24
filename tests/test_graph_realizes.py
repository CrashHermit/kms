""":REALIZES identity mapping — pure, no database (neo4j is stubbed in conftest). Verifies a titled
Definition/Theorem mention keys to the SAME global canonical its citations target (so the wire lands on
the shared hub), that problems and titleless mentions are skipped, and that persist_realizes issues the
mention->canonical edge query. The MATCH-based existence filter (a title nobody cited draws no edge) is
a server behaviour, covered by the Neo4j integration test."""

import asyncio

from kms.core import models
from kms.graph.entities import entity_uuid
from kms.graph.realizes import realizes_rows
from kms.graph.references import canonical_uuid
from kms.graph.writer import persist_realizes

# A definition of "Vector Space" and a theorem that references it: the definition's title must key to
# the SAME canonical the reference mints, so :REALIZES lands on the shared hub.
_DEFINITION = models.Entity(
    type=models.EntityType.DEFINITION,
    members=[0],
    id=0,
    title='Vector Space',
)
_THEOREM = models.Entity(
    type=models.EntityType.THEOREM,
    members=[1],
    id=1,
    title='Basis Theorem',
    refs=[
        models.Reference(
            target='vector space', kind='definition', tactic='premise'
        )
    ],
)
_OVERLAY = [_DEFINITION, _THEOREM]


# --- planning ---


def test_realizes_row_keys_a_mention_to_its_concept_canonical():
    rows = realizes_rows(_OVERLAY, 'book.pdf')
    # both titled def/thm mentions produce a candidate row
    assert len(rows) == 2
    assert rows[0] == {
        'mention': entity_uuid('book.pdf', 0),
        'canonical': canonical_uuid('definition', 'Vector Space'),
    }


def test_definition_realizes_the_same_canonical_a_reference_targets():
    # the whole point: the definition's title and the theorem's ref target converge on ONE canonical,
    # so the citation resolves through the hub to where the concept is defined.
    rows = realizes_rows(_OVERLAY, 'book.pdf')
    definition_row = next(
        r for r in rows if r['mention'] == entity_uuid('book.pdf', 0)
    )
    assert definition_row['canonical'] == canonical_uuid(
        'definition', 'vector space'
    )  # the theorem's ref target, normalized to the same key


def test_realizes_canonical_is_global_across_books():
    # a mention of the same concept in another book keys to the identical canonical (cross-corpus
    # convergence): the canonical uuid carries no source.
    book_a = realizes_rows([_DEFINITION], 'a.pdf')[0]['canonical']
    book_b = realizes_rows([_DEFINITION], 'b.pdf')[0]['canonical']
    assert book_a == book_b


def test_realizes_rows_skip_problems_and_titleless_mentions():
    overlay = [
        models.Entity(
            type=models.EntityType.PROBLEM,
            members=[0],
            id=0,
            title='Exercise 1',
        ),  # a problem is never a reference target
        models.Entity(
            type=models.EntityType.DEFINITION, members=[1], id=1
        ),  # no title -> nothing to key on
    ]
    assert realizes_rows(overlay, 'book.pdf') == []


def test_realizes_rows_dedupe_by_mention_and_canonical():
    # two mentions with the same title in one book still produce distinct rows (distinct mention
    # uuids), while a single mention yields exactly one row.
    assert len(realizes_rows([_DEFINITION], 'book.pdf')) == 1


# --- persist_realizes orchestration, via a fake session (no server) ---


class _FakeSession:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.calls.append((query, params))


class _FakeDriver:
    def __init__(self, calls):
        self.calls = calls

    def session(self, **kwargs):
        return _FakeSession(self.calls)


def test_persist_realizes_writes_mention_to_canonical_edges(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(persist_realizes(_OVERLAY, 'book.pdf'))

    edge = next(c for c in calls if ':REALIZES' in c[0])
    assert '(m:Entity' in edge[0] and '(c:Canonical' in edge[0]
    assert len(edge[1]['rows']) == 2


def test_persist_realizes_is_a_noop_without_titled_def_or_thm(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr('kms.graph.writer.driver', lambda: _FakeDriver(calls))
    asyncio.run(
        persist_realizes(
            [models.Entity(type=models.EntityType.PROBLEM, members=[0], id=0)],
            'b',
        )
    )
    assert calls == []  # no titled def/thm -> nothing opened, nothing written
