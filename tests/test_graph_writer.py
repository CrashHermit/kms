"""Writer planning — the pure grouping/edge logic, no database — plus a fake-session check that
persist_nodes issues the right queries/params in order (the live writes are also covered by the
opt-in integration test)."""

import asyncio

from kms.core.models import ASTNode, NodeType
from kms.graph.nodes import node_uuid, source_uuid
from kms.graph.writer import head_uuid, next_pairs, node_batches, persist_nodes

_STREAM = [
    ASTNode(type=NodeType.HEADER, content="§1", id=0, seg_index=0),
    ASTNode(type=NodeType.PARAGRAPH, content="a", id=1, seg_index=0),
    ASTNode(type=NodeType.MATH, content="$x$", id=2, seg_index=0),
    ASTNode(type=NodeType.PARAGRAPH, content="b", id=3, seg_index=0),
]


def test_batches_group_by_per_type_label():
    batches = node_batches(_STREAM, "book.pdf")
    assert set(batches) == {"Header", "Paragraph", "Math"}
    assert len(batches["Paragraph"]) == 2  # two paragraphs share the label bucket
    assert len(batches["Math"]) == 1


def test_batch_rows_carry_the_uuid_for_the_merge_key():
    rows = node_batches(_STREAM, "book.pdf")["Math"]
    assert rows[0]["uuid"] == node_uuid("book.pdf", 2)


def test_typeless_node_falls_into_the_none_bucket():
    batches = node_batches([ASTNode(content="?", id=0, seg_index=0)], "book.pdf")
    assert set(batches) == {None}


def test_next_pairs_thread_consecutive_nodes_in_order():
    pairs = next_pairs(_STREAM, "book.pdf")
    assert len(pairs) == 3  # n-1 edges for n nodes
    assert pairs[0] == {"from": node_uuid("book.pdf", 0), "to": node_uuid("book.pdf", 1)}
    assert pairs[-1] == {"from": node_uuid("book.pdf", 2), "to": node_uuid("book.pdf", 3)}


def test_next_pairs_empty_for_single_or_no_nodes():
    assert next_pairs(_STREAM[:1], "book.pdf") == []
    assert next_pairs([], "book.pdf") == []


def test_head_uuid_is_the_first_node_or_none():
    assert head_uuid(_STREAM, "book.pdf") == node_uuid("book.pdf", 0)
    assert head_uuid([], "book.pdf") is None


# --- persist_nodes orchestration, via a fake session (no server) ---


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


def test_persist_nodes_writes_source_nodes_head_and_next(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_nodes(_STREAM, "book.pdf", {"title": "T", "author": "A"}))

    queries = [q for q, _ in calls]
    # the :Source root is MERGEd first, carrying the book metadata + its deterministic uuid
    assert "MERGE (s:Source" in queries[0]
    assert calls[0][1]["uuid"] == source_uuid("book.pdf")
    assert calls[0][1]["props"]["title"] == "T" and calls[0][1]["props"]["author"] == "A"
    # a node MERGE applies the per-type :Math label
    assert any("SET n:Math" in q for q in queries)
    # the source hangs off the first node via :HEAD
    head = next(c for c in calls if ":HEAD" in c[0])
    assert head[1]["src"] == source_uuid("book.pdf")
    assert head[1]["head"] == node_uuid("book.pdf", 0)
    # the :NEXT chain has one edge fewer than the number of nodes
    nxt = next(c for c in calls if ":NEXT" in c[0])
    assert len(nxt[1]["pairs"]) == len(_STREAM) - 1


def test_persist_nodes_is_a_noop_for_an_empty_stream(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr("kms.graph.writer.driver", lambda: _FakeDriver(calls))
    asyncio.run(persist_nodes([], "book.pdf"))
    assert calls == []  # nothing opened, nothing written
