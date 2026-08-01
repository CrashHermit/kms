"""Variable extractor — pure functions and context assembly. No network/LLM."""

from kms.core import models, walker
from kms.ingestion import variable_extractor


def _paragraph(content, node_id=0):
    return models.ParagraphNode(content=content, id=node_id)


def _math(content, node_id=0):
    return models.MathNode(content=content, id=node_id)


# --- _units -----------------------------------------------------------------


def test_units_are_statements_procedures_then_unabsorbed_nodes():
    stmt = models.Statement(block=[1, 2], members=[1, 2])
    proc = models.Procedure(block=[5, 6], members=[5, 6])
    nodes = [
        _paragraph('a', node_id=1),
        _paragraph('b', node_id=2),
        _paragraph('c', node_id=5),
        _paragraph('d', node_id=6),
        _paragraph('e', node_id=9),
    ]
    assert variable_extractor._units(nodes, [stmt], [proc]) == [
        (models.UNIT_STATEMENT, [1, 2], [1, 2]),
        (models.UNIT_PROCEDURE, [5, 6], [5, 6]),
        (models.UNIT_NODE, [9], [9]),
    ]


def test_units_cover_a_both_block_with_two_hubs():
    # A statement and a procedure carry the SAME block but are distinct units
    # — the kind namespaces the block, so their artifacts never collide.
    stmt = models.Statement(block=[0, 1, 2, 3], members=[0, 1])
    proc = models.Procedure(block=[0, 1, 2, 3], members=[2, 3])
    nodes = [
        _paragraph('a', node_id=0),
        _paragraph('b', node_id=1),
        _paragraph('c', node_id=2),
        _paragraph('d', node_id=3),
        _paragraph('e', node_id=4),
    ]
    units = variable_extractor._units(nodes, [stmt], [proc])
    assert units == [
        (models.UNIT_STATEMENT, [0, 1, 2, 3], [0, 1]),
        (models.UNIT_PROCEDURE, [0, 1, 2, 3], [2, 3]),
        (models.UNIT_NODE, [4], [4]),
    ]


def test_units_skip_nodes_without_ids():
    nodes = [models.ParagraphNode(content='no id')]
    assert variable_extractor._units(nodes, [], []) == []


# --- _renderable_nodes ------------------------------------------------------


def test_renderable_nodes_returns_position_content_pairs():
    nodes = [
        _paragraph('First.', node_id=0),
        _paragraph('Second.', node_id=1),
        _paragraph('Third.', node_id=2),
    ]
    assert variable_extractor._renderable_nodes(nodes, [], []) == [
        (0, 'First.'),
        (1, 'Second.'),
        (2, 'Third.'),
    ]


def test_renderable_nodes_skips_empty_content():
    nodes = [
        _paragraph('First.', node_id=0),
        _paragraph('', node_id=1),
        _paragraph(None, node_id=2),
        _paragraph('Last.', node_id=3),
    ]
    assert variable_extractor._renderable_nodes(nodes, [], []) == [
        (0, 'First.'),
        (3, 'Last.'),
    ]


def test_renderable_nodes_replaces_group_with_its_block_text():
    stmt = models.Statement(block=[1, 2], members=[1, 2])
    nodes = [
        _paragraph('Prologue.', node_id=0),
        _paragraph('absorbed', node_id=1),
        _paragraph('also absorbed', node_id=2),
        _paragraph('After.', node_id=3),
    ]
    assert variable_extractor._renderable_nodes(nodes, [stmt], []) == [
        (0, 'Prologue.'),
        (1, 'absorbed\n\nalso absorbed'),
        (3, 'After.'),
    ]


def test_renderable_nodes_handles_a_both_block_as_one_entry():
    stmt = models.Statement(block=[1, 2, 3], members=[1, 2])
    proc = models.Procedure(block=[1, 2, 3], members=[2, 3])
    nodes = [
        _paragraph('Prologue.', node_id=0),
        _paragraph('posed', node_id=1),
        _paragraph('shared', node_id=2),
        _paragraph('working', node_id=3),
        _paragraph('After.', node_id=4),
    ]
    # One entry per BLOCK at its start position, the whole block's text.
    assert variable_extractor._renderable_nodes(nodes, [stmt], [proc]) == [
        (0, 'Prologue.'),
        (1, 'posed\n\nshared\n\nworking'),
        (4, 'After.'),
    ]


def test_renderable_nodes_handles_overlapping_statements():
    first = models.Statement(block=[0, 1], members=[0, 1])
    second = models.Statement(block=[1, 2], members=[1, 2])
    nodes = [
        _paragraph('a', node_id=0),
        _paragraph('b', node_id=1),
        _paragraph('c', node_id=2),
    ]
    # The second block's start (1) is used for its position.
    assert variable_extractor._renderable_nodes(nodes, [first, second], []) == [
        (0, 'a\n\nb'),
        (1, 'b\n\nc'),
    ]


# --- walker.context_around with renderable list -----------------------------


def test_context_around_gives_before_and_after():
    items = [(0, 'Prologue.'), (1, 'Focus.'), (3, 'After.')]
    before, after = walker.context_around(items, cursor=1)
    assert before == 'Prologue.'
    assert after == 'After.'


def test_context_around_at_start_gives_no_before():
    items = [(0, 'First.'), (2, 'Second.')]
    before, after = walker.context_around(items, cursor=0)
    assert before is None
    assert after == 'Second.'


def test_context_around_at_end_gives_no_after():
    items = [(0, 'First.'), (1, 'Last.')]
    before, after = walker.context_around(items, cursor=1)
    assert before == 'First.'
    assert after is None


def test_context_around_respects_token_budgets():
    # Three 300-char items (~76 tokens each): the first two fit
    # (~152 tokens), the third would push past the 200-token cap.
    chunk = 'a' * 300
    items = [
        (0, 'Prologue.'),
        (1, 'Focus.'),
        (2, chunk),
        (3, chunk),
        (4, chunk),
    ]
    before, after = walker.context_around(items, cursor=1)
    assert before == 'Prologue.'
    assert after == f'{chunk}\n\n{chunk}'
