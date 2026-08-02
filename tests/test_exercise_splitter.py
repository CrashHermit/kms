"""Exercise splitter: split packed exercise nodes."""

import asyncio

from kms.core import models
from kms.ingestion import splitter


class _ScriptedSplitter:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes):
        return self._scripted.pop(0) if self._scripted else []


def _nodes():
    return [
        models.ASTNode(
            type='paragraph',
            content='In Exercises 3-4, compute the determinant.',
            id=0,
            segment_index=0,
        ),
        models.ASTNode(
            type='list', content='3 matrix A\n4 matrix B', id=1, segment_index=0
        ),
        models.ASTNode(
            type='paragraph', content='ordinary prose', id=2, segment_index=0
        ),
    ]


def test_splits_a_packed_node():
    split = splitter.NodeSplit(
        position=1,
        exercises=[
            splitter.SplitExercise(number='3', content='matrix A'),
            splitter.SplitExercise(number='4', content='matrix B'),
        ],
    )
    out = asyncio.run(
        splitter.split_exercises(_nodes(), module=_ScriptedSplitter([[split]]))
    )
    assert [n.id for n in out] == [0, 1, 2, 3]
    assert [n.content for n in out] == [
        'In Exercises 3-4, compute the determinant.',
        '3 matrix A',
        '4 matrix B',
        'ordinary prose',
    ]
    assert all(n.segment_index == 0 for n in out)
    assert out[1].type == 'list'
    assert out[2].type == 'list'


def test_lead_in_broken_out():
    split = splitter.NodeSplit(
        position=1,
        exercises=[
            splitter.SplitExercise(number='3', content='matrix A'),
            splitter.SplitExercise(number='', content='4-5 find the inverse.'),
            splitter.SplitExercise(number='4', content='matrix B'),
        ],
    )
    out = asyncio.run(
        splitter.split_exercises(_nodes(), module=_ScriptedSplitter([[split]]))
    )
    assert [n.content for n in out] == [
        'In Exercises 3-4, compute the determinant.',
        '3 matrix A',
        '4-5 find the inverse.',
        '4 matrix B',
        'ordinary prose',
    ]


def test_single_exercise_not_split():
    split = splitter.NodeSplit(
        position=1,
        exercises=[splitter.SplitExercise(number='3', content='only one')],
    )
    out = asyncio.run(
        splitter.split_exercises(_nodes(), module=_ScriptedSplitter([[split]]))
    )
    assert len(out) == 3


def test_no_verdict_passes_through():
    out = asyncio.run(
        splitter.split_exercises(_nodes(), module=_ScriptedSplitter([[]]))
    )
    assert [(n.id, n.content) for n in out] == [
        (0, 'In Exercises 3-4, compute the determinant.'),
        (1, '3 matrix A\n4 matrix B'),
        (2, 'ordinary prose'),
    ]


def test_splitter_node():
    split = splitter.NodeSplit(
        position=1,
        exercises=[
            splitter.SplitExercise(number='3', content='matrix A'),
            splitter.SplitExercise(number='4', content='matrix B'),
        ],
    )
    node = splitter.SplitterNode(module=_ScriptedSplitter([[split]]))
    out = asyncio.run(node.run({'nodes': _nodes()}))
    assert set(out) == {'nodes'}
    assert len(out['nodes']) == 4
