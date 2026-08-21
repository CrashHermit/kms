import asyncio

import pytest

from kms.construction import splitter
from kms.core import models


class _ScriptedSplitter:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    async def aforward(self, current_nodes, context_before=None):
        return self._scripted.pop(0) if self._scripted else []


def _nodes():
    return [
        models.Node(
            type='paragraph',
            content='In Exercises 3-4, compute the determinant.',
            uuid='node-0',
            document_index=0,
        ),
        models.Node(
            type='list',
            content='3 matrix A\n4 matrix B',
            uuid='node-1',
            document_index=0,
        ),
        models.Node(
            type='paragraph', content='ordinary prose', uuid='node-2', document_index=0
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
    assert [n.content for n in out] == [
        'In Exercises 3-4, compute the determinant.',
        '3 matrix A',
        '4 matrix B',
        'ordinary prose',
    ]
    assert all(n.document_index == 0 for n in out)
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


def test_invalid_split_position_fails_instead_of_being_clamped():
    split = splitter.NodeSplit(
        position=99,
        exercises=[
            splitter.SplitExercise(number='3', content='matrix A'),
            splitter.SplitExercise(number='4', content='matrix B'),
        ],
    )
    with pytest.raises(ValueError, match='invalid splitter position'):
        asyncio.run(
            splitter.split_exercises(
                _nodes(), module=_ScriptedSplitter([[split]])
            )
        )


def test_duplicate_split_position_fails():
    splits = [
        splitter.NodeSplit(
            position=1,
            exercises=[
                splitter.SplitExercise(number='3', content='matrix A'),
                splitter.SplitExercise(number='4', content='matrix B'),
            ],
        ),
        splitter.NodeSplit(
            position=1,
            exercises=[
                splitter.SplitExercise(number='5', content='matrix C'),
                splitter.SplitExercise(number='6', content='matrix D'),
            ],
        ),
    ]
    with pytest.raises(ValueError, match='duplicate splitter position'):
        asyncio.run(
            splitter.split_exercises(
                _nodes(), module=_ScriptedSplitter([splits])
            )
        )


def test_empty_split_item_fails():
    split = splitter.NodeSplit(
        position=1,
        exercises=[
            splitter.SplitExercise(number='3', content='matrix A'),
            splitter.SplitExercise(number='', content=''),
        ],
    )
    with pytest.raises(ValueError, match='empty exercise item'):
        asyncio.run(
            splitter.split_exercises(
                _nodes(), module=_ScriptedSplitter([[split]])
            )
        )


def test_single_exercise_is_invalid_split_output():
    split = splitter.NodeSplit(
        position=1,
        exercises=[splitter.SplitExercise(number='3', content='only one')],
    )
    with pytest.raises(ValueError, match='fewer than two'):
        asyncio.run(
            splitter.split_exercises(
                _nodes(), module=_ScriptedSplitter([[split]])
            )
        )


def test_no_verdict_passes_through():
    out = asyncio.run(
        splitter.split_exercises(_nodes(), module=_ScriptedSplitter([[]]))
    )
    assert [(n.content) for n in out] == [
        'In Exercises 3-4, compute the determinant.',
        '3 matrix A\n4 matrix B',
        'ordinary prose',
    ]


def test_splitter_preserves_mistral_block_boundaries_while_splitting_packed_lists():
    """Exercises packed by one OCR list block are the splitter's input."""
    nodes = [
        models.Node(
            type=models.NodeType.PARAGRAPH,
            content='Solve each problem below.',
            uuid='node-0',
            document_index=2,
            provenance={'provider': 'mistral', 'provider_index': 4},
        ),
        models.Node(
            type=models.NodeType.LIST,
            content='1. Solve $x + 1 = 2$.\\n2. Prove that $0 < 1$.',
            uuid='node-1',
            document_index=2,
            provenance={'provider': 'mistral', 'provider_index': 5},
        ),
        models.Node(
            type=models.NodeType.MATH,
            content='$$x = 1$$',
            uuid='node-2',
            document_index=2,
            provenance={'provider': 'mistral', 'provider_index': 6},
        ),
    ]
    split = splitter.NodeSplit(
        position=1,
        exercises=[
            splitter.SplitExercise(number='1', content='Solve $x + 1 = 2$.'),
            splitter.SplitExercise(number='2', content='Prove that $0 < 1$.'),
        ],
    )
    out = asyncio.run(
        splitter.split_exercises(nodes, module=_ScriptedSplitter([[split]]))
    )
    assert [node.content for node in out] == [
        'Solve each problem below.',
        '1 Solve $x + 1 = 2$.',
        '2 Prove that $0 < 1$.',
        '$$x = 1$$',
    ]
    assert [node.type for node in out] == [
        models.NodeType.PARAGRAPH,
        models.NodeType.LIST,
        models.NodeType.LIST,
        models.NodeType.MATH,
    ]
    assert [node.document_index for node in out] == [2, 2, 2, 2]


def test_splitter_leaves_already_separate_mistral_blocks_untouched():
    nodes = [
        models.Node(
            type=models.NodeType.LIST,
            content='1. First exercise.',
            uuid='node-0',
            document_index=2,
            provenance={'provider': 'mistral', 'provider_index': 5},
        ),
        models.Node(
            type=models.NodeType.LIST,
            content='2. Second exercise.',
            uuid='node-1',
            document_index=2,
            provenance={'provider': 'mistral', 'provider_index': 6},
        ),
    ]
    out = asyncio.run(
        splitter.split_exercises(nodes, module=_ScriptedSplitter([[]]))
    )
    assert [(node.content) for node in out] == [
        '1. First exercise.',
        '2. Second exercise.',
    ]
    assert [node.provenance['provider_index'] for node in out] == [5, 6]


def test_splitter_node_synchronizes_documents():
    split = splitter.NodeSplit(
        position=1,
        exercises=[
            splitter.SplitExercise(number='3', content='matrix A'),
            splitter.SplitExercise(number='4', content='matrix B'),
        ],
    )
    node = splitter.SplitterNode(module=_ScriptedSplitter([[split]]))
    documents = [
        models.Document(
            index=0,
            image_path='page.png',
            nodes=_nodes(),
        )
    ]
    out = asyncio.run(node.run({'nodes': _nodes(), 'documents': documents, 'source_key': 'test_source'}))
    assert set(out) == {'documents', 'nodes'}
    assert len(out['nodes']) == 4
    assert [node.content for node in out['documents'][0].nodes] == [
        node.content for node in out['nodes']
    ]