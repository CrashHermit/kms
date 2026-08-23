import asyncio
import base64

import pytest

from kms.construction import splitter
from kms.core import content, identity, models, walker


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


def test_encode_wraps_text_and_image_context_as_content_parts(tmp_path):
    image_path = tmp_path / 'context.png'
    image_path.write_bytes(
        base64.b64decode(
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk'
            '+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
        )
    )
    encoded = splitter.Splitter.encode(
        None,
        current_nodes=[
            walker.WindowNode(
                position=2,
                type='paragraph',
                content='candidate text',
            )
        ],
        context_before=[
            walker.WindowNode(
                position=1,
                type='image',
                image_path=str(image_path),
            )
        ],
    )

    assert isinstance(encoded['current_nodes'], content.ContentParts)
    assert isinstance(encoded['context_before'], content.ContentParts)
    assert encoded['current_nodes'].format()[0] == {
        'type': 'text',
        'text': '[2] (paragraph): candidate text',
    }
    assert [block['type'] for block in encoded['context_before'].format()] == [
        'text',
        'image_url',
    ]


def test_nodes_before_preserves_multimodal_nodes_and_order():
    nodes = [
        models.Node(type='image', image_path='/tmp/first.png', uuid='first'),
        models.Node(type='paragraph', content='second', uuid='second'),
        models.Node(type='list', content='target', uuid='target'),
    ]

    before = walker.nodes_before(nodes, cursor=2, budget=100)

    assert before == nodes[:2]
    assert before[0].image_path == '/tmp/first.png'


def test_split_rebuilds_multiple_exercises(tmp_path):
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
    assert out[1].uuid != out[2].uuid
    assert out[1].uuid == identity.split_child_uuid('node-1', 0)
    assert out[2].uuid == identity.split_child_uuid('node-1', 1)


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


def test_rebuild_preserves_image_governance_and_child_identity():
    node = models.Node(
        type='image',
        content='packed source',
        image_path='/tmp/figure.png',
        uuid='node-1',
        document_index=3,
        provenance={'provider': 'mistral'},
        governing_instruction_uuids=['instruction-1'],
    )
    split = splitter.NodeSplit(
        position=0,
        exercises=[
            splitter.SplitExercise(number='1', content='first'),
            splitter.SplitExercise(number='2', content='second'),
        ],
    )

    rebuilt = splitter._rebuild([node], splitter.Decision(splits={0: split.exercises}))

    assert [item.image_path for item in rebuilt] == [
        '/tmp/figure.png',
        '/tmp/figure.png',
    ]
    assert all(
        item.governing_instruction_uuids == ['instruction-1'] for item in rebuilt
    )
    assert rebuilt[0].uuid == identity.split_child_uuid('node-1', 0)
    assert rebuilt[1].uuid == identity.split_child_uuid('node-1', 1)


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