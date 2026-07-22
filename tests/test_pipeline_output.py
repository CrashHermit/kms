"""Pipeline output: flattening the three overlays and persisting nodes + entities."""

import json

from module.state import ASTNode, NodeType, EntityType, Entity
from module.pipeline import _flatten_entities, _write_nodes, _write_entities


def _nodes():
    return [ASTNode(type=NodeType.PARAGRAPH, content=str(i), id=i, seg_index=0) for i in range(6)]


def test_flatten_concatenates_orders_by_document_position_and_assigns_ids():
    nodes = _nodes()
    result = {
        "problem_entities": [Entity(type=EntityType.PROBLEM, members=[3, 4])],
        "definition_entities": [Entity(type=EntityType.DEFINITION, members=[0])],
        "theorem_entities": [Entity(type=EntityType.THEOREM, members=[1, 2])],
    }
    flat = _flatten_entities(result, nodes)
    # Ordered by first member's document position (def@0, thm@1, prob@3), ids 0..2.
    assert [(e.id, e.type.value, e.members) for e in flat] == [
        (0, "definition", [0]),
        (1, "theorem", [1, 2]),
        (2, "problem", [3, 4]),
    ]


def test_write_nodes_persists_the_stream_for_provenance(tmp_path):
    _write_nodes(_nodes(), tmp_path)
    payload = json.loads((tmp_path / "nodes.json").read_text())
    assert [n["id"] for n in payload] == list(range(6))
    assert payload[0] == {"id": 0, "type": "paragraph", "content": "0", "seg_index": 0}


def test_write_entities_is_a_flat_list_with_members_into_nodes(tmp_path):
    entities = [
        Entity(type=EntityType.DEFINITION, members=[0], id=0),
        Entity(type=EntityType.THEOREM, members=[1, 2], id=1),
    ]
    _write_entities(entities, tmp_path)
    payload = json.loads((tmp_path / "entities.json").read_text())
    assert payload == [
        {"id": 0, "type": "definition", "members": [0]},
        {"id": 1, "type": "theorem", "members": [1, 2]},
    ]


def test_governor_split_supersedes_the_coarse_problem_on_its_block():
    """A grouped exercise list (nodes 3,4) that the governor split into two per-exercise
    entities: the problem finder's coarse duplicate over the same nodes is dropped, the
    fine exercises are kept, and a problem elsewhere survives untouched."""
    nodes = _nodes()
    result = {
        "problem_entities": [
            Entity(type=EntityType.PROBLEM, members=[3, 4]),  # coarse, over the governed block
            Entity(type=EntityType.PROBLEM, members=[5]),     # a real problem outside the block
        ],
        "exercise_entities": [
            Entity(type=EntityType.PROBLEM, members=[3, 4], number="1.23", contents=["A"]),
            Entity(type=EntityType.PROBLEM, members=[3, 4], number="1.24", contents=["B"]),
        ],
    }
    flat = _flatten_entities(result, nodes)
    # Coarse [3,4] gone; the two fine exercises kept; the [5] problem survives.
    assert [(e.number, e.members) for e in flat] == [
        ("1.23", [3, 4]),
        ("1.24", [3, 4]),
        (None, [5]),
    ]


def test_write_entities_serializes_the_governor_instruction(tmp_path):
    entities = [
        Entity(type=EntityType.PROBLEM, members=[3, 4], id=0, number="1.23",
               contents=["A"], instruction="find the eigenvalues"),
    ]
    _write_entities(entities, tmp_path)
    payload = json.loads((tmp_path / "entities.json").read_text())
    assert payload == [{
        "id": 0, "type": "problem", "members": [3, 4],
        "number": "1.23", "instruction": "find the eigenvalues", "contents": ["A"],
    }]
