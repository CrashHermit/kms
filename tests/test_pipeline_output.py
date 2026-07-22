"""Pipeline output: flattening the three overlays and persisting nodes + entities."""

import json

from module.pipeline import _flatten_entities, _write_entities, _write_nodes
from module.state import ASTNode, Entity, EntityType, NodeType


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


def test_write_entities_serializes_the_instruction_attribute(tmp_path):
    entities = [
        Entity(
            type=EntityType.PROBLEM,
            members=[3],
            id=0,
            number="1.23",
            contents=["A"],
            instruction="find the eigenvalues",
        ),
    ]
    _write_entities(entities, tmp_path)
    payload = json.loads((tmp_path / "entities.json").read_text())
    assert payload == [
        {
            "id": 0,
            "type": "problem",
            "members": [3],
            "number": "1.23",
            "instruction": "find the eigenvalues",
            "contents": ["A"],
        }
    ]
