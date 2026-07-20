"""The trainset loader reconstructs signature-typed dspy.Examples from captured JSONL."""

import json

from module import trainsets


def _write(tmp_path, name, lines):
    (tmp_path / f"{name}.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return tmp_path


def test_extractor_examples_reconstruct_node_models(tmp_path):
    d = _write(tmp_path, "extractor", [{
        "inputs": {"previous_node_context": None, "current_node": "# H\n\ntext", "next_node_context": None},
        "outputs": {"nodes": [{"type": "header", "content": "# H"}, {"type": "paragraph", "content": "text"}]},
    }])
    examples = trainsets.load("extractor", directory=d)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.current_node == "# H\n\ntext"
    assert [n.content for n in ex.nodes] == ["# H", "text"]


def test_entity_grouper_examples_reconstruct_windownodes_and_spans(tmp_path):
    d = _write(tmp_path, "entity_grouper", [{
        "inputs": {
            "previous_context": None,
            "current_nodes": [
                {"position": 0, "type": "paragraph", "content": "Theorem 1."},
                {"position": 1, "type": "paragraph", "content": "*Proof.* ..."},
            ],
            "next_context": None,
        },
        "outputs": {"entities": [
            {"type": "theorem", "start": 0, "end": 1, "continues_before": False, "continues_after": False}
        ]},
    }])
    ex = trainsets.load("entity_grouper", directory=d)[0]
    assert [w.position for w in ex.current_nodes] == [0, 1]
    assert ex.entities[0].type == "theorem" and ex.entities[0].end == 1


def test_entity_attributor_examples(tmp_path):
    d = _write(tmp_path, "entity_attributor", [{
        "inputs": {"entity_type": "theorem", "members": ["Thm ...", "*Proof.* ..."]},
        "outputs": {"roles": ["statement", "proof"]},
    }])
    ex = trainsets.load("entity_attributor", directory=d)[0]
    assert ex.entity_type == "theorem" and ex.members[0].startswith("Thm") and ex.roles == ["statement", "proof"]


def test_missing_file_returns_empty(tmp_path):
    assert trainsets.load("extractor", directory=tmp_path) == []
