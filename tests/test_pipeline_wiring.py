"""Static graph-wiring checks — no imports, so they need none of the heavy deps."""

import ast
import pathlib
import re

MODULE_DIR = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'kms'


def test_every_send_target_and_dispatch_fallback_is_a_registered_node():
    registered = set(
        re.findall(
            r'add_node\("([a-z_]+)"', (MODULE_DIR / 'pipeline.py').read_text()
        )
    )
    send_targets, fallbacks = set(), set()
    for f in MODULE_DIR.rglob('*.py'):
        text = f.read_text()
        send_targets |= set(re.findall(r'Send\("([a-z_]+)"', text))
        fallbacks |= set(re.findall(r'return sends or "([a-z_]+)"', text))
    assert not (send_targets - registered), (
        f'Send targets with no node: {send_targets - registered}'
    )
    assert not (fallbacks - registered), (
        f'dispatch fallbacks with no node: {fallbacks - registered}'
    )


def test_all_modules_parse():
    for f in MODULE_DIR.rglob('*.py'):
        ast.parse(f.read_text())


def test_both_entity_layers_compile_into_a_graph():
    # the per-type layer and the block layer are interchangeable between the node persister and the
    # collector; either must wire into a compilable graph, which is what makes them comparable.
    from kms.pipeline import build_graph

    for layer in ('per-type', 'block'):
        assert build_graph(layer) is not None


def test_the_entity_layer_defaults_to_the_validated_per_type_path(monkeypatch):
    # the general path is opt-in until it is measured at parity (docs/GENERALIZATION.md, step 5),
    # and a typo in the env var must not silently pick a different layer.
    from kms.pipeline import ENTITY_LAYER_ENV, entity_layer

    monkeypatch.delenv(ENTITY_LAYER_ENV, raising=False)
    assert entity_layer() == 'per-type'
    monkeypatch.setenv(ENTITY_LAYER_ENV, 'blokc')
    assert entity_layer() == 'per-type'
    monkeypatch.setenv(ENTITY_LAYER_ENV, 'block')
    assert entity_layer() == 'block'
