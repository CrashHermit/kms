import ast
import asyncio
from pathlib import Path

from kms.construction import workflow

MODULE_DIR = Path(__file__).resolve().parent.parent / 'src' / 'kms'


def test_canonicalizer_node_only_assigns_source_hubs(monkeypatch):
    calls = []

    async def fake_assign(kind, source, **kwargs):
        calls.append((kind, source))
        return {
            'assigned': 2,
            'new_hubs': 1,
            'hierarchies': 0,
            'changed_hubs': ['hub-a'],
        }

    async def fail_align(*args, **kwargs):
        raise AssertionError('meta alignment belongs to maintenance')

    monkeypatch.setattr(workflow.canonicalizer, 'assign_source', fake_assign)
    monkeypatch.setattr(workflow.canonicalizer, 'align_meta', fail_align)
    monkeypatch.setattr(
        workflow.triplet_hubs,
        'rebuild',
        lambda **kwargs: asyncio.sleep(0, result={'triplet_hubs': 0}),
    )
    monkeypatch.setattr(
        workflow.name_hubs,
        'rebuild',
        lambda *args, **kwargs: asyncio.sleep(0, result={'name_hubs': 0}),
    )

    result = asyncio.run(
        workflow.CanonicalizerNode(object(), object()).run({'source': 'book-a'})
    )

    assert result == {
        'entity_assigned': 2,
        'predicate_assigned': 2,
        'entity_hubs_created': 1,
        'predicate_hubs_created': 1,
        'entity_name_hubs_created': 0,
        'predicate_name_hubs_created': 0,
        'triplet_hubs_created': 0,
    }
    assert calls == [('entity', 'book-a'), ('predicate', 'book-a')]


def test_document_ingestion_delegates_to_langgraph():
    source = (MODULE_DIR / 'construction' / 'runner.py').read_text()
    assert 'workflow.build_workflow' in source
    assert 'graph.ainvoke' in source
    ast.parse(source)


def test_workflow_defines_the_langgraph_composition():
    source = (MODULE_DIR / 'construction' / 'workflow.py').read_text()
    assert 'StateGraph' in source
    assert 'OCRNode' in source
    assert "graph.add_node('ocr'" in source
    assert 'graph.compile()' in source
    ast.parse(source)


def test_all_modules_parse():
    for path in MODULE_DIR.rglob('*.py'):
        ast.parse(path.read_text())
