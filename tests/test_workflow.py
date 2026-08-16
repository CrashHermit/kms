import ast
from pathlib import Path

from kms.construction import workflow

MODULE_DIR = Path(__file__).resolve().parent.parent / 'src' / 'kms'


def test_workflow_builds_independent_semantic_stage_modules(monkeypatch):
    calls = []

    def fake_module_lm(module_name):
        calls.append(module_name)
        return object()

    monkeypatch.setattr(workflow.llm, 'module_lm', fake_module_lm)
    modules = workflow._build_modules(None)

    expected = {
        'entity_enrichment',
        'predicate_enrichment',
        'entity_hub_builder',
        'predicate_hub_builder',
        'triplet_hub_builder',
        'statement_enrichment',
        'procedure_enrichment',
        'statement_hub_builder',
        'procedure_hub_builder',
    }
    assert expected <= modules.keys()
    assert expected <= set(calls)


def test_document_ingestion_delegates_to_langgraph():
    source = (MODULE_DIR / 'construction' / 'runner.py').read_text()
    assert 'workflow.build_workflow' in source
    assert 'graph.ainvoke' in source
    ast.parse(source)


def test_workflow_configures_new_learning_modules(monkeypatch):
    calls = []

    def fake_module_lm(module_name):
        calls.append(module_name)
        return object()

    monkeypatch.setattr(workflow.llm, 'module_lm', fake_module_lm)
    modules = workflow._build_modules(None)

    assert 'statement_enrichment' in modules
    assert 'procedure_enrichment' in modules
    assert 'statement_hub_builder' in modules
    assert 'procedure_hub_builder' in modules
    assert 'statement_enrichment' in calls
    assert 'procedure_enrichment' in calls
    assert 'statement_hub_builder' in calls
    assert 'procedure_hub_builder' in calls


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
