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


def test_workflow_uses_one_final_projector():
    source = (MODULE_DIR / 'construction' / 'workflow.py').read_text()
    assert "graph.add_node('final_projector'" in source
    assert 'source_projector' not in source
    assert 'assertion_projector' not in source
    assert 'procedure_materialization_projector' not in source
    assert 'ingestion_persister' not in source
    assert "procedure_hub_exit, 'final_projector'" in source
    assert "'final_projector', END" in source


def test_managed_workflow_has_one_serial_bundle_route(monkeypatch):
    monkeypatch.setattr(workflow.llm, 'module_lm', lambda name: object())

    class Manager:
        async def aswitch(self, name):
            pass

    graph = workflow.build_workflow(model_manager=Manager()).get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ('hub_input', 'entity_hub_builder') not in edges
    assert ('hub_input', 'switch_to_entity_hub_builder') in edges
    assert (
        'switch_to_entity_hub_builder',
        'entity_hub_builder',
    ) in edges
    assert (
        'switch_to_procedure_enrichment',
        'switch_to_statement_hub_builder',
    ) not in edges
    assert (
        'procedure_enrichment',
        'switch_to_statement_hub_builder',
    ) in edges
    assert ('procedure_hub_builder', 'final_projector') in edges
    assert ('procedure_hub_builder', '__end__') not in edges
    assert ('switch_to_entity_hub_builder', '__end__') not in edges


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
