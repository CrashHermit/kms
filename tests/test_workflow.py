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
        'text_seam_merger',
        'text_seam_rewriter',
        'image_seam_merger',
        'image_enricher',
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


def test_workflow_has_no_model_switch_nodes(monkeypatch):
    monkeypatch.setattr(workflow.llm, 'module_lm', lambda name: object())
    graph = workflow.build_workflow().get_graph()
    node_names = set(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert not any(name.startswith('switch_to_') for name in node_names)
    assert ('statement_procedure_builder', 'governance_walker') in edges
    assert ('governance_walker', 'triplet_extraction') in edges
    assert ('procedure_hub_builder', 'final_projector') in edges


def test_workflow_runs_image_seams_after_text_seams(monkeypatch):
    monkeypatch.setattr(workflow.llm, 'module_lm', lambda name: object())
    graph = workflow.build_workflow().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ('formatter_collect', 'text_seam_even_worker') in edges
    assert ('text_seam_even_collect', 'text_seam_odd_dispatch') in edges
    assert ('text_seam_odd_collect', 'image_seam_even_dispatch') in edges
    assert ('image_seam_even_collect', 'image_seam_odd_dispatch') in edges
    assert ('image_seam_odd_collect', 'image_enrichment') in edges
    assert ('image_enrichment', 'splitter') in edges
    assert ('splitter', 'instruction_finder') in edges
    assert 'exercise_strip_router' not in graph.nodes
    for collector in (
        'text_seam_even_collect',
        'text_seam_odd_collect',
        'image_seam_even_collect',
        'image_seam_odd_collect',
    ):
        assert (collector, collector) not in edges


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
