import ast
import asyncio
import json
from pathlib import Path

from kms.construction import pedagogical_component_finder, workflow
from kms.core import recording

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
        'event_enrichment',
        'predicate_enrichment',
        'entity_hub_builder',
        'event_hub_builder',
        'predicate_hub_builder',
        'procedure_enrichment',
        'statement_hub_builder',
        'triplet_hub_builder',
        'procedure_hub_builder',
        'text_seam_merger',
        'text_seam_rewriter',
        'image_seam_merger',
        'image_enricher',
    }
    assert expected <= modules.keys()
    assert 'statement_hub_adjudicator' not in modules
    assert 'procedure_hub_adjudicator' not in modules
    assert expected <= set(calls)


def test_workflow_builds_pedagogical_start_and_end_routers(monkeypatch):
    calls = []

    def fake_module_lm(module_name):
        calls.append(module_name)
        return object()

    monkeypatch.setattr(workflow.llm, 'module_lm', fake_module_lm)
    modules = workflow._build_modules(None)

    assert isinstance(
        modules['pedagogical_start_router'],
        pedagogical_component_finder.PedagogicalStartRouter,
    )
    assert isinstance(
        modules['pedagogical_end_router'],
        pedagogical_component_finder.PedagogicalEndRouter,
    )
    assert 'pedagogical_window_readiness_router' not in modules
    assert calls.count('pedagogical_component_finder') == 2


def test_entity_hub_progress_persists_diagnostics(tmp_path):
    diagnostics = {
        'records': 3,
        'remaining_records': 0,
        'embedding_comparisons': 3,
        'ambiguous_candidates': 1,
        'reranker_requests': 1,
        'reranker_candidates': 1,
        'reranker_selected': 1,
        'locked_groups': 2,
        'synthesizer_calls': 2,
        'final_hubs': 2,
    }
    recorder = recording.Recorder('source', output_dir=str(tmp_path))

    async def entity_hub_node(_state):
        return {'entity_hub_diagnostics': diagnostics, 'hubs': []}

    async def statement_hub_node(_state):
        return {'statement_hub_diagnostics': diagnostics, 'hubs': []}

    async def procedure_hub_node(_state):
        return {'procedure_hub_diagnostics': diagnostics, 'hubs': []}

    async def ordinary_node(_state):
        return {'triplets': []}

    asyncio.run(
        workflow._timed_stage(
            'entity_hub_builder',
            entity_hub_node,
            {},
            recorder,
        )
    )
    asyncio.run(
        workflow._timed_stage(
            'statement_hub_builder',
            statement_hub_node,
            {},
            recorder,
        )
    )
    asyncio.run(
        workflow._timed_stage(
            'procedure_hub_builder',
            procedure_hub_node,
            {},
            recorder,
        )
    )
    asyncio.run(
        workflow._timed_stage(
            'triplet_extraction',
            ordinary_node,
            {},
            recorder,
        )
    )

    records = [
        json.loads(line)
        for line in (tmp_path / 'progress.jsonl').read_text().splitlines()
    ]
    assert records[0]['details'] == diagnostics
    assert records[1]['details'] == diagnostics
    assert records[2]['details'] == diagnostics
    assert 'details' not in records[3]


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
    assert 'statement_enrichment' in calls
    assert 'procedure_enrichment' in calls
    assert 'statement_hub_builder' in calls


def test_workflow_uses_one_final_projector():
    source = (MODULE_DIR / 'construction' / 'workflow.py').read_text()
    assert "'final_projector'" in source
    assert 'source_projector' not in source
    assert 'assertion_projector' not in source
    assert 'procedure_materialization_projector' not in source
    assert 'ingestion_persister' not in source
    assert "'final_projector', 'triplet_hub_builder'" in source
    assert "'triplet_hub_builder', END" in source


def test_workflow_has_no_model_switch_nodes(monkeypatch):
    monkeypatch.setattr(workflow.llm, 'module_lm', lambda name: object())
    graph = workflow.build_workflow().get_graph()
    node_names = set(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert not any(name.startswith('switch_to_') for name in node_names)
    assert ('statement_procedure_builder', 'governance_walker') in edges
    assert ('triplet_extraction', 'entity_enrichment') in edges
    assert ('entity_enrichment', 'event_enrichment') in edges
    assert ('event_enrichment', 'predicate_enrichment') in edges
    assert ('predicate_enrichment', 'entity_hub_builder') in edges
    assert ('entity_hub_builder', 'event_hub_builder') in edges
    assert ('event_hub_builder', 'predicate_hub_builder') in edges
    assert ('predicate_hub_builder', 'statement_enrichment') in edges
    assert ('final_projector', 'triplet_hub_builder') in edges
    assert ('triplet_hub_builder', '__end__') in edges
    assert 'triplet_hub_builder' in node_names
    assert 'event_hub_builder' in node_names


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
