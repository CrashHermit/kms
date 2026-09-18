"""LangGraph registration for source procedure hub discovery."""

from kms2.node.semantic.source_procedure_hub import SourceProcedureHubNode
from kms2.node.semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from langgraph.graph import StateGraph


def add_source_procedure_hub_phase(
    graph: StateGraph,
    procedure: SourceProcedureHubNode,
    persistence: SourceProcedureHubPersistenceNode,
) -> None:
    """Register source procedure hub discovery, synthesis, and persistence."""
    graph.add_node('source_procedure_hub_load', procedure.load_candidates)
    graph.add_node(
        'source_procedure_hub_rerank_worker', procedure.rerank_worker
    )
    graph.add_node(
        'source_procedure_hub_rerank_collect', procedure.collect_rerank
    )
    graph.add_node('source_procedure_hub_judge_worker', procedure.judge_worker)
    graph.add_node(
        'source_procedure_hub_judge_collect', procedure.collect_judge
    )
    graph.add_node(
        'source_procedure_hub_communities', procedure.detect_communities
    )
    graph.add_node(
        'source_procedure_hub_synthesis_worker', procedure.synthesis_worker
    )
    graph.add_node(
        'source_procedure_hub_synthesis_collect', procedure.collect_synthesis
    )
    graph.add_node('source_procedure_hub_embedding', procedure.embed)
    graph.add_node('source_procedure_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'source_procedure_hub_load',
        procedure.dispatch_rerank,
        [
            'source_procedure_hub_rerank_worker',
            'source_procedure_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'source_procedure_hub_rerank_worker',
        'source_procedure_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'source_procedure_hub_rerank_collect',
        procedure.dispatch_judge,
        [
            'source_procedure_hub_judge_worker',
            'source_procedure_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'source_procedure_hub_judge_worker',
        'source_procedure_hub_judge_collect',
    )
    graph.add_edge(
        'source_procedure_hub_judge_collect',
        'source_procedure_hub_communities',
    )
    graph.add_conditional_edges(
        'source_procedure_hub_communities',
        procedure.dispatch_synthesis,
        [
            'source_procedure_hub_synthesis_worker',
            'source_procedure_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'source_procedure_hub_synthesis_worker',
        'source_procedure_hub_synthesis_collect',
    )
    graph.add_edge(
        'source_procedure_hub_synthesis_collect',
        'source_procedure_hub_embedding',
    )
    graph.add_edge(
        'source_procedure_hub_embedding', 'source_procedure_hub_persistence'
    )
