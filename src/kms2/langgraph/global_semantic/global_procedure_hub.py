"""LangGraph registration for global procedure hub discovery."""

from langgraph.graph import StateGraph

from kms2.node.global_semantic.global_procedure_hub import (
    GlobalProcedureHubNode,
)
from kms2.node.global_semantic.global_procedure_hub_persistence import (
    GlobalProcedureHubPersistenceNode,
)


def add_global_procedure_hub_phase(
    graph: StateGraph,
    procedure: GlobalProcedureHubNode,
    persistence: GlobalProcedureHubPersistenceNode,
) -> None:
    """Register global procedure hub discovery and persistence."""
    graph.add_node('global_procedure_hub_load', procedure.load_candidates)
    graph.add_node(
        'global_procedure_hub_rerank_worker', procedure.rerank_worker
    )
    graph.add_node(
        'global_procedure_hub_rerank_collect', procedure.collect_rerank
    )
    graph.add_node('global_procedure_hub_judge_worker', procedure.judge_worker)
    graph.add_node(
        'global_procedure_hub_judge_collect', procedure.collect_judge
    )
    graph.add_node(
        'global_procedure_hub_communities', procedure.detect_communities
    )
    graph.add_node(
        'global_procedure_hub_synthesis_worker', procedure.synthesis_worker
    )
    graph.add_node(
        'global_procedure_hub_synthesis_collect', procedure.collect_synthesis
    )
    graph.add_node('global_procedure_hub_embedding', procedure.embed)
    graph.add_node('global_procedure_hub_persistence', persistence.run)

    graph.add_conditional_edges(
        'global_procedure_hub_load',
        procedure.dispatch_rerank,
        [
            'global_procedure_hub_rerank_worker',
            'global_procedure_hub_rerank_collect',
        ],
    )
    graph.add_edge(
        'global_procedure_hub_rerank_worker',
        'global_procedure_hub_rerank_collect',
    )
    graph.add_conditional_edges(
        'global_procedure_hub_rerank_collect',
        procedure.dispatch_judge,
        [
            'global_procedure_hub_judge_worker',
            'global_procedure_hub_judge_collect',
        ],
    )
    graph.add_edge(
        'global_procedure_hub_judge_worker',
        'global_procedure_hub_judge_collect',
    )
    graph.add_edge(
        'global_procedure_hub_judge_collect',
        'global_procedure_hub_communities',
    )
    graph.add_conditional_edges(
        'global_procedure_hub_communities',
        procedure.dispatch_synthesis,
        [
            'global_procedure_hub_synthesis_worker',
            'global_procedure_hub_synthesis_collect',
        ],
    )
    graph.add_edge(
        'global_procedure_hub_synthesis_worker',
        'global_procedure_hub_synthesis_collect',
    )
    graph.add_edge(
        'global_procedure_hub_synthesis_collect',
        'global_procedure_hub_embedding',
    )
    graph.add_edge(
        'global_procedure_hub_embedding', 'global_procedure_hub_persistence'
    )
