"""Register the source entity hub phase and its iterative worker subgraph."""

from langgraph.graph import StateGraph

from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.node.source_semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)


def add_source_entity_hub_phase(
    graph: StateGraph,
    entity: SourceEntityHubNode,
    persistence: SourceEntityHubPersistenceNode,
) -> None:
    """Register load and persistence boundaries around source-local reconciliation."""
    loop = StateGraph(SourceSemanticState)
    loop.add_node('source_entity_hub_load', entity.load_candidates)
    loop.add_node('source_entity_hub_rerank_worker', entity.rerank_worker)
    loop.add_node('source_entity_hub_rerank_collect', entity.collect_rerank)
    loop.add_node('source_entity_hub_judge_worker', entity.judge_worker)
    loop.add_node('source_entity_hub_judge_collect', entity.collect_judge)
    loop.add_node('source_entity_hub_merge_groups', entity.merge_groups)
    loop.add_node('source_entity_hub_prepare', entity.prepare_hubs)
    loop.add_conditional_edges(
        'source_entity_hub_load',
        entity.dispatch_rerank,
        ['source_entity_hub_rerank_worker', 'source_entity_hub_rerank_collect'],
    )
    loop.add_edge(
        'source_entity_hub_rerank_worker', 'source_entity_hub_rerank_collect'
    )
    loop.add_conditional_edges(
        'source_entity_hub_rerank_collect',
        entity.dispatch_judge,
        ['source_entity_hub_judge_worker', 'source_entity_hub_judge_collect'],
    )
    loop.add_edge(
        'source_entity_hub_judge_worker', 'source_entity_hub_judge_collect'
    )
    loop.add_edge(
        'source_entity_hub_judge_collect', 'source_entity_hub_merge_groups'
    )
    loop.add_conditional_edges(
        'source_entity_hub_merge_groups',
        entity.repeat_round,
        ['source_entity_hub_load', 'source_entity_hub_prepare'],
    )
    loop.set_entry_point('source_entity_hub_load')
    loop.set_finish_point('source_entity_hub_prepare')
    compiled_loop = loop.compile()

    async def reconcile(state: SourceSemanticState) -> dict[str, object]:
        """Initialize members, run bounded-to-data-size rounds, return hub outputs."""
        initial = await entity.initialize(state)
        members = initial['source_entity_hub_members']
        initial_state = SourceSemanticState(
            source_uuid=state.source_uuid,
            **initial,
        )
        result = await compiled_loop.ainvoke(
            initial_state.model_dump(),
            config={'recursion_limit': 8 * len(members) + 16},
        )
        return {
            'source_entity_hubs': result['source_entity_hubs'],
            'source_entity_hub_memberships': result[
                'source_entity_hub_memberships'
            ],
            'source_entity_hub_count': len(result['source_entity_hubs']),
        }

    graph.add_node('source_entity_hub_load', reconcile)
    graph.add_node('source_entity_hub_persistence', persistence.run)
    graph.add_edge('source_entity_hub_load', 'source_entity_hub_persistence')
