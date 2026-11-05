"""Bounded iterative reconciliation of source-local entity hubs."""

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.node.source_semantic.source_entity_hub import SourceEntityHubNode


class SourceEntityHubGraph:
    """Compile and run the entity hub worker loop."""

    def __init__(self, entity: SourceEntityHubNode) -> None:
        graph = StateGraph(SourceSemanticState)
        graph.add_node('source_entity_hub_load', entity.load_candidates)
        graph.add_node('source_entity_hub_rerank_worker', entity.rerank_worker)
        graph.add_node(
            'source_entity_hub_rerank_collect', entity.collect_rerank
        )
        graph.add_node('source_entity_hub_judge_worker', entity.judge_worker)
        graph.add_node('source_entity_hub_judge_collect', entity.collect_judge)
        graph.add_node('source_entity_hub_merge_groups', entity.merge_groups)
        graph.add_node('source_entity_hub_prepare', entity.prepare_hubs)
        graph.add_conditional_edges(
            'source_entity_hub_load',
            entity.dispatch_rerank,
            [
                'source_entity_hub_rerank_worker',
                'source_entity_hub_rerank_collect',
            ],
        )
        graph.add_edge(
            'source_entity_hub_rerank_worker',
            'source_entity_hub_rerank_collect',
        )
        graph.add_conditional_edges(
            'source_entity_hub_rerank_collect',
            entity.dispatch_judge,
            [
                'source_entity_hub_judge_worker',
                'source_entity_hub_judge_collect',
            ],
        )
        graph.add_edge(
            'source_entity_hub_judge_worker', 'source_entity_hub_judge_collect'
        )
        graph.add_edge(
            'source_entity_hub_judge_collect', 'source_entity_hub_merge_groups'
        )
        graph.add_conditional_edges(
            'source_entity_hub_merge_groups',
            entity.repeat_round,
            ['source_entity_hub_load', 'source_entity_hub_prepare'],
        )
        graph.set_entry_point('source_entity_hub_load')
        graph.set_finish_point('source_entity_hub_prepare')
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: SourceSemanticState) -> SourceSemanticState:
        result = await self._graph.ainvoke(
            state,
            config={
                'recursion_limit': 8 * len(state.source_entity_hub_members) + 16
            },
        )
        return SourceSemanticState.model_validate(result)
