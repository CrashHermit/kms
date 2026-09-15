"""Complete semantic LangGraph assembly for KMS2."""

from kms2.langgraph.semantic.facts import add_fact_phase
from kms2.langgraph.semantic.state import SemanticState
from kms2.langgraph.semantic.triplets import add_triplet_phase
from kms2.node.semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.semantic.source_entity_embedding import SourceEntityEmbeddingNode
from kms2.node.semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms2.node.semantic.source_event_description import (
    SourceEventDescriptionNode,
)
from kms2.node.semantic.source_event_description_load import (
    SourceEventDescriptionLoadNode,
)
from kms2.node.semantic.source_event_embedding import SourceEventEmbeddingNode
from kms2.node.semantic.source_event_persistence import (
    SourceEventPersistenceNode,
)
from kms2.node.semantic.source_predicate_description import (
    SourcePredicateDescriptionNode,
)
from kms2.node.semantic.source_predicate_description_load import (
    SourcePredicateDescriptionLoadNode,
)
from kms2.node.semantic.source_predicate_embedding import (
    SourcePredicateEmbeddingNode,
)
from kms2.node.semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from kms2.node.semantic.triplet import (
    FactExtractionNode,
    TripletDecompositionNode,
)
from kms2.node.semantic.triplet_load import TripletSourceLoadNode
from kms2.node.semantic.triplet_persistence import TripletPersistenceNode
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph


class SemanticGraph:
    """Compile raw extraction and typed descriptions as one semantic graph."""

    def __init__(
        self,
        triplet_source_load: TripletSourceLoadNode,
        fact_extraction: FactExtractionNode,
        triplet_decomposition: TripletDecompositionNode,
        triplet_persistence: TripletPersistenceNode,
        source_entity_description_load: SourceEntityDescriptionLoadNode,
        source_entity_description: SourceEntityDescriptionNode,
        source_entity_embedding: SourceEntityEmbeddingNode,
        source_entity_persistence: SourceEntityPersistenceNode,
        source_event_description_load: SourceEventDescriptionLoadNode,
        source_event_description: SourceEventDescriptionNode,
        source_event_embedding: SourceEventEmbeddingNode,
        source_event_persistence: SourceEventPersistenceNode,
        source_predicate_description_load: SourcePredicateDescriptionLoadNode,
        source_predicate_description: SourcePredicateDescriptionNode,
        source_predicate_embedding: SourcePredicateEmbeddingNode,
        source_predicate_persistence: SourcePredicatePersistenceNode,
    ) -> None:
        self.graph = StateGraph(SemanticState)
        self.triplet_source_load = triplet_source_load
        self.fact_extraction = fact_extraction
        self.triplet_decomposition = triplet_decomposition
        self.triplet_persistence = triplet_persistence
        self.source_entity_description_load = source_entity_description_load
        self.source_entity_description = source_entity_description
        self.source_entity_embedding = source_entity_embedding
        self.source_entity_persistence = source_entity_persistence
        self.source_event_description_load = source_event_description_load
        self.source_event_description = source_event_description
        self.source_event_embedding = source_event_embedding
        self.source_event_persistence = source_event_persistence
        self.source_predicate_description_load = (
            source_predicate_description_load
        )
        self.source_predicate_description = source_predicate_description
        self.source_predicate_embedding = source_predicate_embedding
        self.source_predicate_persistence = source_predicate_persistence

    def build_graph(self, *, raw_only: bool = False) -> CompiledStateGraph:
        """Compile the complete graph or its explicit raw-only prefix."""
        self.graph.add_node('triplet_source_load', self.triplet_source_load.run)
        self.graph.add_edge(START, 'triplet_source_load')
        add_fact_phase(self.graph, self.fact_extraction)
        add_triplet_phase(self.graph, self.triplet_decomposition)
        self.graph.add_node('triplet_persistence', self.triplet_persistence.run)
        self.graph.add_edge('triplet_collect', 'triplet_persistence')

        if raw_only:
            self.graph.add_edge('triplet_persistence', END)
            return self.graph.compile()

        self.graph.add_node(
            'source_entity_description_load',
            self.source_entity_description_load.run,
        )
        self.graph.add_node(
            'source_entity_description_worker',
            self.source_entity_description.worker,
        )
        self.graph.add_node(
            'source_entity_description_collect',
            self.source_entity_description.collect,
        )
        self.graph.add_node(
            'source_entity_embedding_worker',
            self.source_entity_embedding.worker,
        )
        self.graph.add_node(
            'source_entity_embedding_collect',
            self.source_entity_embedding.collect,
        )
        self.graph.add_node(
            'source_entity_persistence',
            self.source_entity_persistence.run,
        )
        self.graph.add_node(
            'source_event_description_load',
            self.source_event_description_load.run,
        )
        self.graph.add_node(
            'source_event_description_worker',
            self.source_event_description.worker,
        )
        self.graph.add_node(
            'source_event_description_collect',
            self.source_event_description.collect,
        )
        self.graph.add_node(
            'source_event_embedding_worker', self.source_event_embedding.worker
        )
        self.graph.add_node(
            'source_event_embedding_collect',
            self.source_event_embedding.collect,
        )
        self.graph.add_node(
            'source_event_persistence',
            self.source_event_persistence.run,
        )
        self.graph.add_node(
            'source_predicate_description_load',
            self.source_predicate_description_load.run,
        )
        self.graph.add_node(
            'source_predicate_description_worker',
            self.source_predicate_description.worker,
        )
        self.graph.add_node(
            'source_predicate_description_collect',
            self.source_predicate_description.collect,
        )
        self.graph.add_node(
            'source_predicate_embedding_worker',
            self.source_predicate_embedding.worker,
        )
        self.graph.add_node(
            'source_predicate_embedding_collect',
            self.source_predicate_embedding.collect,
        )
        self.graph.add_node(
            'source_predicate_persistence',
            self.source_predicate_persistence.run,
        )

        self.graph.add_edge(
            'triplet_persistence', 'source_entity_description_load'
        )
        self.graph.add_conditional_edges(
            'source_entity_description_load',
            self.source_entity_description.dispatch,
            [
                'source_entity_description_worker',
                'source_entity_description_collect',
            ],
        )
        self.graph.add_edge(
            'source_entity_description_worker',
            'source_entity_description_collect',
        )
        self.graph.add_conditional_edges(
            'source_entity_description_collect',
            self.source_entity_embedding.dispatch,
            [
                'source_entity_embedding_worker',
                'source_entity_embedding_collect',
            ],
        )
        self.graph.add_edge(
            'source_entity_embedding_worker', 'source_entity_embedding_collect'
        )
        self.graph.add_edge(
            'source_entity_embedding_collect',
            'source_entity_persistence',
        )
        self.graph.add_edge(
            'source_entity_persistence',
            'source_event_description_load',
        )
        self.graph.add_conditional_edges(
            'source_event_description_load',
            self.source_event_description.dispatch,
            [
                'source_event_description_worker',
                'source_event_description_collect',
            ],
        )
        self.graph.add_edge(
            'source_event_description_worker',
            'source_event_description_collect',
        )
        self.graph.add_conditional_edges(
            'source_event_description_collect',
            self.source_event_embedding.dispatch,
            ['source_event_embedding_worker', 'source_event_embedding_collect'],
        )
        self.graph.add_edge(
            'source_event_embedding_worker', 'source_event_embedding_collect'
        )
        self.graph.add_edge(
            'source_event_embedding_collect',
            'source_event_persistence',
        )
        self.graph.add_edge(
            'source_event_persistence',
            'source_predicate_description_load',
        )
        self.graph.add_conditional_edges(
            'source_predicate_description_load',
            self.source_predicate_description.dispatch,
            [
                'source_predicate_description_worker',
                'source_predicate_description_collect',
            ],
        )
        self.graph.add_edge(
            'source_predicate_description_worker',
            'source_predicate_description_collect',
        )
        self.graph.add_conditional_edges(
            'source_predicate_description_collect',
            self.source_predicate_embedding.dispatch,
            [
                'source_predicate_embedding_worker',
                'source_predicate_embedding_collect',
            ],
        )
        self.graph.add_edge(
            'source_predicate_embedding_worker',
            'source_predicate_embedding_collect',
        )
        self.graph.add_edge(
            'source_predicate_embedding_collect',
            'source_predicate_persistence',
        )
        self.graph.add_edge('source_predicate_persistence', END)
        return self.graph.compile()
