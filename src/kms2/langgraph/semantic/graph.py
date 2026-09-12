"""Complete semantic LangGraph assembly for KMS2."""

from kms2.langgraph.semantic.facts import add_fact_phase
from kms2.langgraph.semantic.state import SemanticState
from kms2.langgraph.semantic.triplets import add_triplet_phase
from kms2.node.semantic.entity_embedding import EntityEmbeddingNode
from kms2.node.semantic.entity_enrichment import EntityEnrichmentNode
from kms2.node.semantic.entity_enrichment_load import EntityEnrichmentLoadNode
from kms2.node.semantic.entity_enrichment_persistence import (
    EntityEnrichmentPersistenceNode,
)
from kms2.node.semantic.event_embedding import EventEmbeddingNode
from kms2.node.semantic.event_enrichment import EventEnrichmentNode
from kms2.node.semantic.event_enrichment_load import EventEnrichmentLoadNode
from kms2.node.semantic.event_enrichment_persistence import (
    EventEnrichmentPersistenceNode,
)
from kms2.node.semantic.predicate_embedding import PredicateEmbeddingNode
from kms2.node.semantic.predicate_enrichment import PredicateEnrichmentNode
from kms2.node.semantic.predicate_enrichment_load import (
    PredicateEnrichmentLoadNode,
)
from kms2.node.semantic.predicate_enrichment_persistence import (
    PredicateEnrichmentPersistenceNode,
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
    """Compile raw extraction and typed enrichment as one semantic graph."""

    def __init__(
        self,
        triplet_source_load: TripletSourceLoadNode,
        fact_extraction: FactExtractionNode,
        triplet_decomposition: TripletDecompositionNode,
        triplet_persistence: TripletPersistenceNode,
        entity_enrichment_load: EntityEnrichmentLoadNode,
        entity_enrichment: EntityEnrichmentNode,
        entity_embedding: EntityEmbeddingNode,
        entity_enrichment_persistence: EntityEnrichmentPersistenceNode,
        event_enrichment_load: EventEnrichmentLoadNode,
        event_enrichment: EventEnrichmentNode,
        event_embedding: EventEmbeddingNode,
        event_enrichment_persistence: EventEnrichmentPersistenceNode,
        predicate_enrichment_load: PredicateEnrichmentLoadNode,
        predicate_enrichment: PredicateEnrichmentNode,
        predicate_embedding: PredicateEmbeddingNode,
        predicate_enrichment_persistence: PredicateEnrichmentPersistenceNode,
    ) -> None:
        self.graph = StateGraph(SemanticState)
        self.triplet_source_load = triplet_source_load
        self.fact_extraction = fact_extraction
        self.triplet_decomposition = triplet_decomposition
        self.triplet_persistence = triplet_persistence
        self.entity_enrichment_load = entity_enrichment_load
        self.entity_enrichment = entity_enrichment
        self.entity_embedding = entity_embedding
        self.entity_enrichment_persistence = entity_enrichment_persistence
        self.event_enrichment_load = event_enrichment_load
        self.event_enrichment = event_enrichment
        self.event_embedding = event_embedding
        self.event_enrichment_persistence = event_enrichment_persistence
        self.predicate_enrichment_load = predicate_enrichment_load
        self.predicate_enrichment = predicate_enrichment
        self.predicate_embedding = predicate_embedding
        self.predicate_enrichment_persistence = predicate_enrichment_persistence

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
            'entity_enrichment_load', self.entity_enrichment_load.run
        )
        self.graph.add_node(
            'entity_enrichment_worker', self.entity_enrichment.worker
        )
        self.graph.add_node(
            'entity_enrichment_collect', self.entity_enrichment.collect
        )
        self.graph.add_node(
            'entity_embedding_worker', self.entity_embedding.worker
        )
        self.graph.add_node(
            'entity_embedding_collect', self.entity_embedding.collect
        )
        self.graph.add_node(
            'entity_enrichment_persistence',
            self.entity_enrichment_persistence.run,
        )
        self.graph.add_node(
            'event_enrichment_load', self.event_enrichment_load.run
        )
        self.graph.add_node(
            'event_enrichment_worker', self.event_enrichment.worker
        )
        self.graph.add_node(
            'event_enrichment_collect', self.event_enrichment.collect
        )
        self.graph.add_node(
            'event_embedding_worker', self.event_embedding.worker
        )
        self.graph.add_node(
            'event_embedding_collect', self.event_embedding.collect
        )
        self.graph.add_node(
            'event_enrichment_persistence',
            self.event_enrichment_persistence.run,
        )
        self.graph.add_node(
            'predicate_enrichment_load', self.predicate_enrichment_load.run
        )
        self.graph.add_node(
            'predicate_enrichment_worker', self.predicate_enrichment.worker
        )
        self.graph.add_node(
            'predicate_enrichment_collect', self.predicate_enrichment.collect
        )
        self.graph.add_node(
            'predicate_embedding_worker', self.predicate_embedding.worker
        )
        self.graph.add_node(
            'predicate_embedding_collect', self.predicate_embedding.collect
        )
        self.graph.add_node(
            'predicate_enrichment_persistence',
            self.predicate_enrichment_persistence.run,
        )

        self.graph.add_edge('triplet_persistence', 'entity_enrichment_load')
        self.graph.add_conditional_edges(
            'entity_enrichment_load',
            self.entity_enrichment.dispatch,
            ['entity_enrichment_worker', 'entity_enrichment_collect'],
        )
        self.graph.add_edge(
            'entity_enrichment_worker', 'entity_enrichment_collect'
        )
        self.graph.add_conditional_edges(
            'entity_enrichment_collect',
            self.entity_embedding.dispatch,
            ['entity_embedding_worker', 'entity_embedding_collect'],
        )
        self.graph.add_edge(
            'entity_embedding_worker', 'entity_embedding_collect'
        )
        self.graph.add_edge(
            'entity_embedding_collect', 'entity_enrichment_persistence'
        )
        self.graph.add_edge(
            'entity_enrichment_persistence', 'event_enrichment_load'
        )
        self.graph.add_conditional_edges(
            'event_enrichment_load',
            self.event_enrichment.dispatch,
            ['event_enrichment_worker', 'event_enrichment_collect'],
        )
        self.graph.add_edge(
            'event_enrichment_worker', 'event_enrichment_collect'
        )
        self.graph.add_conditional_edges(
            'event_enrichment_collect',
            self.event_embedding.dispatch,
            ['event_embedding_worker', 'event_embedding_collect'],
        )
        self.graph.add_edge('event_embedding_worker', 'event_embedding_collect')
        self.graph.add_edge(
            'event_embedding_collect', 'event_enrichment_persistence'
        )
        self.graph.add_edge(
            'event_enrichment_persistence', 'predicate_enrichment_load'
        )
        self.graph.add_conditional_edges(
            'predicate_enrichment_load',
            self.predicate_enrichment.dispatch,
            ['predicate_enrichment_worker', 'predicate_enrichment_collect'],
        )
        self.graph.add_edge(
            'predicate_enrichment_worker', 'predicate_enrichment_collect'
        )
        self.graph.add_conditional_edges(
            'predicate_enrichment_collect',
            self.predicate_embedding.dispatch,
            ['predicate_embedding_worker', 'predicate_embedding_collect'],
        )
        self.graph.add_edge(
            'predicate_embedding_worker', 'predicate_embedding_collect'
        )
        self.graph.add_edge(
            'predicate_embedding_collect', 'predicate_enrichment_persistence'
        )
        self.graph.add_edge('predicate_enrichment_persistence', END)
        return self.graph.compile()
