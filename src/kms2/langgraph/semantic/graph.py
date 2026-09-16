"""Complete semantic LangGraph assembly for KMS2."""

from kms2.langgraph.semantic.facts import add_fact_phase
from kms2.langgraph.semantic.source_entity import add_source_entity_phase
from kms2.langgraph.semantic.source_entity_description import (
    add_source_entity_description_phase,
)
from kms2.langgraph.semantic.source_event import add_source_event_phase
from kms2.langgraph.semantic.source_event_description import (
    add_source_event_description_phase,
)
from kms2.langgraph.semantic.source_hubs import add_source_hub_phase
from kms2.langgraph.semantic.source_predicate import add_source_predicate_phase
from kms2.langgraph.semantic.source_predicate_description import (
    add_source_predicate_description_phase,
)
from kms2.langgraph.semantic.source_procedure import add_source_procedure_phase
from kms2.langgraph.semantic.source_procedure_description import (
    add_source_procedure_description_phase,
)
from kms2.langgraph.semantic.source_statement import add_source_statement_phase
from kms2.langgraph.semantic.source_statement_description import (
    add_source_statement_description_phase,
)
from kms2.langgraph.semantic.state import SemanticState
from kms2.langgraph.semantic.triplets import add_triplet_phase
from kms2.node.semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.semantic.source_entity_embedding import SourceEntityEmbeddingNode
from kms2.node.semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
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
from kms2.node.semantic.source_event_hub import SourceEventHubNode
from kms2.node.semantic.source_event_hub_persistence import (
    SourceEventHubPersistenceNode,
)
from kms2.node.semantic.source_event_persistence import (
    SourceEventPersistenceNode,
)
from kms2.node.semantic.source_hub_join import SourceHubJoinNode
from kms2.node.semantic.source_predicate_description import (
    SourcePredicateDescriptionNode,
)
from kms2.node.semantic.source_predicate_description_load import (
    SourcePredicateDescriptionLoadNode,
)
from kms2.node.semantic.source_predicate_embedding import (
    SourcePredicateEmbeddingNode,
)
from kms2.node.semantic.source_predicate_hub import SourcePredicateHubNode
from kms2.node.semantic.source_predicate_hub_persistence import (
    SourcePredicateHubPersistenceNode,
)
from kms2.node.semantic.source_predicate_persistence import (
    SourcePredicatePersistenceNode,
)
from kms2.node.semantic.source_procedure_description import (
    SourceProcedureDescriptionNode,
)
from kms2.node.semantic.source_procedure_description_load import (
    SourceProcedureDescriptionLoadNode,
)
from kms2.node.semantic.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.semantic.source_procedure_hub import SourceProcedureHubNode
from kms2.node.semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.semantic.source_procedure_persistence import (
    SourceProcedurePersistenceNode,
)
from kms2.node.semantic.source_statement_description import (
    SourceStatementDescriptionNode,
)
from kms2.node.semantic.source_statement_description_load import (
    SourceStatementDescriptionLoadNode,
)
from kms2.node.semantic.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.semantic.source_statement_hub import SourceStatementHubNode
from kms2.node.semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from kms2.node.semantic.source_statement_persistence import (
    SourceStatementPersistenceNode,
)
from kms2.node.semantic.triplet import (
    FactExtractionNode,
    TripletDecompositionNode,
)
from kms2.node.semantic.triplet_load import TripletSourceLoadNode
from kms2.node.semantic.triplet_persistence import TripletPersistenceNode
from langgraph.graph import START, StateGraph
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
        source_statement_description_load: SourceStatementDescriptionLoadNode,
        source_statement_description: SourceStatementDescriptionNode,
        source_statement_embedding: SourceStatementEmbeddingNode,
        source_statement_persistence: SourceStatementPersistenceNode,
        source_procedure_description_load: SourceProcedureDescriptionLoadNode,
        source_procedure_description: SourceProcedureDescriptionNode,
        source_procedure_embedding: SourceProcedureEmbeddingNode,
        source_procedure_persistence: SourceProcedurePersistenceNode,
        source_entity_hub: SourceEntityHubNode,
        source_event_hub: SourceEventHubNode,
        source_predicate_hub: SourcePredicateHubNode,
        source_statement_hub: SourceStatementHubNode,
        source_procedure_hub: SourceProcedureHubNode,
        source_entity_hub_persistence: SourceEntityHubPersistenceNode,
        source_event_hub_persistence: SourceEventHubPersistenceNode,
        source_predicate_hub_persistence: SourcePredicateHubPersistenceNode,
        source_statement_hub_persistence: SourceStatementHubPersistenceNode,
        source_procedure_hub_persistence: SourceProcedureHubPersistenceNode,
        source_hub_join: SourceHubJoinNode,
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
        self.source_statement_description_load = (
            source_statement_description_load
        )
        self.source_statement_description = source_statement_description
        self.source_statement_embedding = source_statement_embedding
        self.source_statement_persistence = source_statement_persistence
        self.source_procedure_description_load = (
            source_procedure_description_load
        )
        self.source_procedure_description = source_procedure_description
        self.source_procedure_embedding = source_procedure_embedding
        self.source_procedure_persistence = source_procedure_persistence
        self.source_entity_hub = source_entity_hub
        self.source_event_hub = source_event_hub
        self.source_predicate_hub = source_predicate_hub
        self.source_statement_hub = source_statement_hub
        self.source_procedure_hub = source_procedure_hub
        self.source_entity_hub_persistence = source_entity_hub_persistence
        self.source_event_hub_persistence = source_event_hub_persistence
        self.source_predicate_hub_persistence = source_predicate_hub_persistence
        self.source_statement_hub_persistence = source_statement_hub_persistence
        self.source_procedure_hub_persistence = source_procedure_hub_persistence
        self.source_hub_join = source_hub_join

    def build_graph(self) -> CompiledStateGraph:
        """Compile the complete semantic graph."""
        self.graph.add_node('triplet_source_load', self.triplet_source_load.run)
        self.graph.add_edge(START, 'triplet_source_load')
        add_fact_phase(self.graph, self.fact_extraction)
        add_triplet_phase(self.graph, self.triplet_decomposition)
        self.graph.add_node('triplet_persistence', self.triplet_persistence.run)
        self.graph.add_edge('triplet_collect', 'triplet_persistence')

        add_source_entity_description_phase(
            self.graph,
            self.source_entity_description_load,
            self.source_entity_description,
        )
        add_source_entity_phase(
            self.graph,
            self.source_entity_embedding,
            self.source_entity_persistence,
        )
        add_source_event_description_phase(
            self.graph,
            self.source_event_description_load,
            self.source_event_description,
        )
        add_source_event_phase(
            self.graph,
            self.source_event_embedding,
            self.source_event_persistence,
        )
        add_source_predicate_description_phase(
            self.graph,
            self.source_predicate_description_load,
            self.source_predicate_description,
        )
        add_source_predicate_phase(
            self.graph,
            self.source_predicate_embedding,
            self.source_predicate_persistence,
        )
        add_source_statement_description_phase(
            self.graph,
            self.source_statement_description_load,
            self.source_statement_description,
        )
        add_source_statement_phase(
            self.graph,
            self.source_statement_embedding,
            self.source_statement_persistence,
        )
        add_source_procedure_description_phase(
            self.graph,
            self.source_procedure_description_load,
            self.source_procedure_description,
        )
        add_source_procedure_phase(
            self.graph,
            self.source_procedure_embedding,
            self.source_procedure_persistence,
        )
        add_source_hub_phase(
            self.graph,
            self.source_entity_hub,
            self.source_event_hub,
            self.source_predicate_hub,
            self.source_statement_hub,
            self.source_procedure_hub,
            self.source_entity_hub_persistence,
            self.source_event_hub_persistence,
            self.source_predicate_hub_persistence,
            self.source_statement_hub_persistence,
            self.source_procedure_hub_persistence,
            self.source_hub_join,
        )
        return self.graph.compile()
