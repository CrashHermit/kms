"""Independent source extraction and local semantic hub graph."""

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.langgraph.source_semantic.source_entity import (
    add_source_entity_phase,
)
from kms2.langgraph.source_semantic.source_entity_description import (
    add_source_entity_description_phase,
)
from kms2.langgraph.source_semantic.source_entity_extraction import (
    add_source_entity_extraction_phase,
)
from kms2.langgraph.source_semantic.source_entity_hub import (
    add_source_entity_hub_phase,
)
from kms2.langgraph.source_semantic.source_procedure_hub import (
    add_source_procedure_hub_phase,
)
from kms2.langgraph.source_semantic.source_statement_hub import (
    add_source_statement_hub_phase,
)
from kms2.langgraph.source_semantic.state import SourceSemanticState
from kms2.node.source_semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms2.node.source_semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms2.node.source_semantic.source_entity_embedding import (
    SourceEntityEmbeddingNode,
)
from kms2.node.source_semantic.source_entity_extraction import (
    SourceEntityExtractionNode,
)
from kms2.node.source_semantic.source_entity_extraction_persistence import (
    SourceEntityExtractionPersistenceNode,
)
from kms2.node.source_semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms2.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms2.node.source_semantic.source_load import SourceSemanticSourceLoadNode
from kms2.node.source_semantic.source_procedure_hub import (
    SourceProcedureHubNode,
)
from kms2.node.source_semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.source_semantic.source_statement_hub import (
    SourceStatementHubNode,
)
from kms2.node.source_semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)


class SourceSemanticGraph:
    """Compile extraction, entity description, and local hub pipelines."""

    def __init__(
        self,
        source_semantic_load: SourceSemanticSourceLoadNode,
        source_entity_extraction: SourceEntityExtractionNode,
        source_entity_extraction_persistence: SourceEntityExtractionPersistenceNode,
        source_entity_description_load: SourceEntityDescriptionLoadNode,
        source_entity_description: SourceEntityDescriptionNode,
        source_entity_embedding: SourceEntityEmbeddingNode,
        source_entity_persistence: SourceEntityPersistenceNode,
        source_entity_hub: SourceEntityHubNode,
        source_entity_hub_persistence: SourceEntityHubPersistenceNode,
        source_statement_hub: SourceStatementHubNode,
        source_procedure_hub: SourceProcedureHubNode,
        source_statement_hub_persistence: SourceStatementHubPersistenceNode,
        source_procedure_hub_persistence: SourceProcedureHubPersistenceNode,
    ) -> None:
        self.source_entity_hub = source_entity_hub
        self.source_entity_hub_persistence = source_entity_hub_persistence
        self.graph = StateGraph(SourceSemanticState)
        self.source_semantic_load = source_semantic_load
        self.source_entity_extraction = source_entity_extraction
        self.source_entity_extraction_persistence = (
            source_entity_extraction_persistence
        )
        self.source_entity_description_load = source_entity_description_load
        self.source_entity_description = source_entity_description
        self.source_entity_embedding = source_entity_embedding
        self.source_entity_persistence = source_entity_persistence
        self.source_statement_hub = source_statement_hub
        self.source_procedure_hub = source_procedure_hub
        self.source_statement_hub_persistence = source_statement_hub_persistence
        self.source_procedure_hub_persistence = source_procedure_hub_persistence

    def build_graph(self) -> CompiledStateGraph:
        """Join extraction and entity persistence before serial hub processing."""
        graph = self.graph
        graph.add_node('source_semantic_load', self.source_semantic_load.run)
        graph.add_edge(START, 'source_semantic_load')
        add_source_entity_extraction_phase(
            graph,
            self.source_entity_extraction,
            self.source_entity_extraction_persistence,
        )
        add_source_entity_description_phase(
            graph,
            self.source_entity_description_load,
            self.source_entity_description,
        )
        add_source_entity_phase(
            graph,
            self.source_entity_embedding,
            self.source_entity_persistence,
        )
        add_source_entity_hub_phase(
            graph, self.source_entity_hub, self.source_entity_hub_persistence
        )
        add_source_statement_hub_phase(
            graph,
            self.source_statement_hub,
            self.source_statement_hub_persistence,
        )
        add_source_procedure_hub_phase(
            graph,
            self.source_procedure_hub,
            self.source_procedure_hub_persistence,
        )
        graph.add_edge(
            'source_entity_hub_persistence', 'source_statement_hub_load'
        )
        graph.add_edge(
            'source_statement_hub_persistence', 'source_procedure_hub_load'
        )
        return graph.compile()
