"""Independent source extraction and local semantic hub graph."""

from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.langgraph.source_semantic.source_entity import (
    add_source_entity_phase,
)
from kms.langgraph.source_semantic.source_entity_description import (
    add_source_entity_description_phase,
)
from kms.langgraph.source_semantic.source_entity_extraction import (
    add_source_entity_extraction_phase,
)
from kms.langgraph.source_semantic.source_entity_hub import (
    add_source_entity_hub_phase,
)
from kms.langgraph.source_semantic.source_procedure_hub import (
    add_source_procedure_hub_phase,
)
from kms.langgraph.source_semantic.source_statement_hub import (
    add_source_statement_hub_phase,
)
from kms.langgraph.source_semantic.state import SourceSemanticState
from kms.node.source_semantic.source_entity_description import (
    SourceEntityDescriptionNode,
)
from kms.node.source_semantic.source_entity_description_load import (
    SourceEntityDescriptionLoadNode,
)
from kms.node.source_semantic.source_entity_embedding import (
    SourceEntityEmbeddingNode,
)
from kms.node.source_semantic.source_entity_extraction import (
    SourceEntityExtractionNode,
)
from kms.node.source_semantic.source_entity_extraction_persistence import (
    SourceEntityExtractionPersistenceNode,
)
from kms.node.source_semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms.node.source_semantic.source_entity_hub_reconciliation import (
    SourceEntityHubReconciliationNode,
)
from kms.node.source_semantic.source_entity_persistence import (
    SourceEntityPersistenceNode,
)
from kms.node.source_semantic.source_load import SourceSemanticSourceLoadNode
from kms.node.source_semantic.source_procedure_hub import (
    SourceProcedureHubNode,
)
from kms.node.source_semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms.node.source_semantic.source_statement_hub import (
    SourceStatementHubNode,
)
from kms.node.source_semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)


class SourceSemanticGraph:
    """Compile and run extraction, entity description, and local hub phases."""

    def __init__(
        self,
        source_semantic_load: SourceSemanticSourceLoadNode,
        source_entity_extraction: SourceEntityExtractionNode,
        source_entity_extraction_persistence: SourceEntityExtractionPersistenceNode,
        source_entity_description_load: SourceEntityDescriptionLoadNode,
        source_entity_description: SourceEntityDescriptionNode,
        source_entity_embedding: SourceEntityEmbeddingNode,
        source_entity_persistence: SourceEntityPersistenceNode,
        source_entity_hub_reconciliation: SourceEntityHubReconciliationNode,
        source_entity_hub_persistence: SourceEntityHubPersistenceNode,
        source_statement_hub: SourceStatementHubNode,
        source_procedure_hub: SourceProcedureHubNode,
        source_statement_hub_persistence: SourceStatementHubPersistenceNode,
        source_procedure_hub_persistence: SourceProcedureHubPersistenceNode,
    ) -> None:
        graph = StateGraph(SourceSemanticState)
        graph.add_node('source_semantic_load', source_semantic_load.run)
        graph.add_edge(START, 'source_semantic_load')
        add_source_entity_extraction_phase(
            graph,
            source_entity_extraction,
            source_entity_extraction_persistence,
        )
        add_source_entity_description_phase(
            graph, source_entity_description_load, source_entity_description
        )
        add_source_entity_phase(
            graph, source_entity_embedding, source_entity_persistence
        )
        add_source_entity_hub_phase(
            graph,
            source_entity_hub_reconciliation,
            source_entity_hub_persistence,
        )
        add_source_statement_hub_phase(
            graph, source_statement_hub, source_statement_hub_persistence
        )
        add_source_procedure_hub_phase(
            graph, source_procedure_hub, source_procedure_hub_persistence
        )
        graph.add_edge(
            'source_entity_hub_persistence', 'source_statement_hub_load'
        )
        graph.add_edge(
            'source_statement_hub_persistence', 'source_procedure_hub_load'
        )
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: SourceSemanticState) -> SourceSemanticState:
        result = await self._graph.ainvoke(state)
        return SourceSemanticState.model_validate(result)
