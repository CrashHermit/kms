"""Global semantic-hub LangGraph assembly for KMS2."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.langgraph.global_semantic.global_entity_hub import (
    add_global_entity_hub_phase,
)
from kms2.langgraph.global_semantic.global_event_hub import (
    add_global_event_hub_phase,
)
from kms2.langgraph.global_semantic.global_predicate_hub import (
    add_global_predicate_hub_phase,
)
from kms2.langgraph.global_semantic.global_procedure_hub import (
    add_global_procedure_hub_phase,
)
from kms2.langgraph.global_semantic.global_statement_hub import (
    add_global_statement_hub_phase,
)
from kms2.langgraph.global_semantic.global_triplets import (
    add_global_triplet_phase,
)
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.node.global_semantic.global_entity_hub import GlobalEntityHubNode
from kms2.node.global_semantic.global_entity_hub_persistence import (
    GlobalEntityHubPersistenceNode,
)
from kms2.node.global_semantic.global_event_hub import GlobalEventHubNode
from kms2.node.global_semantic.global_event_hub_persistence import (
    GlobalEventHubPersistenceNode,
)
from kms2.node.global_semantic.global_predicate_hub import (
    GlobalPredicateHubNode,
)
from kms2.node.global_semantic.global_predicate_hub_persistence import (
    GlobalPredicateHubPersistenceNode,
)
from kms2.node.global_semantic.global_procedure_hub import (
    GlobalProcedureHubNode,
)
from kms2.node.global_semantic.global_procedure_hub_persistence import (
    GlobalProcedureHubPersistenceNode,
)
from kms2.node.global_semantic.global_statement_hub import (
    GlobalStatementHubNode,
)
from kms2.node.global_semantic.global_statement_hub_persistence import (
    GlobalStatementHubPersistenceNode,
)
from kms2.node.global_semantic.global_triplet_hub import GlobalTripletHubNode
from kms2.node.global_semantic.global_triplet_hub_persistence import (
    GlobalTripletHubPersistenceNode,
)
from kms2.node.global_semantic.global_triplet_projection import (
    GlobalTripletProjectionNode,
)


class GlobalSemanticGraph:
    """Compile global semantic consolidation as one graph."""

    def __init__(
        self,
        global_entity_hub: GlobalEntityHubNode,
        global_entity_hub_persistence: GlobalEntityHubPersistenceNode,
        global_event_hub: GlobalEventHubNode,
        global_event_hub_persistence: GlobalEventHubPersistenceNode,
        global_predicate_hub: GlobalPredicateHubNode,
        global_predicate_hub_persistence: GlobalPredicateHubPersistenceNode,
        global_triplet_projection: GlobalTripletProjectionNode,
        global_triplet_hub: GlobalTripletHubNode,
        global_triplet_hub_persistence: GlobalTripletHubPersistenceNode,
        global_statement_hub: GlobalStatementHubNode,
        global_statement_hub_persistence: GlobalStatementHubPersistenceNode,
        global_procedure_hub: GlobalProcedureHubNode,
        global_procedure_hub_persistence: GlobalProcedureHubPersistenceNode,
    ) -> None:
        self.graph = StateGraph(GlobalSemanticState)
        self.global_entity_hub = global_entity_hub
        self.global_entity_hub_persistence = global_entity_hub_persistence
        self.global_event_hub = global_event_hub
        self.global_event_hub_persistence = global_event_hub_persistence
        self.global_predicate_hub = global_predicate_hub
        self.global_triplet_projection = global_triplet_projection
        self.global_triplet_hub = global_triplet_hub
        self.global_triplet_hub_persistence = global_triplet_hub_persistence
        self.global_predicate_hub_persistence = global_predicate_hub_persistence
        self.global_statement_hub = global_statement_hub
        self.global_statement_hub_persistence = global_statement_hub_persistence
        self.global_procedure_hub = global_procedure_hub
        self.global_procedure_hub_persistence = global_procedure_hub_persistence

    def build_graph(self) -> CompiledStateGraph:
        """Compile all global source-hub phases in source semantic order."""
        add_global_entity_hub_phase(
            self.graph,
            self.global_entity_hub,
            self.global_entity_hub_persistence,
        )
        add_global_event_hub_phase(
            self.graph,
            self.global_event_hub,
            self.global_event_hub_persistence,
        )
        add_global_predicate_hub_phase(
            self.graph,
            self.global_predicate_hub,
            self.global_predicate_hub_persistence,
        )
        add_global_triplet_phase(
            self.graph,
            self.global_triplet_projection,
            self.global_triplet_hub,
            self.global_triplet_hub_persistence,
        )
        add_global_statement_hub_phase(
            self.graph,
            self.global_statement_hub,
            self.global_statement_hub_persistence,
        )
        add_global_procedure_hub_phase(
            self.graph,
            self.global_procedure_hub,
            self.global_procedure_hub_persistence,
        )
        self.graph.add_edge(START, 'global_entity_hub_load')
        self.graph.add_edge(
            'global_entity_hub_persistence', 'global_event_hub_load'
        )
        self.graph.add_edge(
            'global_event_hub_persistence', 'global_predicate_hub_load'
        )
        self.graph.add_edge(
            'global_predicate_hub_persistence', 'global_triplet_projection'
        )
        self.graph.add_edge(
            'global_triplet_hub_persistence', 'global_statement_hub_load'
        )
        self.graph.add_edge(
            'global_statement_hub_persistence', 'global_procedure_hub_load'
        )
        self.graph.add_edge('global_procedure_hub_persistence', END)
        return self.graph.compile()
