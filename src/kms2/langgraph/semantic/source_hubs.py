"""LangGraph registration for parallel typed source hub discovery."""

from kms2.node.semantic.source_entity_hub import SourceEntityHubNode
from kms2.node.semantic.source_entity_hub_persistence import (
    SourceEntityHubPersistenceNode,
)
from kms2.node.semantic.source_event_hub import SourceEventHubNode
from kms2.node.semantic.source_event_hub_persistence import (
    SourceEventHubPersistenceNode,
)
from kms2.node.semantic.source_hub_join import SourceHubJoinNode
from kms2.node.semantic.source_predicate_hub import SourcePredicateHubNode
from kms2.node.semantic.source_predicate_hub_persistence import (
    SourcePredicateHubPersistenceNode,
)
from kms2.node.semantic.source_procedure_hub import SourceProcedureHubNode
from kms2.node.semantic.source_procedure_hub_persistence import (
    SourceProcedureHubPersistenceNode,
)
from kms2.node.semantic.source_statement_hub import SourceStatementHubNode
from kms2.node.semantic.source_statement_hub_persistence import (
    SourceStatementHubPersistenceNode,
)
from langgraph.graph import END, StateGraph


def add_source_hub_phase(
    graph: StateGraph,
    entity: SourceEntityHubNode,
    event: SourceEventHubNode,
    predicate: SourcePredicateHubNode,
    statement: SourceStatementHubNode,
    procedure: SourceProcedureHubNode,
    entity_persistence: SourceEntityHubPersistenceNode,
    event_persistence: SourceEventHubPersistenceNode,
    predicate_persistence: SourcePredicateHubPersistenceNode,
    statement_persistence: SourceStatementHubPersistenceNode,
    procedure_persistence: SourceProcedureHubPersistenceNode,
    join: SourceHubJoinNode,
) -> None:
    """Register independent typed hub branches and their final join."""
    graph.add_node('source_entity_hub', entity.run)
    graph.add_node('source_event_hub', event.run)
    graph.add_node('source_predicate_hub', predicate.run)
    graph.add_node('source_statement_hub', statement.run)
    graph.add_node('source_procedure_hub', procedure.run)
    graph.add_node('source_entity_hub_persistence', entity_persistence.run)
    graph.add_node('source_event_hub_persistence', event_persistence.run)
    graph.add_node(
        'source_predicate_hub_persistence', predicate_persistence.run
    )
    graph.add_node(
        'source_statement_hub_persistence', statement_persistence.run
    )
    graph.add_node(
        'source_procedure_hub_persistence', procedure_persistence.run
    )
    graph.add_node('source_hub_join', join.run)
    for kind in ('entity', 'event', 'predicate', 'statement', 'procedure'):
        graph.add_edge(f'source_{kind}_persistence', f'source_{kind}_hub')
        graph.add_edge(f'source_{kind}_hub', f'source_{kind}_hub_persistence')
    graph.add_edge(
        [
            'source_entity_hub_persistence',
            'source_event_hub_persistence',
            'source_predicate_hub_persistence',
            'source_statement_hub_persistence',
            'source_procedure_hub_persistence',
        ],
        'source_hub_join',
    )
    graph.add_edge('source_hub_join', END)
