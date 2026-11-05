"""Source-pedagogical LangGraph assembly for KMS2."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.langgraph.source_pedagogical.exercise_finder import (
    add_exercise_finder_phase,
)
from kms.langgraph.source_pedagogical.exercise_splitter import (
    add_exercise_splitter_phase,
)
from kms.langgraph.source_pedagogical.instruction_finder import (
    add_instruction_finder_phase,
)
from kms.langgraph.source_pedagogical.instruction_governance import (
    add_instruction_governance_phase,
)
from kms.langgraph.source_pedagogical.pedagogical_finder import (
    add_pedagogical_finder_phase,
)
from kms.langgraph.source_pedagogical.source_procedure_reconstruction import (
    add_source_procedure_reconstruction_phase,
)
from kms.langgraph.source_pedagogical.source_statement_reconstruction import (
    add_source_statement_reconstruction_phase,
)
from kms.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms.langgraph.source_pedagogical.statement_procedure import (
    add_statement_procedure_phase,
)
from kms.node.source_pedagogical.embedding import (
    SourcePedagogicalEmbeddingNode,
)
from kms.node.source_pedagogical.exercise_finder import ExerciseFinderNode
from kms.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode
from kms.node.source_pedagogical.instruction_finder import (
    InstructionFinderNode,
)
from kms.node.source_pedagogical.instruction_governance import (
    InstructionGovernanceNode,
)
from kms.node.source_pedagogical.pedagogical_finder import (
    PedagogicalFinderNode,
)
from kms.node.source_pedagogical.persistence import (
    SourcePedagogicalPersistenceNode,
)
from kms.node.source_pedagogical.source_load import (
    SourcePedagogicalSourceLoadNode,
)
from kms.node.source_pedagogical.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)
from kms.node.source_pedagogical.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)
from kms.node.source_pedagogical.statement_procedure import (
    StatementProcedureNode,
)


class SourcePedagogicalGraph:
    """Compile and run the source-pedagogical workflow."""

    def __init__(
        self,
        source_load: SourcePedagogicalSourceLoadNode,
        exercise_splitter: ExerciseSplitterNode,
        instruction_finder: InstructionFinderNode,
        exercise_finder: ExerciseFinderNode,
        pedagogical_finder: PedagogicalFinderNode,
        statement_procedure: StatementProcedureNode,
        instruction_governance: InstructionGovernanceNode,
        source_statement_reconstruction: SourceStatementReconstructionNode,
        source_procedure_reconstruction: SourceProcedureReconstructionNode,
        source_statement_embedding: SourceStatementEmbeddingNode,
        source_procedure_embedding: SourceProcedureEmbeddingNode,
        embedding: SourcePedagogicalEmbeddingNode,
        persistence: SourcePedagogicalPersistenceNode,
    ) -> None:
        graph = StateGraph(SourcePedagogicalState)
        graph.add_node('source_pedagogical_load', source_load.run)
        graph.add_edge(START, 'source_pedagogical_load')
        add_exercise_splitter_phase(graph, exercise_splitter)
        add_instruction_finder_phase(graph, instruction_finder)
        add_exercise_finder_phase(graph, exercise_finder)
        add_pedagogical_finder_phase(graph, pedagogical_finder)
        add_statement_procedure_phase(graph, statement_procedure)
        add_instruction_governance_phase(graph, instruction_governance)
        add_source_statement_reconstruction_phase(
            graph, source_statement_reconstruction
        )
        add_source_procedure_reconstruction_phase(
            graph, source_procedure_reconstruction
        )
        graph.add_node(
            'source_statement_embedding', source_statement_embedding.run
        )
        graph.add_node(
            'source_procedure_embedding', source_procedure_embedding.run
        )
        graph.add_node('source_pedagogical_embedding', embedding.run)
        graph.add_node('source_pedagogical_persistence', persistence.run)
        graph.add_edge('instruction_governance', 'source_pedagogical_embedding')
        graph.add_edge(
            [
                'source_statement_embedding',
                'source_procedure_embedding',
                'source_pedagogical_embedding',
            ],
            'source_pedagogical_persistence',
        )
        graph.add_edge('source_pedagogical_persistence', END)
        self._graph: CompiledStateGraph = graph.compile()

    async def run(
        self, state: SourcePedagogicalState
    ) -> SourcePedagogicalState:
        result = await self._graph.ainvoke(state)
        return SourcePedagogicalState.model_validate(result)
