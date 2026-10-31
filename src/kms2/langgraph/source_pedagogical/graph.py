"""Source-pedagogical LangGraph assembly for KMS2."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms2.langgraph.source_pedagogical.exercise_finder import (
    add_exercise_finder_phase,
)
from kms2.langgraph.source_pedagogical.exercise_splitter import (
    add_exercise_splitter_phase,
)
from kms2.langgraph.source_pedagogical.instruction_finder import (
    add_instruction_finder_phase,
)
from kms2.langgraph.source_pedagogical.instruction_governance import (
    add_instruction_governance_phase,
)
from kms2.langgraph.source_pedagogical.pedagogical_finder import (
    add_pedagogical_finder_phase,
)
from kms2.langgraph.source_pedagogical.source_procedure_reconstruction import (
    add_source_procedure_reconstruction_phase,
)
from kms2.langgraph.source_pedagogical.source_statement_reconstruction import (
    add_source_statement_reconstruction_phase,
)
from kms2.langgraph.source_pedagogical.state import SourcePedagogicalState
from kms2.langgraph.source_pedagogical.statement_procedure import (
    add_statement_procedure_phase,
)
from kms2.node.source_pedagogical.embedding import (
    SourcePedagogicalEmbeddingNode,
)
from kms2.node.source_pedagogical.exercise_finder import ExerciseFinderNode
from kms2.node.source_pedagogical.exercise_splitter import ExerciseSplitterNode
from kms2.node.source_pedagogical.instruction_finder import (
    InstructionFinderNode,
)
from kms2.node.source_pedagogical.instruction_governance import (
    InstructionGovernanceNode,
)
from kms2.node.source_pedagogical.pedagogical_finder import (
    PedagogicalFinderNode,
)
from kms2.node.source_pedagogical.persistence import (
    SourcePedagogicalPersistenceNode,
)
from kms2.node.source_pedagogical.source_load import (
    SourcePedagogicalSourceLoadNode,
)
from kms2.node.source_pedagogical.source_procedure_embedding import (
    SourceProcedureEmbeddingNode,
)
from kms2.node.source_pedagogical.source_procedure_reconstruction import (
    SourceProcedureReconstructionNode,
)
from kms2.node.source_pedagogical.source_statement_embedding import (
    SourceStatementEmbeddingNode,
)
from kms2.node.source_pedagogical.source_statement_reconstruction import (
    SourceStatementReconstructionNode,
)
from kms2.node.source_pedagogical.statement_procedure import (
    StatementProcedureNode,
)


class SourcePedagogicalGraph:
    """Compile the separately runnable source-pedagogical graph."""

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
        self.graph = StateGraph(SourcePedagogicalState)
        self.source_load = source_load
        self.exercise_splitter = exercise_splitter
        self.instruction_finder = instruction_finder
        self.exercise_finder = exercise_finder
        self.pedagogical_finder = pedagogical_finder
        self.statement_procedure = statement_procedure
        self.instruction_governance = instruction_governance
        self.source_statement_reconstruction = source_statement_reconstruction
        self.source_procedure_reconstruction = source_procedure_reconstruction
        self.source_statement_embedding = source_statement_embedding
        self.source_procedure_embedding = source_procedure_embedding
        self.embedding = embedding
        self.persistence = persistence

    def build_graph(self) -> CompiledStateGraph:
        """Register stage phases and compile the graph."""
        self.graph.add_node('source_pedagogical_load', self.source_load.run)
        self.graph.add_edge(START, 'source_pedagogical_load')
        add_exercise_splitter_phase(self.graph, self.exercise_splitter)
        add_instruction_finder_phase(self.graph, self.instruction_finder)
        add_exercise_finder_phase(self.graph, self.exercise_finder)
        add_pedagogical_finder_phase(self.graph, self.pedagogical_finder)
        add_statement_procedure_phase(self.graph, self.statement_procedure)
        add_instruction_governance_phase(
            self.graph, self.instruction_governance
        )
        add_source_statement_reconstruction_phase(
            self.graph, self.source_statement_reconstruction
        )
        add_source_procedure_reconstruction_phase(
            self.graph, self.source_procedure_reconstruction
        )
        self.graph.add_node(
            'source_statement_embedding', self.source_statement_embedding.run
        )
        self.graph.add_node(
            'source_procedure_embedding', self.source_procedure_embedding.run
        )
        self.graph.add_node('source_pedagogical_embedding', self.embedding.run)
        self.graph.add_node(
            'source_pedagogical_persistence', self.persistence.run
        )
        self.graph.add_edge(
            'instruction_governance', 'source_pedagogical_embedding'
        )
        self.graph.add_edge(
            [
                'source_statement_embedding',
                'source_procedure_embedding',
                'source_pedagogical_embedding',
            ],
            'source_pedagogical_persistence',
        )
        self.graph.add_edge('source_pedagogical_persistence', END)
        return self.graph.compile()
