"""Source-processing LangGraph assembly for KMS2."""

from kms2.langgraph.source_processing.content_correction import (
    add_content_correction_phase,
)
from kms2.langgraph.source_processing.embedding import add_embedding_phase
from kms2.langgraph.source_processing.exercise_finder import (
    add_exercise_finder_phase,
)
from kms2.langgraph.source_processing.exercise_splitter import (
    add_exercise_splitter_phase,
)
from kms2.langgraph.source_processing.formatting import add_formatter_phase
from kms2.langgraph.source_processing.image_description import (
    add_image_description_phase,
)
from kms2.langgraph.source_processing.image_seam import add_image_seam_phase
from kms2.langgraph.source_processing.instruction_finder import (
    add_instruction_finder_phase,
)
from kms2.langgraph.source_processing.instruction_governance import (
    add_instruction_governance_phase,
)
from kms2.langgraph.source_processing.ocr import add_ocr_phase
from kms2.langgraph.source_processing.pedagogical_finder import (
    add_pedagogical_finder_phase,
)
from kms2.langgraph.source_processing.persistence import add_persistence_phase
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.langgraph.source_processing.statement_procedure import (
    add_statement_procedure_phase,
)
from kms2.langgraph.source_processing.text_seam import add_text_seam_phase
from kms2.node.source_processing.content_correction import ContentCorrectionNode
from kms2.node.source_processing.embedding import EmbeddingNode
from kms2.node.source_processing.exercise_finder import ExerciseFinderNode
from kms2.node.source_processing.exercise_splitter import ExerciseSplitterNode
from kms2.node.source_processing.formatting import FormattingNode
from kms2.node.source_processing.image_description import ImageDescriptionNode
from kms2.node.source_processing.image_seam import ImageSeamNode
from kms2.node.source_processing.instruction_finder import InstructionFinderNode
from kms2.node.source_processing.instruction_governance import (
    InstructionGovernanceNode,
)
from kms2.node.source_processing.ocr import OCRNode
from kms2.node.source_processing.pedagogical_finder import PedagogicalFinderNode
from kms2.node.source_processing.persistence import SourcePersistenceNode
from kms2.node.source_processing.statement_procedure import (
    StatementProcedureNode,
)
from kms2.node.source_processing.text_seam import TextSeamNode
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph


class SourceProcessingGraph:
    """Compile the source-processing graph from injected node behaviors."""

    def __init__(
        self,
        ocr: OCRNode,
        content_correction: ContentCorrectionNode,
        formatter: FormattingNode,
        text_seam: TextSeamNode,
        image_seam: ImageSeamNode,
        image_description: ImageDescriptionNode,
        exercise_splitter: ExerciseSplitterNode,
        instruction_finder: InstructionFinderNode,
        exercise_finder: ExerciseFinderNode,
        pedagogical_finder: PedagogicalFinderNode,
        statement_procedure: StatementProcedureNode,
        instruction_governance: InstructionGovernanceNode,
        embedding: EmbeddingNode,
        persistence: SourcePersistenceNode,
    ) -> None:
        self.graph = StateGraph(SourceProcessingState)
        self.ocr = ocr
        self.content_correction = content_correction
        self.formatter = formatter
        self.text_seam = text_seam
        self.image_seam = image_seam
        self.image_description = image_description
        self.exercise_splitter = exercise_splitter
        self.instruction_finder = instruction_finder
        self.exercise_finder = exercise_finder
        self.pedagogical_finder = pedagogical_finder
        self.statement_procedure = statement_procedure
        self.instruction_governance = instruction_governance
        self.embedding = embedding
        self.persistence = persistence

    def build_graph(self) -> CompiledStateGraph:
        """Register source phases and compile the graph."""
        add_ocr_phase(self.graph, self.ocr)
        add_content_correction_phase(self.graph, self.content_correction)
        add_formatter_phase(self.graph, self.formatter)
        add_text_seam_phase(self.graph, self.text_seam)
        add_image_seam_phase(self.graph, self.image_seam)
        add_image_description_phase(self.graph, self.image_description)
        add_exercise_splitter_phase(self.graph, self.exercise_splitter)
        add_instruction_finder_phase(self.graph, self.instruction_finder)
        add_exercise_finder_phase(self.graph, self.exercise_finder)
        add_pedagogical_finder_phase(self.graph, self.pedagogical_finder)
        add_statement_procedure_phase(self.graph, self.statement_procedure)
        add_instruction_governance_phase(
            self.graph, self.instruction_governance
        )
        add_embedding_phase(self.graph, self.embedding)
        add_persistence_phase(self.graph, self.persistence)
        return self.graph.compile()
