"""Merged source-processing and pedagogical LangGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.exercise_finder.node import ExerciseFinderNode
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode
from kms.source.processing.formatting.node import FormattingNode
from kms.source.processing.image_filter.node import ImageFilterNode
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.instruction_finder.node import InstructionFinderNode
from kms.source.processing.instruction_governance.node import (
    InstructionGovernanceNode,
)
from kms.source.processing.ocr import OCRNode
from kms.source.processing.pedagogical_finder.node import PedagogicalFinderNode
from kms.source.processing.procedure_reconstruction.embedding import (
    SourceProcedureEmbeddingNode,
)
from kms.source.processing.procedure_reconstruction.node import (
    SourceProcedureReconstructionNode,
)
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.statement_procedure.node import (
    StatementProcedurePartitionNode,
)
from kms.source.processing.statement_reconstruction.embedding import (
    SourceStatementEmbeddingNode,
)
from kms.source.processing.statement_reconstruction.node import (
    SourceStatementReconstructionNode,
)
from kms.source.processing.text_seam.node import TextSeamNode


class SourceProcessingGraph:
    """Compile the one processing-only graph for a complete source snapshot."""

    def __init__(
        self,
        ocr: OCRNode,
        content_correction: ContentCorrectionNode,
        formatter: FormattingNode,
        text_seam: TextSeamNode,
        image_seam: ImageSeamNode,
        image_filter: ImageFilterNode,
        exercise_splitter: ExerciseSplitterNode,
        instruction_finder: InstructionFinderNode,
        exercise_finder: ExerciseFinderNode,
        pedagogical_finder: PedagogicalFinderNode,
        statement_procedure: StatementProcedurePartitionNode,
        instruction_governance: InstructionGovernanceNode,
        source_statement_reconstruction: SourceStatementReconstructionNode,
        source_procedure_reconstruction: SourceProcedureReconstructionNode,
        source_statement_embedding: SourceStatementEmbeddingNode,
        source_procedure_embedding: SourceProcedureEmbeddingNode,
        embedding: SourceBlockEmbeddingNode,
    ) -> None:
        graph = StateGraph(SourceProcessingState)
        graph.add_node('ocr', ocr.run)
        graph.add_edge(START, 'ocr')

        graph.add_node('content_correction_worker', content_correction.worker)
        graph.add_node('content_correction_collect', content_correction.collect)
        graph.add_conditional_edges(
            'ocr',
            content_correction.dispatch,
            ['content_correction_worker', 'content_correction_collect'],
        )
        graph.add_edge(
            'content_correction_worker', 'content_correction_collect'
        )

        graph.add_node('formatter_worker', formatter.worker)
        graph.add_node('formatter_collect', formatter.collect)
        graph.add_conditional_edges(
            'content_correction_collect',
            formatter.dispatch,
            ['formatter_worker', 'formatter_collect'],
        )
        graph.add_edge('formatter_worker', 'formatter_collect')

        graph.add_node('text_seam_even_worker', text_seam.even_worker)
        graph.add_node('text_seam_even_collect', text_seam.even_collect)
        graph.add_node('text_seam_odd_worker', text_seam.odd_worker)
        graph.add_node('text_seam_odd_collect', text_seam.odd_collect)
        graph.add_conditional_edges(
            'formatter_collect',
            text_seam.dispatch_even,
            ['text_seam_even_worker', 'text_seam_even_collect'],
        )
        graph.add_edge('text_seam_even_worker', 'text_seam_even_collect')
        graph.add_conditional_edges(
            'text_seam_even_collect',
            text_seam.dispatch_odd,
            ['text_seam_odd_worker', 'text_seam_odd_collect'],
        )
        graph.add_edge('text_seam_odd_worker', 'text_seam_odd_collect')

        graph.add_node('image_seam_worker', image_seam.worker)
        graph.add_node('image_seam_collect', image_seam.collect)
        graph.add_conditional_edges(
            'text_seam_odd_collect',
            image_seam.dispatch,
            ['image_seam_worker', 'image_seam_collect'],
        )
        graph.add_edge('image_seam_worker', 'image_seam_collect')

        graph.add_node('image_filter_worker', image_filter.worker)
        graph.add_node('image_filter_collect', image_filter.collect)
        graph.add_conditional_edges(
            'image_seam_collect',
            image_filter.dispatch,
            ['image_filter_worker', 'image_filter_collect'],
        )
        graph.add_edge('image_filter_worker', 'image_filter_collect')

        graph.add_node('exercise_splitter_worker', exercise_splitter.worker)
        graph.add_node('exercise_splitter_collect', exercise_splitter.collect)
        graph.add_conditional_edges(
            'image_filter_collect',
            exercise_splitter.dispatch,
            ['exercise_splitter_worker', 'exercise_splitter_collect'],
        )
        graph.add_edge('exercise_splitter_worker', 'exercise_splitter_collect')

        graph.add_node('instruction_finder', instruction_finder.run)
        graph.add_edge('exercise_splitter_collect', 'instruction_finder')
        graph.add_node('exercise_finder', exercise_finder.run)
        graph.add_edge('instruction_finder', 'exercise_finder')
        graph.add_node('pedagogical_finder', pedagogical_finder.run)
        graph.add_edge('exercise_finder', 'pedagogical_finder')
        graph.add_node('statement_procedure', statement_procedure.run)
        graph.add_edge('pedagogical_finder', 'statement_procedure')
        graph.add_node('instruction_governance', instruction_governance.run)
        graph.add_edge('statement_procedure', 'instruction_governance')

        graph.add_node(
            'source_statement_reconstruction',
            source_statement_reconstruction.run,
        )
        graph.add_node(
            'source_statement_reconstruction_worker',
            source_statement_reconstruction.worker,
        )
        graph.add_node(
            'source_statement_reconstruction_collect',
            source_statement_reconstruction.collect,
        )
        graph.add_edge(
            'instruction_governance', 'source_statement_reconstruction'
        )
        graph.add_conditional_edges(
            'source_statement_reconstruction',
            source_statement_reconstruction.dispatch,
            [
                'source_statement_reconstruction_worker',
                'source_statement_reconstruction_collect',
            ],
        )
        graph.add_edge(
            'source_statement_reconstruction_worker',
            'source_statement_reconstruction_collect',
        )
        graph.add_edge(
            'source_statement_reconstruction_collect',
            'source_statement_embedding',
        )

        graph.add_node(
            'source_procedure_reconstruction',
            source_procedure_reconstruction.run,
        )
        graph.add_node(
            'source_procedure_reconstruction_worker',
            source_procedure_reconstruction.worker,
        )
        graph.add_node(
            'source_procedure_reconstruction_collect',
            source_procedure_reconstruction.collect,
        )
        graph.add_edge(
            'instruction_governance', 'source_procedure_reconstruction'
        )
        graph.add_conditional_edges(
            'source_procedure_reconstruction',
            source_procedure_reconstruction.dispatch,
            [
                'source_procedure_reconstruction_worker',
                'source_procedure_reconstruction_collect',
            ],
        )
        graph.add_edge(
            'source_procedure_reconstruction_worker',
            'source_procedure_reconstruction_collect',
        )
        graph.add_edge(
            'source_procedure_reconstruction_collect',
            'source_procedure_embedding',
        )
        graph.add_node(
            'source_statement_embedding', source_statement_embedding.run
        )
        graph.add_node(
            'source_procedure_embedding', source_procedure_embedding.run
        )
        graph.add_node('embedding', embedding.run)
        graph.add_edge(
            ['source_statement_embedding', 'source_procedure_embedding'],
            'embedding',
        )
        graph.add_edge('embedding', END)
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: SourceProcessingState) -> SourceProcessingState:
        """Run OCR through final block embedding without publishing data."""
        result = await self._graph.ainvoke(state)
        return SourceProcessingState.model_validate(result)
