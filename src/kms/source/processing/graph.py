"""Source processing LangGraph assembly."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.correction.node import ContentCorrectionNode
from kms.source.processing.exercise_finder.node import ExerciseFinderNode
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode
from kms.source.processing.formatting.node import FormattingNode
from kms.source.processing.image_filter.node import ImageFilterNode
from kms.source.processing.image_seam.node import ImageSeamNode
from kms.source.processing.instruction_redistribution.node import (
    InstructionRedistributionNode,
)
from kms.source.processing.ocr import OCRNode
from kms.source.processing.procedure_finder.node import ProcedureFinderNode
from kms.source.processing.procedure_splitter.node import ProcedureSplitterNode
from kms.source.processing.state import SourceProcessingState
from kms.source.processing.text_seam.node import TextSeamNode
from kms.source.processing.unit_assembly import SourceUnitAssemblyNode
from kms.source.processing.unit_embedding import SourceUnitEmbeddingNode


class SourceProcessingGraph:
    """Compile the processing-only graph for a complete source snapshot."""

    def __init__(
        self,
        ocr: OCRNode,
        content_correction: ContentCorrectionNode,
        formatter: FormattingNode,
        text_seam: TextSeamNode,
        image_seam: ImageSeamNode,
        image_filter: ImageFilterNode,
        exercise_splitter: ExerciseSplitterNode,
        procedure_splitter: ProcedureSplitterNode,
        instruction_redistribution: InstructionRedistributionNode,
        exercise_finder: ExerciseFinderNode,
        procedure_finder: ProcedureFinderNode,
        unit_assembly: SourceUnitAssemblyNode,
        block_embedding: SourceBlockEmbeddingNode,
        unit_embedding: SourceUnitEmbeddingNode,
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

        graph.add_node('text_seam_judge_worker', text_seam.worker)
        graph.add_node('text_seam_judge_collect', text_seam.prepare_rewrites)
        graph.add_node('text_seam_rewrite_worker', text_seam.rewrite_worker)
        graph.add_node('text_seam_collect', text_seam.collect)
        graph.add_conditional_edges(
            'formatter_collect',
            text_seam.dispatch,
            ['text_seam_judge_worker', 'text_seam_collect'],
        )
        graph.add_edge('text_seam_judge_worker', 'text_seam_judge_collect')
        graph.add_conditional_edges(
            'text_seam_judge_collect',
            text_seam.dispatch_rewrites,
            ['text_seam_rewrite_worker', 'text_seam_collect'],
        )
        graph.add_edge('text_seam_rewrite_worker', 'text_seam_collect')

        graph.add_node('image_seam_worker', image_seam.worker)
        graph.add_node('image_seam_collect', image_seam.collect)
        graph.add_conditional_edges(
            'text_seam_collect',
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

        graph.add_node('procedure_splitter_worker', procedure_splitter.worker)
        graph.add_node('procedure_splitter_collect', procedure_splitter.collect)
        graph.add_conditional_edges(
            'exercise_splitter_collect',
            procedure_splitter.dispatch,
            ['procedure_splitter_worker', 'procedure_splitter_collect'],
        )
        graph.add_edge(
            'procedure_splitter_worker', 'procedure_splitter_collect'
        )

        graph.add_node(
            'instruction_detection', instruction_redistribution.detect
        )
        graph.add_node('exercise_finder', exercise_finder.run)
        graph.add_node('procedure_finder', procedure_finder.run)
        graph.add_node('unit_assembly', unit_assembly.run)
        graph.add_node(
            'instruction_redistribution', instruction_redistribution.run
        )
        graph.add_node('block_embedding', block_embedding.run)
        graph.add_node('unit_embedding', unit_embedding.run)
        graph.add_edge('procedure_splitter_collect', 'instruction_detection')
        graph.add_edge('instruction_detection', 'exercise_finder')
        graph.add_edge('exercise_finder', 'procedure_finder')
        graph.add_edge('procedure_finder', 'unit_assembly')
        graph.add_edge('unit_assembly', 'instruction_redistribution')
        graph.add_edge('instruction_redistribution', 'block_embedding')
        graph.add_edge('block_embedding', 'unit_embedding')
        graph.add_edge('unit_embedding', END)
        self._graph: CompiledStateGraph = graph.compile()

    async def run(self, state: SourceProcessingState) -> SourceProcessingState:
        """Run source processing without publishing data."""
        result = await self._graph.ainvoke(state)
        return SourceProcessingState.model_validate(result)
