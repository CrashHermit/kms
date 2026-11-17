"""Dependency composition for the source-processing graph."""

from kms.config.settings import Settings
from kms.inference.embedding.resources import EmbeddingResources
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.ocr.mistral import MistralOCRProvider
from kms.source.processing.block_embedding import SourceBlockEmbeddingNode
from kms.source.processing.correction import composition as correction
from kms.source.processing.exercise_finder import composition as exercise_finder
from kms.source.processing.exercise_splitter import (
    composition as exercise_splitter,
)
from kms.source.processing.formatting import composition as formatting
from kms.source.processing.graph import SourceProcessingGraph
from kms.source.processing.image_filter import composition as image_filter
from kms.source.processing.image_seam import composition as image_seam
from kms.source.processing.instruction_redistribution import (
    composition as instruction_redistribution,
)
from kms.source.processing.ocr import OCRNode
from kms.source.processing.procedure_finder import (
    composition as procedure_finder,
)
from kms.source.processing.procedure_splitter import (
    composition as procedure_splitter,
)
from kms.source.processing.text_seam import composition as text_seam
from kms.source.processing.unit_assembly import SourceUnitAssemblyNode
from kms.source.processing.unit_embedding import SourceUnitEmbeddingNode


def build_source_processing_graph(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    embeddings: EmbeddingResources,
    tokenizers: LanguageModelTokenizers,
) -> SourceProcessingGraph:
    """Build the source graph from shared inference resources."""
    return SourceProcessingGraph(
        ocr=OCRNode(MistralOCRProvider(settings.ocr)),
        content_correction=correction.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        formatter=formatting.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        text_seam=text_seam.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        image_seam=image_seam.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        image_filter=image_filter.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        exercise_splitter=exercise_splitter.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        procedure_splitter=procedure_splitter.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        instruction_redistribution=instruction_redistribution.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        exercise_finder=exercise_finder.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        procedure_finder=procedure_finder.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        ),
        unit_assembly=SourceUnitAssemblyNode(),
        block_embedding=SourceBlockEmbeddingNode(
            embeddings.embedder,
            model=embeddings.model.model,
            dimension=embeddings.model.dimension,
        ),
        unit_embedding=SourceUnitEmbeddingNode(embeddings.embedder),
    )
