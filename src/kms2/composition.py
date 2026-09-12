"""Explicit dependency composition for KMS2 runtime components."""

from functools import partial

from kms2.config import Settings
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.semantic.repository import SemanticRepository
from kms2.database.source.repository import SourceRepository
from kms2.langgraph.semantic.graph import SemanticGraph
from kms2.langgraph.source.graph import SourceGraph
from kms2.local_models import LocalModelRuntime
from kms2.module.semantic.entity_enrichment import (
    EntityEnrichmentModule,
    EntityEnrichmentSignature,
)
from kms2.module.semantic.event_enrichment import (
    EventEnrichmentModule,
    EventEnrichmentSignature,
)
from kms2.module.semantic.predicate_enrichment import (
    PredicateEnrichmentModule,
    PredicateEnrichmentSignature,
)
from kms2.module.semantic.triplet import (
    FactExtractionSignature,
    FactExtractorModule,
    TripletDecomposerModule,
    TripletDecompositionSignature,
)
from kms2.module.source.content_correction import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms2.module.source.formatting import FormatterModule, FormatterSignature
from kms2.module.source.image_enrichment import (
    ImageEnrichmentModule,
    ImageEnrichmentSignature,
)
from kms2.module.source.image_seam import (
    ImageSeamJudgeModule,
    ImageSeamSignature,
)
from kms2.module.source.splitter import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms2.module.source.text_seam import (
    TextSeamJudgeModule,
    TextSeamRewriterModule,
    TextSeamRewriteSignature,
    TextSeamSignature,
)
from kms2.node.semantic.entity_embedding import EntityEmbeddingNode
from kms2.node.semantic.entity_enrichment import EntityEnrichmentNode
from kms2.node.semantic.entity_enrichment_load import EntityEnrichmentLoadNode
from kms2.node.semantic.entity_enrichment_persistence import (
    EntityEnrichmentPersistenceNode,
)
from kms2.node.semantic.event_embedding import EventEmbeddingNode
from kms2.node.semantic.event_enrichment import EventEnrichmentNode
from kms2.node.semantic.event_enrichment_load import EventEnrichmentLoadNode
from kms2.node.semantic.event_enrichment_persistence import (
    EventEnrichmentPersistenceNode,
)
from kms2.node.semantic.predicate_embedding import PredicateEmbeddingNode
from kms2.node.semantic.predicate_enrichment import PredicateEnrichmentNode
from kms2.node.semantic.predicate_enrichment_load import (
    PredicateEnrichmentLoadNode,
)
from kms2.node.semantic.predicate_enrichment_persistence import (
    PredicateEnrichmentPersistenceNode,
)
from kms2.node.semantic.triplet import (
    FactExtractionNode,
    TripletDecompositionNode,
)
from kms2.node.semantic.triplet_load import TripletSourceLoadNode
from kms2.node.semantic.triplet_persistence import TripletPersistenceNode
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.embedding import EmbeddingNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_enrichment import ImageEnrichmentNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.ocr import OCRNode
from kms2.node.source.persistence import SourcePersistenceNode
from kms2.node.source.splitter import SplitterNode
from kms2.node.source.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider


def build_source_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> SourceGraph:
    """Compose the source graph from settings and started runtime resources."""
    source = settings.source
    content_corrector = ContentCorrectorModule(
        local_models.predictor(
            source.content_correction,
            ContentCorrectorSignature,
        )
    )
    formatter = FormatterModule(
        local_models.predictor(source.formatting, FormatterSignature)
    )
    text_seam_judge = TextSeamJudgeModule(
        local_models.predictor(source.text_seam.judge, TextSeamSignature)
    )
    text_seam_rewriter = TextSeamRewriterModule(
        local_models.predictor(
            source.text_seam.rewriter,
            TextSeamRewriteSignature,
        )
    )
    image_seam_judge = ImageSeamJudgeModule(
        local_models.predictor(source.image_seam, ImageSeamSignature)
    )
    image_enricher = ImageEnrichmentModule(
        local_models.predictor(
            source.image_enrichment.inference,
            ImageEnrichmentSignature,
        )
    )
    splitter_router = ExerciseStripRouterModule(
        local_models.predictor(
            source.splitter.router,
            ExerciseStripRouterSignature,
        )
    )
    splitter = ExerciseSplitterModule(
        local_models.predictor(
            source.splitter.splitter,
            ExerciseSplitterSignature,
        )
    )

    return SourceGraph(
        ocr=OCRNode(MistralOCRProvider(settings.ocr)),
        content_correction=ContentCorrectionNode(content_corrector),
        formatter=FormattingNode(formatter),
        text_seam=TextSeamNode(text_seam_judge, text_seam_rewriter),
        image_seam=ImageSeamNode(image_seam_judge),
        image_enrichment=ImageEnrichmentNode(
            image_enricher,
            source.image_enrichment.context_window,
        ),
        splitter=SplitterNode(
            splitter_router,
            splitter,
            source.splitter.context_window,
        ),
        embedding=EmbeddingNode(local_models.embedding),
        persistence=SourcePersistenceNode(
            SourceRepository(database.session),
            partial(schema.ensure_schema, database.session),
        ),
    )


def build_semantic_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
) -> SemanticGraph:
    """Compose the complete semantic graph from shared runtime resources."""
    semantic = settings.semantic
    source_repository = SourceRepository(database.session)
    semantic_repository = SemanticRepository(database.session)
    fact_extractor = FactExtractorModule(
        local_models.predictor(
            semantic.fact_extraction,
            FactExtractionSignature,
        )
    )
    triplet_decomposer = TripletDecomposerModule(
        local_models.predictor(
            semantic.triplet_decomposition,
            TripletDecompositionSignature,
        )
    )
    return SemanticGraph(
        triplet_source_load=TripletSourceLoadNode(source_repository),
        fact_extraction=FactExtractionNode(
            fact_extractor,
            semantic.context_window,
        ),
        triplet_decomposition=TripletDecompositionNode(triplet_decomposer),
        triplet_persistence=TripletPersistenceNode(
            semantic_repository,
            partial(schema.ensure_schema, database.session),
        ),
        entity_enrichment_load=EntityEnrichmentLoadNode(
            source_repository,
            semantic_repository,
            semantic.entity_enrichment.context_window,
        ),
        entity_enrichment=EntityEnrichmentNode(
            EntityEnrichmentModule(
                local_models.predictor(
                    semantic.entity_enrichment.inference,
                    EntityEnrichmentSignature,
                )
            )
        ),
        entity_embedding=EntityEmbeddingNode(local_models.embedding),
        entity_enrichment_persistence=EntityEnrichmentPersistenceNode(
            semantic_repository,
            partial(schema.ensure_schema, database.session),
        ),
        event_enrichment_load=EventEnrichmentLoadNode(
            source_repository,
            semantic_repository,
            semantic.event_enrichment.context_window,
        ),
        event_enrichment=EventEnrichmentNode(
            EventEnrichmentModule(
                local_models.predictor(
                    semantic.event_enrichment.inference,
                    EventEnrichmentSignature,
                )
            )
        ),
        event_embedding=EventEmbeddingNode(local_models.embedding),
        event_enrichment_persistence=EventEnrichmentPersistenceNode(
            semantic_repository,
            partial(schema.ensure_schema, database.session),
        ),
        predicate_enrichment_load=PredicateEnrichmentLoadNode(
            source_repository,
            semantic_repository,
            semantic.predicate_enrichment.context_window,
        ),
        predicate_enrichment=PredicateEnrichmentNode(
            PredicateEnrichmentModule(
                local_models.predictor(
                    semantic.predicate_enrichment.inference,
                    PredicateEnrichmentSignature,
                )
            )
        ),
        predicate_embedding=PredicateEmbeddingNode(local_models.embedding),
        predicate_enrichment_persistence=PredicateEnrichmentPersistenceNode(
            semantic_repository,
            partial(schema.ensure_schema, database.session),
        ),
    )


__all__ = ['build_semantic_graph', 'build_source_graph']
