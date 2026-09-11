"""Explicit dependency composition for KMS2 runtime components."""

from functools import partial

from kms2.config import Settings
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.database.source.repository import SourceRepository
from kms2.langgraph.source.graph import SourceGraph
from kms2.local_models import LocalModelRuntime
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
from kms2.node.source.content_correction import ContentCorrectionNode
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
        persistence=SourcePersistenceNode(
            SourceRepository(database.session),
            partial(schema.ensure_schema, database.session),
        ),
    )


__all__ = ['build_source_graph']
