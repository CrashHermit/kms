"""Dependency composition for the KMS2 source graph."""

from kms.composition.embedding import EmbeddingResources
from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.media_budget import ImagePayloadBudget
from kms.database.client import DatabaseClient
from kms.database.source.source_graph_repository import SourceGraphRepository
from kms.langgraph.source_processing.graph import SourceProcessingGraph
from kms.local_models.runtime import LocalModelRuntime
from kms.module.source_processing.content_correction import (
    ContentCorrectorModule,
    ContentCorrectorSignature,
)
from kms.module.source_processing.examples.content_correction import (
    content_correction_examples,
)
from kms.module.source_processing.examples.formatting import (
    formatting_examples,
)
from kms.module.source_processing.examples.image_description import (
    image_description_examples,
)
from kms.module.source_processing.examples.image_filter import (
    image_filter_examples,
)
from kms.module.source_processing.examples.image_seam import (
    image_seam_examples,
)
from kms.module.source_processing.examples.text_seam import (
    text_seam_judge_examples,
    text_seam_rewrite_examples,
)
from kms.module.source_processing.formatting import (
    FormatterModule,
    FormatterSignature,
)
from kms.module.source_processing.image_description import (
    ImageDescriptionModule,
    ImageDescriptionSignature,
)
from kms.module.source_processing.image_filter import (
    ImageFilterModule,
    ImageFilterSignature,
)
from kms.module.source_processing.image_seam import (
    ImageSeamJudgeModule,
    ImageSeamSignature,
)
from kms.module.source_processing.text_seam_judge import (
    TextSeamJudgeModule,
    TextSeamSignature,
)
from kms.module.source_processing.text_seam_rewriter import (
    TextSeamRewriterModule,
    TextSeamRewriteSignature,
)
from kms.node.source_processing.content_correction import ContentCorrectionNode
from kms.node.source_processing.embedding import EmbeddingNode
from kms.node.source_processing.formatting import FormattingNode
from kms.node.source_processing.image_description import ImageDescriptionNode
from kms.node.source_processing.image_filter import ImageFilterNode
from kms.node.source_processing.image_seam import ImageSeamNode
from kms.node.source_processing.ocr import OCRNode
from kms.node.source_processing.persistence import SourcePersistenceNode
from kms.node.source_processing.text_seam import TextSeamNode
from kms.ocr.mistral import MistralOCRProvider
from kms.service.source_artifacts import SourceArtifactStore
from kms.train.recorder import Recorder


def build_source_processing_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    artifacts: SourceArtifactStore,
    embeddings: EmbeddingResources,
    tokenizers: LanguageModelTokenizers,
    recorder: Recorder | None = None,
    predictors: PredictorFactory | None = None,
    persist: bool = True,
) -> SourceProcessingGraph:
    """Compose the source graph from settings and started runtime resources."""
    source = settings.source_processing
    predictor_factory = predictors or PredictorFactory(
        settings, local_models, recorder
    )
    content_corrector = ContentCorrectorModule(
        predictor_factory.create(
            ContentCorrectorModule,
            source.content_correction.inference,
            ContentCorrectorSignature,
            examples=content_correction_examples(),
        )
    )
    formatter = FormatterModule(
        predictor_factory.create(
            FormatterModule,
            source.formatting.inference,
            FormatterSignature,
            examples=formatting_examples(),
        )
    )
    text_seam_judge = TextSeamJudgeModule(
        predictor_factory.create(
            TextSeamJudgeModule,
            source.text_seam.judge,
            TextSeamSignature,
            examples=text_seam_judge_examples(),
        )
    )
    text_seam_rewriter = TextSeamRewriterModule(
        predictor_factory.create(
            TextSeamRewriterModule,
            source.text_seam.rewriter,
            TextSeamRewriteSignature,
            examples=text_seam_rewrite_examples(),
        )
    )
    image_seam_judge = ImageSeamJudgeModule(
        predictor_factory.create(
            ImageSeamJudgeModule,
            source.image_seam.inference,
            ImageSeamSignature,
            examples=image_seam_examples(),
        )
    )
    image_describer = ImageDescriptionModule(
        predictor_factory.create(
            ImageDescriptionModule,
            source.image_description.inference,
            ImageDescriptionSignature,
            examples=image_description_examples(),
        )
    )
    image_filter = ImageFilterModule(
        predictor_factory.create(
            ImageFilterModule,
            source.image_filter.inference,
            ImageFilterSignature,
            examples=image_filter_examples(),
        )
    )
    return SourceProcessingGraph(
        ocr=OCRNode(MistralOCRProvider(settings.ocr)),
        content_correction=ContentCorrectionNode(
            content_corrector,
            budget=build_token_budget(
                settings,
                tokenizers,
                source.content_correction.inference,
                input_token_budget=source.content_correction.input_token_budget,
                safety_margin_tokens=source.content_correction.safety_margin_tokens,
            ),
            image_budget=ImagePayloadBudget(
                **source.content_correction.image_payload.model_dump()
            ),
            batch_size=source.content_correction.batch_size,
        ),
        formatter=FormattingNode(
            formatter,
            budget=build_token_budget(
                settings,
                tokenizers,
                source.formatting.inference,
                input_token_budget=source.formatting.input_token_budget,
                safety_margin_tokens=source.formatting.safety_margin_tokens,
            ),
            batch_size=source.formatting.batch_size,
        ),
        text_seam=TextSeamNode(
            text_seam_judge,
            text_seam_rewriter,
            judge_budget=build_token_budget(
                settings,
                tokenizers,
                source.text_seam.judge,
                input_token_budget=source.text_seam.judge_input_token_budget,
                safety_margin_tokens=source.text_seam.safety_margin_tokens,
            ),
            rewrite_budget=build_token_budget(
                settings,
                tokenizers,
                source.text_seam.rewriter,
                input_token_budget=source.text_seam.rewrite_input_token_budget,
                safety_margin_tokens=source.text_seam.safety_margin_tokens,
            ),
            judge_batch_size=source.text_seam.judge_batch_size,
            rewrite_batch_size=source.text_seam.rewrite_batch_size,
        ),
        image_seam=ImageSeamNode(
            image_seam_judge,
            budget=build_token_budget(
                settings,
                tokenizers,
                source.image_seam.inference,
                input_token_budget=source.image_seam.input_token_budget,
                safety_margin_tokens=source.image_seam.safety_margin_tokens,
            ),
            image_budget=ImagePayloadBudget(
                **source.image_seam.image_payload.model_dump()
            ),
            batch_size=source.image_seam.batch_size,
        ),
        image_filter=ImageFilterNode(
            image_filter,
            source.image_filter.context_window,
            budget=build_token_budget(
                settings,
                tokenizers,
                source.image_filter.inference,
                input_token_budget=source.image_filter.input_token_budget,
                safety_margin_tokens=source.image_filter.safety_margin_tokens,
            ),
            image_budget=ImagePayloadBudget(
                **source.image_filter.image_payload.model_dump()
            ),
            batch_size=source.image_filter.batch_size,
            token_counters=tokenizers.counters(source.image_filter.inference),
        ),
        image_description=ImageDescriptionNode(
            image_describer,
            source.image_description.context_window,
            budget=build_token_budget(
                settings,
                tokenizers,
                source.image_description.inference,
                input_token_budget=source.image_description.input_token_budget,
                safety_margin_tokens=source.image_description.safety_margin_tokens,
            ),
            image_budget=ImagePayloadBudget(
                **source.image_description.image_payload.model_dump()
            ),
            batch_size=source.image_description.batch_size,
            token_counters=tokenizers.counters(
                source.image_description.inference
            ),
        ),
        embedding=EmbeddingNode(
            embeddings.embedder,
            model=embeddings.model.model,
            dimension=embeddings.model.dimension,
        ),
        persistence=SourcePersistenceNode(
            SourceGraphRepository(database.session),
            artifacts,
        ),
        persist=persist,
    )
