import dspy

from kms2 import composition
from kms2.config import (
    ContextWindowSettings,
    ImageEnrichmentSettings,
    OCRSettings,
    PredictorStrategy,
    Settings,
    SourceSettings,
    SplitterSettings,
    StageInferenceSettings,
    TextSeamSettings,
)
from kms2.database.client import DatabaseClient
from kms2.langgraph.source.graph import SourceGraph
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_enrichment import ImageEnrichmentNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.ocr import OCRNode
from kms2.node.source.persistence import SourcePersistenceNode
from kms2.node.source.splitter import SplitterNode
from kms2.node.source.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider


class _RecordingRuntime:
    calls: list[tuple[str, PredictorStrategy, type[dspy.Signature]]]

    def predictor(
        self,
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
    ) -> dspy.Module:
        self.calls.append(
            (
                inference.model_server_profile,
                inference.strategy,
                signature,
            )
        )
        if inference.strategy is PredictorStrategy.PREDICT:
            return dspy.Predict(signature)
        return dspy.ChainOfThought(signature)


def _stage_inference(
    model_server_profile: str,
    strategy: PredictorStrategy,
) -> StageInferenceSettings:
    return StageInferenceSettings(
        model_server_profile=model_server_profile,
        strategy=strategy,
    )


def _settings() -> Settings:
    return Settings(
        source=SourceSettings(
            content_correction=_stage_inference(
                'content-correction', PredictorStrategy.PREDICT
            ),
            formatting=_stage_inference(
                'formatting', PredictorStrategy.CHAIN_OF_THOUGHT
            ),
            text_seam=TextSeamSettings(
                judge=_stage_inference('text-judge', PredictorStrategy.PREDICT),
                rewriter=_stage_inference(
                    'text-rewriter', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            image_seam=_stage_inference(
                'image-judge', PredictorStrategy.PREDICT
            ),
            image_enrichment=ImageEnrichmentSettings(
                inference=_stage_inference(
                    'image-enrichment', PredictorStrategy.PREDICT
                ),
                context_window=ContextWindowSettings(
                    backward_budget=200,
                    forward_budget=200,
                ),
            ),
            splitter=SplitterSettings(
                router=_stage_inference(
                    'splitter-router', PredictorStrategy.PREDICT
                ),
                splitter=_stage_inference(
                    'splitter', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
                context_window=ContextWindowSettings(
                    backward_budget=4,
                    forward_budget=12,
                ),
            ),
        ),
        ocr=OCRSettings(api_key='test-key'),
    )


def test_build_source_graph_composes_all_source_dependencies():
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)

    graph = composition.build_source_graph(settings, local_models, database)

    assert isinstance(graph, SourceGraph)
    assert isinstance(graph.ocr, OCRNode)
    assert isinstance(graph.content_correction, ContentCorrectionNode)
    assert isinstance(graph.formatter, FormattingNode)
    assert isinstance(graph.ocr.provider, MistralOCRProvider)
    assert graph.ocr.provider._settings is settings.ocr
    assert isinstance(graph.text_seam, TextSeamNode)
    assert isinstance(graph.image_seam, ImageSeamNode)
    assert isinstance(graph.image_enrichment, ImageEnrichmentNode)
    assert isinstance(graph.splitter, SplitterNode)
    assert isinstance(graph.persistence, SourcePersistenceNode)
    assert (
        graph.splitter._context_window
        == settings.source.splitter.context_window
    )

    assert type(graph.content_correction._corrector.predictor) is dspy.Predict
    assert type(graph.formatter._formatter.predictor) is dspy.ChainOfThought
    assert type(graph.text_seam._judge.predictor) is dspy.Predict
    assert type(graph.text_seam._rewriter.predictor) is dspy.ChainOfThought
    assert type(graph.image_seam._judge.predictor) is dspy.Predict
    assert (
        graph.image_enrichment._context_window
        == settings.source.image_enrichment.context_window
    )
    assert type(graph.image_enrichment._enricher.predictor) is dspy.Predict
    assert type(graph.splitter._router.predictor) is dspy.Predict
    assert type(graph.splitter._splitter.predictor) is dspy.ChainOfThought

    assert [
        model_server_profile
        for model_server_profile, _, _ in local_models.calls
    ] == [
        'content-correction',
        'formatting',
        'text-judge',
        'text-rewriter',
        'image-judge',
        'image-enrichment',
        'splitter-router',
        'splitter',
    ]
