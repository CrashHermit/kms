from collections.abc import Sequence

import dspy

from kms2.composition.source_processing import build_source_processing_graph
from kms2.config.inference import (
    ContextWindowSettings,
    PredictorStrategy,
    StageInferenceSettings,
    TextInferenceSettings,
    VisionInferenceSettings,
)
from kms2.config.llm import LLMSettings, LocalLLMProfile
from kms2.config.services import OCRSettings
from kms2.config.settings import Settings
from kms2.config.source_processing import (
    ContentCorrectionSettings,
    FormattingSettings,
    ImageDescriptionSettings,
    ImageSeamSettings,
    SourceProcessingSettings,
    TextSeamSettings,
)
from kms2.database.client import DatabaseClient
from kms2.langgraph.source_processing.graph import SourceProcessingGraph
from kms2.module.source_processing.content_correction import (
    ContentCorrectorModule,
)
from kms2.node.source_processing.content_correction import ContentCorrectionNode
from kms2.node.source_processing.embedding import EmbeddingNode
from kms2.node.source_processing.formatting import FormattingNode
from kms2.node.source_processing.image_description import ImageDescriptionNode
from kms2.node.source_processing.image_seam import ImageSeamNode
from kms2.node.source_processing.ocr import OCRNode
from kms2.node.source_processing.persistence import SourcePersistenceNode
from kms2.node.source_processing.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider
from kms2.train.recorder import Recorder, RecordingModule


class _RecordingRuntime:
    calls: list[tuple[str, PredictorStrategy, type[dspy.Signature]]]

    def predictor(
        self,
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
        *,
        model_server_profile: str,
        examples: Sequence[dspy.Example] | None = None,
    ) -> dspy.Module:
        self.calls.append(
            (
                model_server_profile,
                inference.strategy,
                signature,
            )
        )
        if inference.strategy is PredictorStrategy.PREDICT:
            return dspy.Predict(signature)
        return dspy.ChainOfThought(signature)


class _Counter:
    def count_texts(self, texts):
        return [1] * len(texts)


class _Tokenizers:
    def for_inference(self, _inference):
        return _Counter()

    def text_counters(self, *inferences):
        return tuple(
            _Counter()
            for _ in dict.fromkeys(
                inference.profile for inference in inferences
            )
        )

    embedding = _Counter()
    reranker = _Counter()


def _stage_inference(
    profile: str,
    strategy: PredictorStrategy,
) -> TextInferenceSettings:
    return TextInferenceSettings(profile=profile, strategy=strategy)


def _vision_inference(
    profile: str,
    strategy: PredictorStrategy,
) -> VisionInferenceSettings:
    return VisionInferenceSettings(profile=profile, strategy=strategy)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        source_processing=SourceProcessingSettings(
            content_correction=ContentCorrectionSettings(
                inference=_vision_inference(
                    'content-correction', PredictorStrategy.PREDICT
                )
            ),
            formatting=FormattingSettings(
                inference=_stage_inference(
                    'formatting', PredictorStrategy.CHAIN_OF_THOUGHT
                )
            ),
            text_seam=TextSeamSettings(
                judge=_stage_inference('text-judge', PredictorStrategy.PREDICT),
                rewriter=_stage_inference(
                    'text-rewriter', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            image_seam=ImageSeamSettings(
                inference=_vision_inference(
                    'image-judge', PredictorStrategy.PREDICT
                )
            ),
            image_description=ImageDescriptionSettings(
                inference=_vision_inference(
                    'image-description', PredictorStrategy.PREDICT
                ),
                context_window=ContextWindowSettings(
                    backward_budget=200,
                    forward_budget=200,
                ),
            ),
        ),
        local_models={
            'router': {
                'model_server_profiles': {
                    'content-correction': {
                        'model_path': 'unused.gguf',
                        'tokenizer_path': 'unused.json',
                    },
                    'formatting': {
                        'model_path': 'unused.gguf',
                        'tokenizer_path': 'unused.json',
                    },
                    'text-judge': {
                        'model_path': 'unused.gguf',
                        'tokenizer_path': 'unused.json',
                        'context_size': 32768,
                        'parallel': 1,
                    },
                    'text-rewriter': {
                        'model_path': 'unused.gguf',
                        'tokenizer_path': 'unused.json',
                        'context_size': 32768,
                        'parallel': 1,
                    },
                    'image-judge': {
                        'model_path': 'unused.gguf',
                        'tokenizer_path': 'unused.json',
                        'context_size': 8192,
                        'parallel': 1,
                    },
                    'image-description': {
                        'model_path': 'unused.gguf',
                        'tokenizer_path': 'unused.json',
                        'context_size': 8192,
                        'parallel': 1,
                    },
                },
            },
        },
        ocr=OCRSettings(api_key='test-key'),
        llm=LLMSettings(
            profiles={
                **LLMSettings().profiles,
                **{
                    name: LocalLLMProfile(model_server_profile=name)
                    for name in (
                        'content-correction',
                        'formatting',
                        'text-judge',
                        'text-rewriter',
                        'image-judge',
                        'image-description',
                    )
                },
            }
        ),
    )


def test_build_source_processing_graph_composes_all_source_dependencies():
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)
    tokenizers = _Tokenizers()

    graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        embedding_client=object(),
        tokenizers=tokenizers,
    )

    assert isinstance(graph, SourceProcessingGraph)
    assert isinstance(graph.ocr, OCRNode)
    assert isinstance(graph.content_correction, ContentCorrectionNode)
    assert isinstance(graph.formatter, FormattingNode)
    assert isinstance(graph.ocr.provider, MistralOCRProvider)
    assert graph.ocr.provider._settings is settings.ocr
    assert isinstance(graph.text_seam, TextSeamNode)
    assert isinstance(graph.image_seam, ImageSeamNode)
    assert isinstance(graph.embedding, EmbeddingNode)
    assert isinstance(graph.image_description, ImageDescriptionNode)
    assert isinstance(graph.persistence, SourcePersistenceNode)
    assert not hasattr(graph.persistence, '_schema_initializer')

    assert type(graph.content_correction._corrector.predictor) is dspy.Predict
    assert type(graph.formatter._formatter.predictor) is dspy.ChainOfThought
    assert type(graph.text_seam._judge.predictor) is dspy.Predict
    assert type(graph.text_seam._rewriter.predictor) is dspy.ChainOfThought
    assert type(graph.image_seam._judge.predictor) is dspy.Predict
    assert (
        graph.image_description._context_window
        == settings.source_processing.image_description.context_window
    )
    assert type(graph.image_description._describer.predictor) is dspy.Predict


def test_build_source_processing_graph_wraps_predictors_when_recording_is_enabled(
    tmp_path,
):
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)
    tokenizers = _Tokenizers()
    recorder = Recorder(tmp_path)

    graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        embedding_client=object(),
        tokenizers=tokenizers,
        recorder=recorder,
    )

    content_predictor = graph.content_correction._corrector.predictor
    assert isinstance(content_predictor, RecordingModule)
    assert type(content_predictor.predictor) is dspy.Predict
    assert content_predictor.module is ContentCorrectorModule
