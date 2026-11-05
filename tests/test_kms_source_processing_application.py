from collections.abc import Sequence

import dspy

from kms.composition.embedding import EmbeddingResources
from kms.composition.source_processing import build_source_processing_graph
from kms.config.inference import (
    ContextWindowSettings,
    PredictorStrategy,
    StageInferenceSettings,
    TextInferenceSettings,
    VisionInferenceSettings,
)
from kms.config.llm import LLMSettings, LocalLLMProfile
from kms.config.services import OCRSettings
from kms.config.settings import Settings
from kms.config.source_processing import (
    ContentCorrectionSettings,
    FormattingSettings,
    ImageDescriptionSettings,
    ImageFilterSettings,
    ImageSeamSettings,
    SourceProcessingSettings,
    TextSeamSettings,
)
from kms.core.embedding import EmbeddingModel
from kms.core.tokenization.images import ImageMetadataCache
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.database.client import DatabaseClient
from kms.inference.embedding.budget import VoyageEmbeddingBudget
from kms.langgraph.source_processing.graph import SourceProcessingGraph
from kms.service.source_artifacts import SourceArtifactStore
from kms.train.recorder import Recorder


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
        return MaterialTokenEstimator(_Counter())

    def counters(self, *inferences):
        return tuple(
            MaterialTokenEstimator(_Counter())
            for _ in dict.fromkeys(
                inference.profile for inference in inferences
            )
        )


def _embedding_resources() -> EmbeddingResources:
    text_counter = _Counter()
    return EmbeddingResources(
        embedder=object(),
        model=EmbeddingModel(
            model='voyage/voyage-multimodal-3.5',
            dimension=1024,
        ),
        budget=VoyageEmbeddingBudget(
            text_counter=text_counter,
            image_metadata=ImageMetadataCache(),
        ),
    )


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
            image_filter=ImageFilterSettings(
                inference=_vision_inference(
                    'image-filter', PredictorStrategy.PREDICT
                )
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
                    'image-filter': {
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
                        'image-filter',
                    )
                },
            },
        ),
    )


def test_build_source_processing_graph_returns_compiled_facade(
    tmp_path,
):
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)
    tokenizers = _Tokenizers()

    graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        artifacts=SourceArtifactStore(tmp_path),
        embeddings=_embedding_resources(),
        tokenizers=tokenizers,
    )
    assert isinstance(graph, SourceProcessingGraph)
    assert callable(graph.run)
    assert not hasattr(graph, 'build_graph')
    assert not hasattr(graph, 'ainvoke')
    assert not hasattr(graph, 'graph')


def test_build_source_processing_graph_keeps_compiled_facade_private(
    tmp_path,
):
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    graph = build_source_processing_graph(
        settings,
        local_models,
        DatabaseClient(settings.database),
        artifacts=SourceArtifactStore(tmp_path),
        embeddings=_embedding_resources(),
        tokenizers=_Tokenizers(),
        recorder=Recorder(tmp_path),
    )
    assert isinstance(graph, SourceProcessingGraph)
    assert callable(graph.run)
    assert not hasattr(graph, 'build_graph')
    assert not hasattr(graph, 'ainvoke')
    assert not hasattr(graph, 'graph')
