"""Token counters selected by each inference stage's configured profile."""

import tiktoken

from kms2.config.embedding import EmbeddingProviderSettings
from kms2.config.inference import StageInferenceSettings
from kms2.config.llm import ChatGPTLLMProfile, LocalLLMProfile
from kms2.config.settings import Settings
from kms2.core.windowing import TextTokenCounter
from kms2.inference.embedding import VoyageEmbeddingTokenCounter
from kms2.inference.reranking import VoyageRerankTokenCounter
from kms2.local_models.token_counting import LocalTokenizers


class _EncodingTokenCounter:
    """Estimate text tokens with a configured tiktoken encoding."""

    def __init__(self, encoding_name: str) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return one encoding token count per text, preserving input order."""
        return [
            len(self._encoding.encode(text, disallowed_special=()))
            for text in texts
        ]


class _LocalEmbeddingCounter:
    """Load the local embedding tokenizer only when local text is counted."""

    def __init__(self, tokenizers: LocalTokenizers) -> None:
        self._tokenizers = tokenizers

    def count_texts(self, texts: list[str]) -> list[int]:
        """Delegate to the lazily cached local model tokenizer."""
        return self._tokenizers.embedding.count_texts(texts)


class InferenceTokenizers:
    """Select per-stage text counters and retain local retrieval counters."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._local_tokenizers = LocalTokenizers(settings.local_models)
        self._local_embedding_counter = _LocalEmbeddingCounter(
            self._local_tokenizers
        )
        self._encoding_counters: dict[str, _EncodingTokenCounter] = {}
        self._hosted_embedding_counter: VoyageEmbeddingTokenCounter | None = (
            None
        )
        self._hosted_reranker_counter: VoyageRerankTokenCounter | None = None

    def for_inference(
        self,
        inference: StageInferenceSettings,
    ) -> TextTokenCounter:
        """Return the counter for the inference stage's resolved profile."""
        profile = self._settings.llm.resolve(inference)
        if isinstance(profile, LocalLLMProfile):
            return self._local_tokenizers.for_profile(
                profile.model_server_profile
            )
        if isinstance(profile, ChatGPTLLMProfile):
            if profile.tokenizer_encoding not in self._encoding_counters:
                self._encoding_counters[profile.tokenizer_encoding] = (
                    _EncodingTokenCounter(profile.tokenizer_encoding)
                )
            return self._encoding_counters[profile.tokenizer_encoding]
        raise TypeError(f'Unsupported LLM profile: {type(profile).__name__}')

    def text_counters(
        self, *inferences: StageInferenceSettings
    ) -> tuple[TextTokenCounter, ...]:
        """Return distinct stage counters in first-use order."""
        return tuple(
            dict.fromkeys(self.for_inference(stage) for stage in inferences)
        )

    @property
    def embedding(self) -> TextTokenCounter:
        """Return a lazy counter for the dedicated local embedding model."""
        return self._local_embedding_counter

    @property
    def selected_embedding(self) -> TextTokenCounter:
        """Return the exact counter for the configured embedding backend."""
        settings: EmbeddingProviderSettings = self._settings.embedding
        if settings.backend == 'local':
            return self.embedding
        if settings.backend == 'openrouter':
            if self._hosted_embedding_counter is None:
                self._hosted_embedding_counter = VoyageEmbeddingTokenCounter(
                    settings.tokenizer_path
                )
            return self._hosted_embedding_counter
        raise ValueError(f'Unsupported embedding backend: {settings.backend}')

    @property
    def reranker(self) -> TextTokenCounter:
        """Return the exact counter for the configured reranker backend."""
        settings = self._settings.reranker
        if settings.backend == 'local':
            return self._local_tokenizers.reranker
        if settings.backend == 'voyage':
            if self._hosted_reranker_counter is None:
                self._hosted_reranker_counter = VoyageRerankTokenCounter(
                    settings.tokenizer_path
                )
            return self._hosted_reranker_counter
        raise ValueError(f'Unsupported reranker backend: {settings.backend}')
