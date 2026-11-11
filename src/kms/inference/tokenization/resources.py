"""Own tokenizer counters and shared image metadata for LLM stages."""

from pathlib import Path

from kms.config.inference import StageInferenceSettings
from kms.config.llm import ChatGPTLLMProfile, LocalLLMProfile
from kms.config.settings import Settings
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.tokenization.text import (
    HuggingFaceTextTokenCounter,
    TiktokenTextTokenCounter,
)
from kms.media.metadata import ImageMetadataCache


class LanguageModelTokenizers:
    """Select and cache only generative-model material estimators."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._image_metadata = ImageMetadataCache()
        self._local_counters: dict[str, HuggingFaceTextTokenCounter] = {}
        self._encoding_counters: dict[str, TiktokenTextTokenCounter] = {}
        self._estimators: dict[object, MaterialTokenEstimator] = {}

    def for_inference(
        self, stage: StageInferenceSettings
    ) -> MaterialTokenEstimator:
        """Return the estimator for the resolved inference profile."""
        profile = self._settings.llm.resolve(stage)
        if isinstance(profile, LocalLLMProfile):
            model_profile = (
                self._settings.local_models.router.model_server_profiles[
                    profile.model_server_profile
                ]
            )
            tokenizer_path = str(
                Path(model_profile.tokenizer_path).expanduser().resolve()
            )
            if tokenizer_path not in self._local_counters:
                self._local_counters[tokenizer_path] = (
                    HuggingFaceTextTokenCounter(
                        tokenizer_path, add_special_tokens=False
                    )
                )
            text_counter = self._local_counters[tokenizer_path]
        elif isinstance(profile, ChatGPTLLMProfile):
            encoding = profile.tokenizer_encoding
            if encoding not in self._encoding_counters:
                self._encoding_counters[encoding] = TiktokenTextTokenCounter(
                    encoding
                )
            text_counter = self._encoding_counters[encoding]
        else:
            raise TypeError(
                f'Unsupported LLM profile: {type(profile).__name__}'
            )

        estimator = self._estimators.get(text_counter)
        if estimator is None:
            estimator = MaterialTokenEstimator(
                text_counter, self._image_metadata
            )
            self._estimators[text_counter] = estimator
        return estimator

    def counters(
        self, *stages: StageInferenceSettings
    ) -> tuple[MaterialTokenEstimator, ...]:
        """Return distinct stage estimators in first-use order."""
        return tuple(
            dict.fromkeys(self.for_inference(stage) for stage in stages)
        )
