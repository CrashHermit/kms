"""Local model-specific token counting."""

from pathlib import Path

from tokenizers import Tokenizer

from kms2.config.runtime import LocalModelRuntimeSettings
from kms2.core.windowing import TextTokenCounter


class LocalTokenCounter:
    """Count raw text with one local tokenizer vocabulary."""

    def __init__(self, tokenizer_path: Path) -> None:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.no_padding()
        tokenizer.no_truncation()
        self._tokenizer = tokenizer

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return one raw token count per text, preserving input order."""
        return [
            len(encoding)
            for encoding in self._tokenizer.encode_batch_fast(
                texts, add_special_tokens=False
            )
        ]


class LocalTokenizers:
    """Lazily cache the configured local tokenizers for one application."""

    def __init__(self, settings: LocalModelRuntimeSettings) -> None:
        self._settings = settings
        self._counters: dict[Path, LocalTokenCounter] = {}

    def for_profile(self, model_server_profile: str) -> LocalTokenCounter:
        """Return the cached counter for one configured LLM profile."""
        profile = self._settings.router.model_server_profiles[
            model_server_profile
        ]
        return self._for_path(Path(profile.tokenizer_path))

    def text_counters(
        self, *model_server_profiles: str
    ) -> tuple[TextTokenCounter, ...]:
        """Return distinct profile counters in first-use order."""
        return tuple(
            dict.fromkeys(
                self.for_profile(model_server_profile)
                for model_server_profile in model_server_profiles
            )
        )

    @property
    def embedding(self) -> LocalTokenCounter:
        """Return the dedicated embedding tokenizer counter."""
        return self._for_path(
            Path(self._settings.embedding.model.tokenizer_path)
        )

    @property
    def reranker(self) -> LocalTokenCounter:
        """Return the dedicated reranker tokenizer counter."""
        return self._for_path(
            Path(self._settings.reranker.model.tokenizer_path)
        )

    def _for_path(self, path: Path) -> LocalTokenCounter:
        resolved_path = path.expanduser().resolve()
        if resolved_path not in self._counters:
            self._counters[resolved_path] = LocalTokenCounter(resolved_path)
        return self._counters[resolved_path]
