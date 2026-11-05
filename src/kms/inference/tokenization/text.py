"""Lazy text counters for configured local and hosted LLM tokenizers."""

from pathlib import Path

import tiktoken
from tokenizers import Tokenizer


class HuggingFaceTextTokenCounter:
    """Count text with a lazily loaded Hugging Face tokenizer JSON/model."""

    def __init__(
        self, tokenizer_path: str, *, add_special_tokens: bool
    ) -> None:
        self._tokenizer_path = tokenizer_path
        self._add_special_tokens = add_special_tokens
        self._tokenizer: Tokenizer | None = None

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return raw ordered counts, loading only for nonempty batches."""
        if not any(texts):
            return [0] * len(texts)
        tokenizer = self._load()
        return [
            len(encoding)
            for encoding in tokenizer.encode_batch_fast(
                texts, add_special_tokens=self._add_special_tokens
            )
        ]

    def _load(self) -> Tokenizer:
        if self._tokenizer is None:
            path = Path(self._tokenizer_path).expanduser()
            local = (
                path.is_absolute()
                or self._tokenizer_path.startswith(('~', '.'))
                or path.is_file()
            )
            tokenizer = (
                Tokenizer.from_file(str(path))
                if local
                else Tokenizer.from_pretrained(self._tokenizer_path)
            )
            tokenizer.no_padding()
            tokenizer.no_truncation()
            self._tokenizer = tokenizer
        return self._tokenizer


class TiktokenTextTokenCounter:
    """Count text with a configured tiktoken encoding."""

    def __init__(self, encoding_name: str) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_texts(self, texts: list[str]) -> list[int]:
        """Return one encoding token count per text, preserving order."""
        return [
            len(self._encoding.encode(text, disallowed_special=()))
            for text in texts
        ]
