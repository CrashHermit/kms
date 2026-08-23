"""Deterministic exact cosine-similarity indexes for construction."""

from collections.abc import Sequence

import faiss
import numpy as np
from pydantic import BaseModel


class Match(BaseModel):
    """One stable-key vector match."""

    key: str
    score: float


class ExactCosineIndex:
    """An application-owned wrapper around a FAISS exact index."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError('dimension must be positive')
        self.dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._keys: list[str] = []
        self._key_set: set[str] = set()

    def __len__(self) -> int:
        """Returns the number of indexed vectors."""
        return len(self._keys)

    def add(self, items: Sequence[tuple[str, Sequence[float]]]) -> None:
        """Adds stable-key vectors after validating the complete batch."""
        if not items:
            return
        keys = [key for key, _ in items]
        if len(keys) != len(set(keys)):
            raise ValueError('batch contains duplicate keys')
        if self._key_set.intersection(keys):
            raise ValueError('key already exists in index')
        vectors = np.asarray([vector for _, vector in items], dtype=np.float32)
        self._validate_vectors(vectors)
        faiss.normalize_L2(vectors)
        self._index.add(np.ascontiguousarray(vectors))
        self._keys.extend(keys)
        self._key_set.update(keys)

    def search(self, query: Sequence[float], *, top_k: int) -> list[Match]:
        """Returns deterministic top matches ordered by score, then key."""
        if top_k < 0:
            raise ValueError('top_k must not be negative')
        vector = np.asarray([query], dtype=np.float32)
        self._validate_vectors(vector)
        if top_k == 0 or not self._keys:
            return []
        faiss.normalize_L2(vector)
        scores, indices = self._index.search(vector, len(self._keys))
        matches = [
            Match(key=self._keys[index], score=float(score))
            for score, index in zip(scores[0], indices[0], strict=True)
            if index >= 0
        ]
        return sorted(matches, key=lambda match: (-match.score, match.key))[
            :top_k
        ]

    def _validate_vectors(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f'vectors must have dimension {self.dimension}')
        if not np.isfinite(vectors).all():
            raise ValueError('vectors must contain only finite values')
        if np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError('vectors must not have zero norm')
