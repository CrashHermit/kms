"""Durable content-addressed storage for finalized source media."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from kms.core.model.page import SourcePage
from kms.core.model.visual_asset import VisualAsset


class SourceArtifactStore:
    """Store source artifacts by digest and finalize media references."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def store(self, data: bytes, *, suffix: str = '.bin') -> Path:
        """Atomically store bytes and verify any existing content address."""
        digest = hashlib.sha256(data).hexdigest()
        path = self._root / digest[:2] / f'{digest}{suffix}'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._verify(path, digest)
            return path

        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f'.{digest}.',
            suffix='.tmp',
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'wb') as temporary_file:
                temporary_file.write(data)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        self._verify(path, digest)
        return path

    @staticmethod
    def read(path: str | Path) -> bytes:
        """Read an artifact by its path without assigning source semantics."""
        return Path(path).read_bytes()

    def finalize_pages(self, pages: list[SourcePage]) -> list[SourcePage]:
        """Copy every crop and asset durably, adding digests of stored bytes."""
        finalized_paths: dict[str, tuple[Path, str]] = {}
        stored_paths: dict[tuple[str, str], Path] = {}

        def finalize_path(raw_path: str) -> tuple[Path, str]:
            if raw_path not in finalized_paths:
                data = Path(raw_path).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                suffix = Path(raw_path).suffix or '.bin'
                key = (digest, suffix)
                if key not in stored_paths:
                    stored_paths[key] = self.store(data, suffix=suffix)
                finalized_paths[raw_path] = stored_paths[key], digest
            return finalized_paths[raw_path]

        finalized_pages = []
        for page in pages:
            blocks = []
            for block in page.blocks:
                crop_path = block.crop_path
                crop_sha256 = block.crop_sha256
                if crop_path is not None:
                    durable_path, crop_sha256 = finalize_path(crop_path)
                    crop_path = str(durable_path)
                assets = []
                for asset in block.assets:
                    durable_path, content_sha256 = finalize_path(asset.path)
                    assets.append(
                        asset.model_copy(
                            update={
                                'path': str(durable_path),
                                'content_sha256': content_sha256,
                            }
                        )
                    )
                blocks.append(
                    block.model_copy(
                        update={
                            'crop_path': crop_path,
                            'crop_sha256': crop_sha256,
                            'assets': assets,
                        }
                    )
                )
            finalized_pages.append(page.model_copy(update={'blocks': blocks}))
        return finalized_pages

    @staticmethod
    def _verify(path: Path, expected_digest: str) -> None:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError(
                'Stored source artifact content does not match its digest'
            )


def read_asset_bytes(asset: VisualAsset) -> bytes:
    """Read finalized asset bytes and verify the recorded SHA-256."""
    data = Path(asset.path).read_bytes()
    if (
        asset.content_sha256 is None
        or hashlib.sha256(data).hexdigest() != asset.content_sha256
    ):
        raise ValueError(
            'Source asset content does not match its recorded digest'
        )
    return data
