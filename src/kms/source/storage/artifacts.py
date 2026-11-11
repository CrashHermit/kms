"""Durable source media finalization and image reads."""

from __future__ import annotations

import hashlib
from pathlib import Path

from kms.artifacts import ContentAddressedArtifactStore
from kms.source.models.source import SourcePage


class SourceArtifactStore(ContentAddressedArtifactStore):
    """Store source artifacts and finalize block media references."""

    def finalize_pages(self, pages: list[SourcePage]) -> list[SourcePage]:
        """Durably store block crops and canonical images with digest caching."""
        finalized_paths: dict[str, tuple[Path, str]] = {}
        stored_paths: dict[tuple[str, str], Path] = {}

        def finalize_path(raw_path: str) -> tuple[Path, str]:
            if raw_path not in finalized_paths:
                data = Path(raw_path).read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                suffix = Path(raw_path).suffix or '.bin'
                key = (digest, suffix)
                if key not in stored_paths:
                    stored_paths[key] = self.write(data, suffix=suffix)
                finalized_paths[raw_path] = stored_paths[key], digest
            return finalized_paths[raw_path]

        finalized_pages = []
        for page in pages:
            blocks = []
            for block in page.blocks:
                updates: dict[str, object] = {}
                if block.crop_path is not None:
                    crop_path, crop_sha256 = finalize_path(block.crop_path)
                    updates.update(
                        crop_path=str(crop_path), crop_sha256=crop_sha256
                    )
                if block.image_path is not None:
                    image_path, image_sha256 = finalize_path(block.image_path)
                    updates.update(
                        image_path=str(image_path), image_sha256=image_sha256
                    )
                blocks.append(block.model_copy(update=updates))
            finalized_pages.append(page.model_copy(update={'blocks': blocks}))
        return finalized_pages


def read_image_bytes(image_path: str, image_sha256: str) -> bytes:
    """Read and verify the canonical image stored on a source block."""
    data = Path(image_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != image_sha256:
        raise ValueError(
            'Source image content does not match its recorded digest'
        )
    return data
