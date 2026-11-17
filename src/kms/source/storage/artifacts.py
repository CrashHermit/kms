"""Durable source media finalization and image reads."""

from __future__ import annotations

import hashlib
from pathlib import Path

from kms.artifacts import ContentAddressedArtifactStore
from kms.source.models.unit import SourceUnit


class SourceArtifactStore(ContentAddressedArtifactStore):
    """Store source artifacts and finalize block media references."""

    def finalize_units(self, units: list[SourceUnit]) -> list[SourceUnit]:
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

        finalized_units = []
        for unit in units:
            blocks = []
            for block in unit.blocks:
                updates: dict[str, object] = {}
                if block.crop_path is not None:
                    crop_path, crop_sha256 = finalize_path(block.crop_path)
                    if (
                        block.crop_sha256 is not None
                        and block.crop_sha256 != crop_sha256
                    ):
                        raise ValueError(
                            'Source crop content does not match its recorded digest'
                        )
                    updates.update(
                        crop_path=str(crop_path), crop_sha256=crop_sha256
                    )
                if block.image_path is not None:
                    image_path, image_sha256 = finalize_path(block.image_path)
                    if (
                        block.image_sha256 is not None
                        and block.image_sha256 != image_sha256
                    ):
                        raise ValueError(
                            'Source image content does not match its recorded digest'
                        )
                    updates.update(
                        image_path=str(image_path), image_sha256=image_sha256
                    )
                blocks.append(block.model_copy(update=updates))
            finalized_units.append(unit.model_copy(update={'blocks': blocks}))
        return finalized_units


def read_image_bytes(image_path: str, image_sha256: str) -> bytes:
    """Read and verify the canonical image stored on a source block."""
    data = Path(image_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != image_sha256:
        raise ValueError(
            'Source image content does not match its recorded digest'
        )
    return data
