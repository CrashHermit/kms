"""Content-addressed storage for durable artifacts."""

import hashlib
import os
import re
import tempfile
from pathlib import Path

_SHA256_NAME = re.compile(r'[0-9a-f]{64}')


class ArtifactIntegrityError(RuntimeError):
    """A content-addressed artifact no longer matches its identity."""


class ContentAddressedArtifactStore:
    """Store immutable artifacts by SHA-256 while retaining media suffixes."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()

    def write(self, data: bytes, *, suffix: str = '.bin') -> Path:
        """Atomically persist bytes and verify any existing content address."""
        digest = hashlib.sha256(data).hexdigest()
        path = self._root / digest[:2] / f'{digest}{suffix}'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._verify(path, digest)
            return path

        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f'.{digest}.', suffix='.tmp'
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
        """Read a blob and verify its digest when its name is content-addressed."""
        artifact_path = Path(path)
        data = artifact_path.read_bytes()
        expected_digest = artifact_path.stem
        if _SHA256_NAME.fullmatch(expected_digest):
            ContentAddressedArtifactStore._verify_bytes(data, expected_digest)
        return data

    def _verify(self, path: Path, expected_digest: str) -> None:
        self._verify_bytes(path.read_bytes(), expected_digest)

    @staticmethod
    def _verify_bytes(data: bytes, expected_digest: str) -> None:
        if hashlib.sha256(data).hexdigest() != expected_digest:
            raise ArtifactIntegrityError(
                'Content-addressed artifact failed its SHA-256 check'
            )
