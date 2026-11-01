"""Serialize session workspaces while keeping image bytes in source files."""

import asyncio
import json
from pathlib import Path
from typing import Any

from kms2.core.model.teaching.workspace import TeachingEvidenceWorkspace


def serialize_workspace(workspace: TeachingEvidenceWorkspace) -> str:
    """Serialize workspace state without duplicating native image payloads."""
    return workspace.model_dump_json(
        exclude={
            'image_registry': {'__all__': {'data'}},
            'receipts': {'__all__': {'data'}},
        }
    )


async def restore_workspace(
    workspace_json: str | None,
) -> TeachingEvidenceWorkspace:
    """Restore the session workspace and rehydrate inspected original images."""
    if workspace_json is None:
        return TeachingEvidenceWorkspace()
    payload: dict[str, Any] = json.loads(workspace_json)
    await asyncio.to_thread(_restore_image_data, payload)
    return TeachingEvidenceWorkspace.model_validate(payload)


def _restore_image_data(payload: dict[str, Any]) -> None:
    """Read each inspected asset once and restore all of its workspace uses."""
    cache: dict[str, bytes] = {}

    def image_data(path: str) -> bytes:
        if path not in cache:
            cache[path] = Path(path).read_bytes()
        return cache[path]

    for position in payload['inspected_image_positions']:
        entry = payload['image_registry'][position]
        entry['data'] = image_data(entry['asset']['path'])
    for receipt in payload['receipts']:
        if receipt['kind'] == 'image':
            receipt['data'] = image_data(receipt['asset']['path'])
