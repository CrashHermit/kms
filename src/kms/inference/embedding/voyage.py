"""Translate embedding materials into Voyage's multimodal input format."""

from collections.abc import Mapping

from kms.core.embedding import EmbeddingRequest


def voyage_inputs(
    request: EmbeddingRequest, image_uris: Mapping[tuple[str, ...], str]
) -> list[dict[str, object]]:
    """Build Voyage content parts while preserving material and part order."""
    inputs: list[dict[str, object]] = []
    for material in request.materials:
        parts: list[dict[str, object]] = []
        for part in material.parts:
            if part.type == 'text':
                parts.append({'type': 'text', 'text': part.text})
            else:
                # LiteLLM strips the data URL prefix when normalizing image_url.
                parts.append(
                    {
                        'type': 'image_base64',
                        'image_base64': image_uris[part.asset.key],
                    }
                )
        inputs.append({'content': parts})
    return inputs
