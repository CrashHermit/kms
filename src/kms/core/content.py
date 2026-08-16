"""Typed multimodal content: ordered text and image parts."""

import base64
import io
from pathlib import Path
from typing import Any

import dspy
from PIL import Image
from pydantic import BaseModel, Field

from kms import config


def _resize_bytes(data: bytes, max_dim: int) -> bytes:
    """Resizes image bytes so the longest side is at most max_dim.

    Images already within the cap are returned unchanged.
    """
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= max_dim:
            return data
        scale = max_dim / longest
        resized = img.resize(
            (round(width * scale), round(height * scale)),
            Image.Resampling.LANCZOS,
        )
        buf = io.BytesIO()
        resized.save(buf, format='PNG')
        return buf.getvalue()


def load_image(
    path: str | None, max_dim: int | None = None
) -> dspy.Image | None:
    """Loads an image file into a dspy.Image, or None if absent.

    Args:
        path: Path to the image file.
        max_dim: Optional longest-side cap in pixels; larger images are
            resized down while preserving aspect ratio.
    """
    if not path:
        return None
    data = Path(path).read_bytes()
    if max_dim is not None:
        data = _resize_bytes(data, max_dim)
    encoded = base64.b64encode(data).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


def image_url(image: dspy.Image) -> str:
    """Converts a dspy.Image to a data URI or URL for payloads."""
    url = image.url or ''
    if url.startswith('data:') or url.startswith('http'):
        return url
    path = Path(url)
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode('utf-8')
        return f'data:image/png;base64,{encoded}'
    return url


class TextPart(BaseModel):
    """A single text part of a Content value."""

    text: str


class ImagePart(BaseModel):
    """A single image part of a Content value."""

    image: dspy.Image


class Content(BaseModel):
    """An ordered list of text and image parts.

    The canonical representation for anything that mixes prose with
    figures: search queries, statement text with pictures, retrieval
    candidates.  Use the from_* constructors to assemble content from
    the forms other modules produce, and the block/render converters
    to feed it to LLMs, embedders, or labels.
    """

    parts: list[TextPart | ImagePart] = Field(default_factory=list)

    @classmethod
    def from_parts(cls, parts: list[str | dspy.Image]) -> 'Content':
        """Builds Content from a flat list of strings and images."""
        return cls(
            parts=[
                TextPart(text=part)
                if isinstance(part, str)
                else ImagePart(image=part)
                for part in parts
            ]
        )

    @classmethod
    def from_text(cls, text: str) -> 'Content':
        """Builds Content from a single text string."""
        return cls(parts=[TextPart(text=text)])

    @classmethod
    def from_text_and_pictures(
        cls, text: str, pictures: list[dict]
    ) -> 'Content':
        """Builds Content from text plus picture records with image paths."""
        parts: list[TextPart | ImagePart] = []
        if text:
            parts.append(TextPart(text=text))
        for picture in pictures:
            image = load_image(
                picture.get('image_path'),
                max_dim=config.get_settings().image.max_dim,
            )
            if image:
                parts.append(ImagePart(image=image))
        return cls(parts=parts)

    def render(self) -> str:
        """Renders the parts as a text label; images appear as [image]."""
        return ' '.join(
            part.text if isinstance(part, TextPart) else '[image]'
            for part in self.parts
        )

    def openai_blocks(self) -> list[dict[str, Any]]:
        """Returns OpenAI-style content blocks for LLM payloads."""
        blocks: list[dict[str, Any]] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                blocks.append({'type': 'text', 'text': part.text})
            else:
                blocks.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': image_url(part.image)},
                    }
                )
        return blocks

    def embedding_blocks(self) -> list[dict[str, Any]]:
        """Returns Voyage-style content blocks for the embedding API."""
        blocks: list[dict[str, Any]] = []
        for part in self.parts:
            if isinstance(part, TextPart):
                blocks.append({'type': 'text', 'text': part.text})
            else:
                url = image_url(part.image)
                if url.startswith('data:'):
                    blocks.append({'type': 'image_base64', 'image_base64': url})
                else:
                    blocks.append({'type': 'image_url', 'image_url': url})
        return blocks


def labeled_content(nodes: list[Any]) -> Content:
    """Builds a labelled text+image Content from node-like objects.

    Each item exposes `position`, `type`, `content`, and `image_path`.
    A text node renders as ``[position] (type): content``; an image node
    renders as a ``[position] (type):`` label followed by the image.
    """
    parts: list[TextPart | ImagePart] = []
    for node in nodes:
        label = f'[{node.position}] ({node.type})'
        if node.image_path:
            image = load_image(
                node.image_path,
                max_dim=config.get_settings().image.max_dim,
            )
            parts.append(TextPart(text=label))
            if image:
                parts.append(ImagePart(image=image))
        else:
            parts.append(TextPart(text=f'{label}: {node.content or ""}'))
    return Content(parts=parts)


class ContentParts(dspy.Type):
    """The dspy input type for a Content value."""

    content: Content

    def format(self) -> list[dict[str, Any]]:
        """Returns OpenAI-style content blocks for LLM payloads."""
        return self.content.openai_blocks()


def labeled_content_parts(nodes: list[Any]) -> ContentParts:
    """Builds a ContentParts dspy input from node-like objects."""
    return ContentParts(content=labeled_content(nodes))
