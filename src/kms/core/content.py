"""Typed multimodal content: ordered text and image parts."""

import base64
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel, Field


def load_image(path: str | None) -> dspy.Image | None:
    """Loads an image file into a dspy.Image, or None if absent."""
    if not path:
        return None
    encoded = base64.b64encode(Path(path).read_bytes()).decode('utf-8')
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
    text: str


class ImagePart(BaseModel):
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
        return cls(parts=[TextPart(text=text)])

    @classmethod
    def from_text_and_pictures(
        cls, text: str, pictures: list[dict]
    ) -> 'Content':
        parts: list[TextPart | ImagePart] = []
        if text:
            parts.append(TextPart(text=text))
        for picture in pictures:
            image = load_image(picture.get('image_path'))
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
            image = load_image(node.image_path)
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
        return self.content.openai_blocks()


def labeled_content_parts(nodes: list[Any]) -> ContentParts:
    return ContentParts(content=labeled_content(nodes))
