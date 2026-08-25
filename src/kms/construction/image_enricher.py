"""Describes canonical image nodes with their local source context."""

import asyncio
from typing import Any

import dspy

from kms import config
from kms.core import context_window, images, llm, models, module, state


class ImageEnrichmentSignature(dspy.Signature):
    r"""
    Describe the supplied target image or ordered group of images.

    Use the image pixels as the primary evidence. Use the text immediately
    before and after the target only to resolve local role, captions, and
    references. Do not invent details, treat nearby text as proof of unseen
    visual content, or transcribe text that is not legible. Return one concise
    description of the target visual content and its source-local role.
    """

    images: list[dspy.Image] = dspy.InputField(
        description='The ordered visual assets belonging to one target image node.'
    )
    context_before: str = dspy.InputField(
        description='Ordered source context immediately before the target image.'
    )
    context_after: str = dspy.InputField(
        description='Ordered source context immediately after the target image.'
    )
    description: str = dspy.OutputField(
        description='One concise, evidence-grounded description of the target image.'
    )


class ImageEnricher(module.Module):
    """Describes one post-seam image node."""

    signature = ImageEnrichmentSignature
    record_name = 'image_enrichment'

    def encode(
        self,
        images: list[dspy.Image],
        context_before: str,
        context_after: str,
    ) -> dict[str, object]:
        """Passes native image values and separated text context to DSPy."""
        return {
            'images': images,
            'context_before': context_before,
            'context_after': context_after,
        }

    def decode(self, prediction: Any, **inputs: Any) -> str:
        """Returns one validated image description."""
        return module.require_text(prediction.description, 'description')


def _context_text(nodes: list[context_window.ContextNode]) -> str:
    """Serializes one selected context side without including image bytes."""
    lines: list[str] = []
    for node in nodes:
        node_type = node.type or 'unknown'
        if node.type == models.NodeType.IMAGE or node.assets:
            value = '[IMAGE_OMITTED]'
        else:
            value = node.content or ''
        lines.append(f'[{node.position}] ({node_type}): {value}')
    return '\n'.join(lines)


def _load_images(node: models.SourceNode) -> list[dspy.Image]:
    """Loads every ordered visual asset for one target node."""
    loaded_images: list[dspy.Image] = []
    max_dim = config.get_settings().image.max_dim
    for asset in node.assets:
        try:
            image = images.load_image(asset.path, max_dim=max_dim)
        except OSError as exc:
            raise ValueError(
                f'image asset does not exist: {asset.path}'
            ) from exc
        if image is None:
            raise ValueError(f'image asset does not exist: {asset.path}')
        loaded_images.append(image)
    return loaded_images


class ImageEnrichmentNode:
    """Describes image nodes after seam merging and before splitting."""

    def __init__(
        self,
        enricher: ImageEnricher,
        backward_budget: int | None = None,
        forward_budget: int | None = None,
        max_concurrent_calls: int | None = None,
    ) -> None:
        stage = config.get_settings().stages.image_enrichment
        self._enricher = enricher
        self.backward_budget = (
            backward_budget
            if backward_budget is not None
            else stage.before_budget
        )
        self.forward_budget = (
            forward_budget if forward_budget is not None else stage.after_budget
        )
        self.max_concurrent_calls = (
            max_concurrent_calls
            if max_concurrent_calls is not None
            else stage.max_concurrent_calls
        )

    async def _describe(
        self,
        node: models.SourceNode,
        before: str,
        after: str,
        gate: asyncio.Semaphore,
    ) -> str:
        """Runs one bounded image-description call."""
        async with gate:
            return await self._enricher.aforward(
                images=_load_images(node),
                context_before=before,
                context_after=after,
            )

    async def run(self, current_state: state.State) -> dict[str, object]:
        """Describes every asset-bearing image node in source order."""
        documents = current_state.get('documents', [])
        nodes = current_state.get('nodes', [])
        targets: list[tuple[models.SourceNode, str, str]] = []
        for position, node in enumerate(nodes):
            if node.type != models.NodeType.IMAGE or not node.assets:
                continue
            selected = context_window.select_around(
                nodes,
                [position],
                backward_budget=self.backward_budget,
                forward_budget=self.forward_budget,
                marker='target',
            )
            target_index = next(
                index
                for index, context_node in enumerate(selected)
                if context_node.marker == 'target'
            )
            targets.append(
                (
                    node,
                    _context_text(selected[:target_index]),
                    _context_text(selected[target_index + 1 :]),
                )
            )

        if not targets:
            return {'documents': documents, 'nodes': nodes}

        gate = llm.gate(self.max_concurrent_calls)
        descriptions = await asyncio.gather(
            *(
                self._describe(node, before, after, gate)
                for node, before, after in targets
            )
        )
        for (node, _, _), description in zip(
            targets, descriptions, strict=True
        ):
            node.content = description.strip()
        return {'documents': documents, 'nodes': nodes}
