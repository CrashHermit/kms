"""Detects visual blocks continued across page seams."""

import logging
from typing import Any

import dspy
from langgraph.types import Send

from kms.core import images, models, module, state

logger = logging.getLogger(__name__)

_APPARATUS = {'bibliographic', 'note'}


class ImageSeamSignature(dspy.Signature):
    """
    Decide whether two supplied images are one visual object split by a page
    seam. Return TRUE only when the bottom of the first image continues into
    the top of the second image. Return FALSE for two separate figures,
    unrelated images, or images where continuation is uncertain. Do not
    describe, transcribe, rewrite, stitch, or otherwise modify either image.
    """

    top_image: dspy.Image = dspy.InputField(
        description='The visual asset at the bottom of the top page.'
    )
    bottom_image: dspy.Image = dspy.InputField(
        description='The visual asset at the top of the bottom page.'
    )
    is_continuation: bool = dspy.OutputField(
        description='True only when both images are parts of one split visual object.'
    )


class ImageSeamMerger(module.Module):
    """Decides whether two image nodes form one continued visual object."""

    signature = ImageSeamSignature
    record_name = 'image_seam_merger'

    def encode(
        self,
        top_node: models.SourceNode,
        bottom_node: models.SourceNode,
    ) -> dict[str, dspy.Image]:
        """Loads the two eligible source assets for the vision boundary."""
        top_asset = top_node.assets[0]
        bottom_asset = bottom_node.assets[0]
        try:
            top_image = images.load_image(top_asset.path)
        except OSError as exc:
            raise ValueError(
                f'image asset does not exist: {top_asset.path}'
            ) from exc
        try:
            bottom_image = images.load_image(bottom_asset.path)
        except OSError as exc:
            raise ValueError(
                f'image asset does not exist: {bottom_asset.path}'
            ) from exc
        if top_image is None:
            raise ValueError(f'image asset does not exist: {top_asset.path}')
        if bottom_image is None:
            raise ValueError(f'image asset does not exist: {bottom_asset.path}')
        return {'top_image': top_image, 'bottom_image': bottom_image}

    def decode(self, prediction: Any, **inputs: Any) -> bool:
        """Returns the validated visual-continuation decision."""
        return module.require_bool(
            prediction.is_continuation, 'is_continuation'
        )


def _image_mergeable(node: models.SourceNode) -> bool:
    """Returns whether one source node is eligible for image seam judgment."""
    return bool(
        node.type == models.NodeType.IMAGE
        and len(node.assets) == 1
        and not (node.content or '').strip()
    )


def _image_edge_index(
    nodes: list[models.SourceNode], *, reverse: bool
) -> int | None:
    """Returns the nearest eligible image edge, skipping apparatus nodes."""
    indices = range(len(nodes) - 1, -1, -1) if reverse else range(len(nodes))
    for index in indices:
        node = nodes[index]
        if node.type in _APPARATUS:
            continue
        return index if _image_mergeable(node) else None
    return None


def _pairs(
    documents: list[models.Document], parity: int
) -> list[tuple[models.Document, models.Document]]:
    """Returns adjacent document pairs with eligible image edges."""
    pairs: list[tuple[models.Document, models.Document]] = []
    for index in range(len(documents) - 1):
        top = documents[index]
        bottom = documents[index + 1]
        if top.index % 2 != parity:
            continue
        if _image_edge_index(top.nodes, reverse=True) is None:
            continue
        if _image_edge_index(bottom.nodes, reverse=False) is None:
            continue
        pairs.append((top, bottom))
    return pairs


async def _merge_pair(
    merger: ImageSeamMerger,
    top: models.Document,
    bottom: models.Document,
) -> list[tuple[int, list[models.SourceNode]]]:
    """Judges and applies one eligible image seam."""
    top_nodes = list(top.nodes)
    bottom_nodes = list(bottom.nodes)
    top_index = _image_edge_index(top_nodes, reverse=True)
    bottom_index = _image_edge_index(bottom_nodes, reverse=False)
    if top_index is None or bottom_index is None:
        return [(top.index, top_nodes), (bottom.index, bottom_nodes)]

    top_image = top_nodes[top_index]
    bottom_image = bottom_nodes[bottom_index]
    is_continuation = await merger.aforward(
        top_node=top_image,
        bottom_node=bottom_image,
    )
    logger.debug(
        'image seam %d/%d: %s | %r + %r',
        top.index,
        bottom.index,
        'merged' if is_continuation else 'left separate',
        top_image.assets[0].path,
        bottom_image.assets[0].path,
    )
    if is_continuation:
        top_image.assets = [*top_image.assets, *bottom_image.assets]
        del bottom_nodes[bottom_index]

    return [(top.index, top_nodes), (bottom.index, bottom_nodes)]


class ImageSeamMergerNode:
    """Two-pass graph node that merges confirmed image seam continuations."""

    def __init__(self, merger: ImageSeamMerger) -> None:
        self.merger = merger

    def dispatch_even(self, current_state: state.State) -> list[Send] | str:
        """Sends one worker per even-indexed eligible image pair."""
        pairs = _pairs(current_state.get('documents', []), parity=0)
        sends = [
            Send('image_seam_even_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'image_seam_even_collect'

    def dispatch_odd(self, current_state: state.State) -> list[Send] | str:
        """Sends one worker per odd-indexed eligible image pair."""
        pairs = _pairs(current_state.get('documents', []), parity=1)
        sends = [
            Send('image_seam_odd_worker', {'top': top, 'bottom': bottom})
            for top, bottom in pairs
        ]
        return sends or 'image_seam_odd_collect'

    async def even_worker(
        self, current_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Merges one even-indexed image pair."""
        result = await _merge_pair(
            self.merger, current_state['top'], current_state['bottom']
        )
        return {'image_seam_even_results': result}

    async def odd_worker(self, current_state: dict[str, Any]) -> dict[str, Any]:
        """Merges one odd-indexed image pair."""
        result = await _merge_pair(
            self.merger, current_state['top'], current_state['bottom']
        )
        return {'image_seam_odd_results': result}

    def _collect(
        self, current_state: state.State, channel: str
    ) -> dict[str, Any]:
        """Applies one image worker result channel to the documents."""
        documents = current_state['documents']
        by_index = dict(current_state.get(channel, []))
        for document in documents:
            if document.index in by_index:
                document.nodes = by_index[document.index]
        return {'documents': documents}

    def even_collect(self, current_state: state.State) -> dict[str, Any]:
        """Collects the even image pass."""
        return self._collect(current_state, 'image_seam_even_results')

    def odd_collect(self, current_state: state.State) -> dict[str, Any]:
        """Collects the odd image pass and flattens the source stream."""
        result = self._collect(current_state, 'image_seam_odd_results')
        documents = result['documents']
        return {
            'documents': documents,
            'nodes': models.flatten_documents(documents),
        }
