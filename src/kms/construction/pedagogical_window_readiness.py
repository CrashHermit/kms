"""Decides whether a node window is ready for pedagogical span extraction."""

import logging
from typing import cast

import dspy

from kms.core import context_window, models, module

logger = logging.getLogger(__name__)


class PedagogicalWindowReadinessSignature(dspy.Signature):
    r"""
    Decide only whether the supplied node window is ready for pedagogical span
    extraction.

    The router decides whether the supplied run of nodes is sufficient to
    determine all pedagogical-unit boundaries that start within it. It returns
    False when the final supplied node may continue a labelled or numbered unit,
    proof, worked solution, subpart sequence, prescribed procedure, or OCR
    fragment. It returns True when the final node closes the current unit or no
    unit is present.

    The router never emits spans, classifies unit kinds, or owns nodes.

    Return only a boolean. Answer only True or False.
    """

    current_nodes: list[models.NodeInput] = dspy.InputField(
        description=(
            'Ordered local node records with one-based index, node_type, and '
            'text. Use only index for span endpoints; text numbers are content.'
        )
    )
    is_complete: bool = dspy.OutputField(
        description=(
            'True only when the supplied window closes every pedagogical unit '
            'that starts within it.'
        )
    )


class PedagogicalWindowReadinessRouter(module.Module):
    """Routes span extraction only over complete pedagogical windows."""

    signature = PedagogicalWindowReadinessSignature
    record_name = 'pedagogical_window_readiness_router'

    def encode(self, **inputs: object) -> dict[str, object]:
        """Projects context nodes as one-based structured model inputs."""
        current_nodes = cast(
            list[context_window.ContextNode], inputs['current_nodes']
        )
        return {
            'current_nodes': [
                context_window.node_input(node, local_index)
                for local_index, node in enumerate(current_nodes)
            ]
        }

    def decode(self, prediction: dspy.Prediction, **inputs: object) -> bool:
        """Returns a strictly boolean readiness decision."""
        value = prediction.is_complete
        if type(value) is not bool:
            raise TypeError('is_complete must be a bool')
        return value
