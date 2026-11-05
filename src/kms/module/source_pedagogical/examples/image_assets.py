"""Packaged images used by source-pedagogical demonstrations."""

from pathlib import Path

import dspy


def source_image(name: str) -> dspy.Image:
    """Load a checked-in source-processing image for a DSPy example."""
    asset_path = (
        Path(__file__).parents[2]
        / 'source_processing'
        / 'examples'
        / 'assets'
        / name
    )
    return dspy.Image.from_path(str(asset_path))
