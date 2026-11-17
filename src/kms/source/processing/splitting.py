"""Mechanical replacement of selected blocks with ordered text fragments."""

from kms.source.models.source import SourceBlock


def apply_block_splits(
    blocks: list[SourceBlock], replacements: dict[int, list[str]]
) -> list[SourceBlock]:
    """Replace selected positions, preserving only containing crop provenance."""
    result: list[SourceBlock] = []
    for position, block in enumerate(blocks):
        pieces = replacements.get(position)
        if pieces is None:
            result.append(block)
            continue
        result.extend(
            SourceBlock(
                block_type=block.block_type,
                content=piece,
                crop_path=block.crop_path,
                crop_sha256=block.crop_sha256,
                crop_bbox=block.crop_bbox,
            )
            for piece in pieces
        )
    return result
