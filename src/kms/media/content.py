"""Collect native image assets at inference boundaries."""

from collections.abc import Mapping

from pydantic import BaseModel

from kms.media.models import ImageAsset, Material


def collect_image_assets(value: object) -> list[ImageAsset]:
    """Collect unique image assets in encounter order from nested values."""
    assets: list[ImageAsset] = []
    keys: set[tuple[str, ...]] = set()

    def visit(item: object) -> None:
        if isinstance(item, ImageAsset):
            if item.key not in keys:
                keys.add(item.key)
                assets.append(item)
        elif isinstance(item, Material):
            for part in item.parts:
                visit(part)
        elif isinstance(item, BaseModel):
            for name, field in type(item).model_fields.items():
                if not field.exclude:
                    visit(getattr(item, name))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return assets
