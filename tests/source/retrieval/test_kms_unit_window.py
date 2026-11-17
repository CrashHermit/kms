import asyncio
from pathlib import Path

import pytest
from PIL import Image

from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.media.budget import ImagePayloadBudget
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.models import SourceUnitRead, SourceUnitWindow
from kms.source.retrieval.read import SourceReader


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Repository:
    def __init__(self, units: list[SourceUnit]) -> None:
        self.source = Source(uuid='source', key='source')
        self.units = units

    async def read_units(
        self,
        user_uuid: str,
        unit_uuids: list[str],
        *,
        before_units: int = 0,
        after_units: int = 0,
    ) -> list[SourceUnitRead]:
        if user_uuid != 'owner':
            return []
        for index, unit in enumerate(self.units):
            if unit.uuid == unit_uuids[0]:
                return [
                    SourceUnitRead(source=self.source, unit=item)
                    for item in self.units[
                        max(0, index - before_units) : index + after_units + 1
                    ]
                ]
        return []


def _unit(uuid: str, text: str, image: Path | None = None) -> SourceUnit:
    blocks = [SourceBlock(block_type=BlockType.PARAGRAPH, content=text)]
    if image is not None:
        blocks.append(
            SourceBlock(block_type=BlockType.IMAGE, image_path=str(image))
        )
    return SourceUnit(uuid=uuid, kind=SourceUnitKind.CONTENT, blocks=blocks)


def _inputs(window: SourceUnitWindow) -> dict[str, object]:
    return {
        'request': {
            'goal': 'understand',
            'before': window.before,
            'target': window.target,
            'after': window.after,
        }
    }


def _budget(limit: int = 10000) -> TokenBudget:
    return TokenBudget(
        counter=MaterialTokenEstimator(_Counter()), token_limit=limit
    )


def _images(max_images: int = 10) -> ImagePayloadBudget:
    return ImagePayloadBudget(
        max_images=max_images, max_bytes=100000, max_pixels=10000
    )


def test_unit_window_preserves_order_and_native_images(tmp_path: Path) -> None:
    image = tmp_path / 'figure.png'
    Image.new('RGB', (8, 8)).save(image)
    units = [
        _unit('first', 'a'),
        _unit('before', 'b', image),
        _unit('target', 'c', image),
        _unit('after', 'd', image),
        _unit('last', 'e'),
    ]
    reader = SourceReader(_Repository(units))

    window = asyncio.run(
        reader.read_unit_window(
            'owner',
            'target',
            before_tokens=10,
            after_tokens=10,
            budget=_budget(),
            image_budget=_images(),
            request_inputs=_inputs,
        )
    )

    assert window is not None
    assert [unit.uuid for unit in window.before] == ['first', 'before']
    assert window.target.uuid == 'target'
    assert [unit.uuid for unit in window.after] == ['after', 'last']
    for unit in (window.before[-1], window.target, window.after[0]):
        assert [part.type for part in unit.material.parts] == ['text', 'image']
        assert unit.material.images[0].payload == image


def test_window_stops_at_nearest_unit_over_side_or_image_budget(
    tmp_path: Path,
) -> None:
    first = tmp_path / 'first.png'
    second = tmp_path / 'second.png'
    Image.new('RGB', (8, 8)).save(first)
    Image.new('RGB', (8, 8)).save(second)
    units = [
        _unit('far-before', 'x'),
        _unit('before', 'ab', first),
        _unit('target', 't'),
        _unit('after', 'c', second),
        _unit('far-after', 'x'),
    ]
    reader = SourceReader(_Repository(units))

    window = asyncio.run(
        reader.read_unit_window(
            'owner',
            'target',
            before_tokens=4,
            after_tokens=20,
            budget=_budget(),
            image_budget=_images(1),
            request_inputs=_inputs,
        )
    )
    assert window is not None
    assert [unit.uuid for unit in window.before] == ['before']
    assert window.after == []

    window = asyncio.run(
        reader.read_unit_window(
            'owner',
            'target',
            before_tokens=3,
            after_tokens=0,
            budget=_budget(),
            image_budget=_images(),
            request_inputs=_inputs,
        )
    )
    assert window is not None
    assert window.before == []
    assert window.after == []


def test_complete_request_budget_limits_context() -> None:
    units = [_unit('before', 'b'), _unit('target', 't'), _unit('after', 'a')]
    reader = SourceReader(_Repository(units))
    budget = TokenBudget(
        counter=MaterialTokenEstimator(_Counter()),
        token_limit=10000,
        context_token_limit=1,
        format_request=lambda inputs: (
            'x'
            * (
                1
                + len(inputs['request']['before'])
                + len(inputs['request']['after'])
            )
        ),
    )
    window = asyncio.run(
        reader.read_unit_window(
            'owner',
            'target',
            before_tokens=10,
            after_tokens=10,
            budget=budget,
            image_budget=_images(),
            request_inputs=_inputs,
        )
    )
    assert window is not None
    assert window.target.uuid == 'target'
    assert window.before == []
    assert window.after == []


def test_window_checks_complete_target_request_and_ownership() -> None:
    reader = SourceReader(_Repository([_unit('target', 'passage')]))
    assert (
        asyncio.run(
            reader.read_unit_window(
                'other',
                'target',
                before_tokens=10,
                after_tokens=10,
                budget=_budget(),
                image_budget=_images(),
                request_inputs=_inputs,
            )
        )
        is None
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            reader.read_unit_window(
                'owner',
                'target',
                before_tokens=10,
                after_tokens=10,
                budget=_budget(1),
                image_budget=_images(),
                request_inputs=_inputs,
            )
        )
