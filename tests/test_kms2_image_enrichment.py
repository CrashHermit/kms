import asyncio

import dspy

from kms2.config import ContextWindowSettings
from kms2.core.model import (
    BlockType,
    ImageEnrichmentRequest,
    ImageEnrichmentResult,
    Source,
    SourceBlock,
    SourceBlockContext,
    SourceContextWindow,
    SourcePage,
    VisualAsset,
)
from kms2.langgraph.source.state import SourceState
from kms2.module.source import image_enrichment
from kms2.node.source.image_enrichment import ImageEnrichmentNode


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(description='described image')

    async def acall(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(description='described image')


def _block(
    *,
    block_type: BlockType,
    content: str | None = None,
    assets: list[str] | None = None,
) -> SourceBlock:
    return SourceBlock(
        block_type=block_type,
        content=content,
        assets=[VisualAsset(path=path) for path in assets or []],
    )


def _state(pages: list[SourcePage]) -> SourceState:
    return SourceState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        image_seam_pages=pages,
    )


def _context_window() -> ContextWindowSettings:
    return ContextWindowSettings(backward_budget=200, forward_budget=200)


def test_signature_exposes_image_enrichment_contract() -> None:
    assert set(image_enrichment.ImageEnrichmentSignature.input_fields) == {
        'target_images',
        'context_before',
        'context_after',
    }
    assert set(image_enrichment.ImageEnrichmentSignature.output_fields) == {
        'description'
    }


def test_module_preserves_ordered_target_images_sync_and_async():
    predictor = _Predictor()
    module = image_enrichment.ImageEnrichmentModule(predictor)
    source_block = _block(
        block_type=BlockType.IMAGE,
        assets=[
            'data:image/png;base64,AAAA',
            'data:image/png;base64,BBBB',
        ],
    )
    before = [
        SourceBlockContext(block_type=BlockType.PARAGRAPH, content='before')
    ]
    after = [
        SourceBlockContext(block_type=BlockType.PARAGRAPH, content='after')
    ]

    assert (
        module(
            source_block=source_block,
            context_before=before,
            context_after=after,
        )
        == 'described image'
    )
    assert (
        asyncio.run(
            module.aforward(
                source_block=source_block,
                context_before=before,
                context_after=after,
            )
        )
        == 'described image'
    )

    assert [image.url for image in predictor.calls[0]['target_images']] == [
        'data:image/png;base64,AAAA',
        'data:image/png;base64,BBBB',
    ]
    assert [image.url for image in predictor.calls[1]['target_images']] == [
        'data:image/png;base64,AAAA',
        'data:image/png;base64,BBBB',
    ]


def test_dspy_formats_ordered_image_list_as_native_image_blocks():
    messages = dspy.ChatAdapter().format(
        image_enrichment.ImageEnrichmentSignature,
        demos=[],
        inputs={
            'target_images': [
                dspy.Image(url='data:image/png;base64,AAAA'),
                dspy.Image(url='data:image/png;base64,BBBB'),
            ],
            'context_before': [
                SourceBlockContext(
                    block_type=BlockType.PARAGRAPH, content='before'
                )
            ],
            'context_after': [
                SourceBlockContext(
                    block_type=BlockType.PARAGRAPH, content='after'
                )
            ],
        },
    )
    blocks = messages[1]['content']
    assert sum(block['type'] == 'image_url' for block in blocks) == 2


def test_dispatch_targets_only_images_but_keeps_generic_neighbor_context():
    pages = [
        SourcePage(
            index=0,
            blocks=[
                _block(block_type=BlockType.PARAGRAPH, content='before'),
                _block(block_type=BlockType.IMAGE, assets=['target.png']),
                _block(block_type=BlockType.IMAGE, content='nearby image'),
                _block(block_type=BlockType.PARAGRAPH, content='after'),
            ],
        )
    ]
    node = ImageEnrichmentNode(
        object(),
        _context_window(),
    )

    sends = node.dispatch(_state(pages))

    assert isinstance(sends, list)
    assert len(sends) == 2
    first = sends[0].arg['image_enrichment_request']
    assert first.flat_position == 1
    assert [item.content for item in first.window.context_before] == ['before']
    assert [item.content for item in first.window.context_after] == [
        'nearby image',
        'after',
    ]
    assert first.window.context_after[0].block_type is BlockType.IMAGE


def test_worker_passes_merged_assets_as_one_ordered_target():
    predictor = _Predictor()
    module = image_enrichment.ImageEnrichmentModule(predictor)
    node = ImageEnrichmentNode(
        module,
        _context_window(),
    )
    source_block = _block(
        block_type=BlockType.IMAGE,
        assets=[
            'data:image/png;base64,AAAA',
            'data:image/png;base64,BBBB',
        ],
    )
    request = {
        'image_enrichment_request': ImageEnrichmentRequest(
            flat_position=3,
            source_block=source_block,
            window=SourceContextWindow(),
        )
    }

    result = asyncio.run(node.worker(request))

    assert [image.url for image in predictor.calls[0]['target_images']] == [
        'data:image/png;base64,AAAA',
        'data:image/png;base64,BBBB',
    ]
    assert result['image_enrichment_results'] == [
        ImageEnrichmentResult(flat_position=3, description='described image')
    ]


def test_no_images_route_directly_to_collect():
    node = ImageEnrichmentNode(
        object(),
        _context_window(),
    )
    state = _state(
        [
            SourcePage(
                index=0,
                blocks=[_block(block_type=BlockType.PARAGRAPH, content='x')],
            )
        ]
    )

    assert node.dispatch(state) == 'image_enrichment_collect'


def test_collect_applies_out_of_order_results_without_losing_metadata():
    first = _block(block_type=BlockType.PARAGRAPH, content='first')
    image = _block(
        block_type=BlockType.IMAGE,
        assets=['target.png'],
    )
    second = _block(block_type=BlockType.PARAGRAPH, content='second')
    pages = [
        SourcePage(index=0, blocks=[first, image]),
        SourcePage(index=1, blocks=[second]),
    ]
    state = _state(pages).model_copy(
        update={
            'image_enrichment_results': [
                ImageEnrichmentResult(
                    flat_position=1, description='description'
                )
            ]
        }
    )

    result = ImageEnrichmentNode(
        object(),
        _context_window(),
    ).collect(state)
    enriched_pages = result['image_enriched_pages']

    assert [page.index for page in enriched_pages] == [0, 1]
    assert enriched_pages[0].blocks[0] is first
    assert enriched_pages[0].blocks[1].content == 'description'
    assert enriched_pages[0].blocks[1].block_type is BlockType.IMAGE
    assert enriched_pages[0].blocks[1].assets == image.assets
    assert enriched_pages[1].blocks[0] is second
