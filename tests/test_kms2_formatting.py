import asyncio

import dspy

from kms2.core.model.formatting import FormattingRequest, FormattingResult
from kms2.core.model.source import Source, SourceBlock, SourcePage
from kms2.langgraph.source.state import SourceState
from kms2.module.source import formatting
from kms2.node.source.formatting import FormattingNode


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(formatted_content=r'inline \(x\)')

    async def acall(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(formatted_content=r'display \[x^2\]')


def _module() -> tuple[formatting.FormatterModule, _Predictor]:
    module = formatting.FormatterModule(dspy.LM('openai/dummy', api_key='test'))
    predictor = _Predictor()
    module.predictor = predictor
    return module, predictor


def _state(contents: list[SourceBlock]) -> SourceState:
    return SourceState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        corrected_pages=[SourcePage(index=3, blocks=contents)],
    )


def test_signature_exposes_full_block_contract():
    assert set(formatting.FormatterSignature.input_fields) == {'content'}
    assert set(formatting.FormatterSignature.output_fields) == {
        'formatted_content'
    }


def test_forward_passes_content_and_normalizes_math_delimiters():
    module, predictor = _module()

    result = module(content='original')

    assert result == 'inline $x$'
    assert predictor.calls == [{'content': 'original'}]


def test_aforward_passes_content_and_normalizes_math_delimiters():
    module, predictor = _module()

    result = asyncio.run(module.aforward(content='original'))

    assert result == 'display $$x^2$$'
    assert predictor.calls == [{'content': 'original'}]


def test_dispatch_indexes_non_empty_content_and_skips_empty_content():
    node = FormattingNode(object())
    sends = node.dispatch(
        _state(
            [
                SourceBlock(
                    uuid='content-0',
                    block_type='text',
                ),
                SourceBlock(
                    uuid='content-1',
                    block_type='text',
                    content='',
                ),
                SourceBlock(
                    uuid='content-2',
                    block_type='text',
                    content='first',
                ),
                SourceBlock(
                    uuid='content-3',
                    block_type='text',
                    content='second',
                ),
            ]
        )
    )

    assert [
        (
            send.node,
            send.arg['formatting_request'].page_index,
            send.arg['formatting_request'].block_position,
        )
        for send in sends
    ] == [
        ('formatter_worker', 3, 2),
        ('formatter_worker', 3, 3),
    ]
    assert [send.arg['formatting_request'].content for send in sends] == [
        'first',
        'second',
    ]


def test_dispatch_excludes_image_blocks_from_text_formatting():
    node = FormattingNode(object())
    state = _state(
        [
            SourceBlock(
                uuid='image-1',
                block_type='image',
                content='OCR image label',
            )
        ]
    )

    assert node.dispatch(state) == 'formatter_collect'


def test_dispatch_routes_directly_to_collect_without_eligible_content():
    node = FormattingNode(object())

    assert (
        node.dispatch(
            _state(
                [
                    SourceBlock(
                        uuid='content-0',
                        block_type='text',
                    ),
                    SourceBlock(
                        uuid='content-1',
                        block_type='text',
                        content='',
                    ),
                ]
            )
        )
        == 'formatter_collect'
    )


def test_worker_returns_one_positioned_formatting_result():
    class _Formatter:
        async def acall(self, *, content: str) -> str:
            assert content == 'original'
            return 'formatted'

    node = FormattingNode(_Formatter())
    request = {
        'formatting_request': FormattingRequest(
            page_index=3,
            block_position=4,
            content='original',
        )
    }

    result = asyncio.run(node.worker(request))

    assert result == {
        'formatting_results': [
            FormattingResult(
                page_index=3,
                block_position=4,
                formatted_content='formatted',
            )
        ]
    }


def test_collect_orders_results_and_preserves_source_content_identity_data():
    first = SourceBlock(
        uuid='content-1',
        block_type='text',
        content='first',
        assets=[{'path': 'images/first.png'}],
    )
    skipped = SourceBlock(
        uuid='content-2',
        block_type='text',
        content='',
    )
    third = SourceBlock(
        uuid='content-3',
        block_type='text',
        content='third',
        assets=[{'path': 'images/third.png'}],
    )
    state = _state([first, skipped, third])
    state.formatting_results = [
        FormattingResult(
            page_index=3,
            block_position=2,
            formatted_content='formatted third',
        ),
        FormattingResult(
            page_index=3,
            block_position=0,
            formatted_content='formatted first',
        ),
    ]

    result = FormattingNode(object()).collect(state)

    formatted_contents = result['formatted_pages'][0].blocks
    assert [content.content for content in formatted_contents] == [
        'formatted first',
        '',
        'formatted third',
    ]
    assert formatted_contents[0].uuid == 'content-1'
    assert formatted_contents[0].assets == first.assets
    assert formatted_contents[1] is skipped
    assert formatted_contents[2].uuid == 'content-3'
    assert formatted_contents[2].assets == third.assets
