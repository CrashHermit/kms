import asyncio

import dspy
import pytest

from kms2.core.model.block import SourceBlock
from kms2.core.model.page import SourcePage
from kms2.core.model.source import Source
from kms2.core.model.source_processing.formatting import (
    FormattingInput,
    FormattingOutput,
    FormattingRequest,
)
from kms2.core.windowing import TokenBudget
from kms2.langgraph.source_processing.state import SourceProcessingState
from kms2.module.source_processing import formatting
from kms2.node.source_processing.formatting import FormattingNode


class _Counter:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.texts.extend(texts)
        return [len(text) for text in texts]


class _ScriptedPredictor:
    def __init__(self) -> None:
        self.calls: list[list[FormattingInput]] = []
        self.output_order: list[int] | None = None

    def __call__(self, *, blocks: list[FormattingInput]) -> dspy.Prediction:
        return self._prediction(blocks)

    async def acall(self, *, blocks: list[FormattingInput]) -> dspy.Prediction:
        return self._prediction(blocks)

    def _prediction(self, blocks: list[FormattingInput]) -> dspy.Prediction:
        self.calls.append(blocks)
        outputs = [
            FormattingOutput(
                position=block.position,
                formatted_content=f'formatted {block.content}',
            )
            for block in blocks
        ]
        if self.output_order is not None:
            by_position = {output.position: output for output in outputs}
            outputs = [by_position[position] for position in self.output_order]
        return dspy.Prediction(formatted_blocks=outputs)


def _state(pages: list[SourcePage]) -> SourceProcessingState:
    return SourceProcessingState(
        pdf_path='document.pdf',
        source=Source(uuid='source-1', key='document.pdf'),
        corrected_pages=pages,
    )


def _node(
    predictor: _ScriptedPredictor,
    *,
    token_limit: int = 10_000,
    batch_size: int = 32,
    counter: _Counter | None = None,
) -> tuple[FormattingNode, _Counter]:
    token_counter = counter or _Counter()
    module = formatting.FormatterModule(predictor)
    return (
        FormattingNode(
            module,
            budget=TokenBudget(counter=token_counter, token_limit=token_limit),
            batch_size=batch_size,
        ),
        token_counter,
    )


@pytest.mark.parametrize(
    'outputs',
    [
        [
            FormattingOutput(position=0, formatted_content='first'),
            FormattingOutput(position=0, formatted_content='duplicate'),
        ],
        [FormattingOutput(position=0, formatted_content='missing second')],
        [
            FormattingOutput(position=0, formatted_content='first'),
            FormattingOutput(position=2, formatted_content='out of range'),
        ],
    ],
    ids=['duplicate-position', 'missing-position', 'out-of-range-position'],
)
def test_module_rejects_non_bijective_output_positions(outputs):
    class Predictor:
        def __call__(self, **kwargs: object) -> dspy.Prediction:
            return dspy.Prediction(formatted_blocks=outputs)

    module = formatting.FormatterModule(Predictor())
    with pytest.raises(ValueError):
        module(
            blocks=[
                FormattingInput(position=0, content='first'),
                FormattingInput(position=1, content='second'),
            ]
        )


def test_dispatch_worker_and_collect_batch_across_pages_by_original_identity():
    first = SourceBlock(
        uuid='first',
        block_type='paragraph',
        content='first content',
        assets=[{'path': 'first.png'}],
    )
    skipped = SourceBlock(uuid='empty', block_type='paragraph', content='')
    image = SourceBlock(uuid='image', block_type='image', content='image label')
    last = SourceBlock(
        uuid='last',
        block_type='paragraph',
        content='last content',
        assets=[{'path': 'last.png'}],
    )
    state = _state(
        [
            SourcePage(index=4, blocks=[first, skipped]),
            SourcePage(index=9, blocks=[image, last]),
        ]
    )
    predictor = _ScriptedPredictor()
    predictor.output_order = [1, 0]
    node, _ = _node(predictor)

    sends = node.dispatch(state)
    assert isinstance(sends, list)
    assert len(sends) == 1
    requests = sends[0].arg['formatting_requests']
    assert requests == [
        FormattingRequest(
            page_index=4, block_position=0, content='first content'
        ),
        FormattingRequest(
            page_index=9, block_position=1, content='last content'
        ),
    ]
    worker_result = asyncio.run(node.worker(sends[0].arg))
    state.formatting_results = worker_result['formatting_results']
    formatted_pages = node.collect(state)['formatted_pages']

    assert [block.content for block in formatted_pages[0].blocks] == [
        'formatted first content',
        '',
    ]
    assert [block.content for block in formatted_pages[1].blocks] == [
        'image label',
        'formatted last content',
    ]
    assert formatted_pages[0].blocks[0].uuid == first.uuid
    assert formatted_pages[0].blocks[0].assets == first.assets
    assert formatted_pages[0].blocks[1] is skipped
    assert formatted_pages[1].blocks[0] is image
    assert formatted_pages[1].blocks[1].uuid == last.uuid
    assert formatted_pages[1].blocks[1].assets == last.assets
    assert predictor.calls[0] == [
        FormattingInput(position=0, content='first content'),
        FormattingInput(position=1, content='last content'),
    ]


def test_dispatch_splits_on_token_budget_and_uses_local_indexes_per_worker():
    pages = [
        SourcePage(
            index=page_index,
            blocks=[
                SourceBlock(
                    uuid=f'block-{page_index}',
                    block_type='paragraph',
                    content=f'content-{page_index}',
                )
            ],
        )
        for page_index in range(3)
    ]
    state = _state(pages)
    predictor = _ScriptedPredictor()
    counter = _Counter()
    node, _ = _node(predictor, token_limit=55, counter=counter)

    sends = node.dispatch(state)

    assert isinstance(sends, list)
    assert len(counter.texts) == 4
    assert len(sends) > 1
    assert all(len(send.arg['formatting_requests']) == 1 for send in sends)
    results = [
        result
        for send in sends
        for result in asyncio.run(node.worker(send.arg))['formatting_results']
    ]
    state.formatting_results = results
    assert [
        page.blocks[0].content
        for page in node.collect(state)['formatted_pages']
    ] == [
        'formatted content-0',
        'formatted content-1',
        'formatted content-2',
    ]
    assert [call[0].position for call in predictor.calls] == [0] * len(sends)


def test_batch_size_one_runs_one_eligible_block_per_model_call():
    state = _state(
        [
            SourcePage(
                index=0,
                blocks=[
                    SourceBlock(uuid='a', block_type='paragraph', content='a'),
                    SourceBlock(uuid='b', block_type='paragraph', content='b'),
                ],
            )
        ]
    )
    predictor = _ScriptedPredictor()
    node, _ = _node(predictor, batch_size=1)

    sends = node.dispatch(state)

    assert isinstance(sends, list)
    assert [len(send.arg['formatting_requests']) for send in sends] == [1, 1]
    state.formatting_results = [
        result
        for send in sends
        for result in asyncio.run(node.worker(send.arg))['formatting_results']
    ]
    assert [len(call) for call in predictor.calls] == [1, 1]
    assert [[block.position for block in call] for call in predictor.calls] == [
        [0],
        [0],
    ]


def test_dispatch_skips_empty_and_image_blocks_without_changing_them():
    empty = SourceBlock(uuid='empty', block_type='paragraph', content='')
    image = SourceBlock(uuid='image', block_type='image', content='caption')
    state = _state([SourcePage(index=0, blocks=[empty, image])])
    node, _ = _node(_ScriptedPredictor())

    assert node.dispatch(state) == 'formatter_collect'
    result = node.collect(state)['formatted_pages'][0].blocks
    assert result == [empty, image]
    assert result[0] is empty
    assert result[1] is image
