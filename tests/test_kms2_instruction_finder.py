import asyncio

from kms2.config import InstructionFinderSettings
from kms2.core.model import BlockType, Source, SourceBlock, SourcePage
from kms2.langgraph.source.state import SourceState
from kms2.node.source.instruction_finder import InstructionFinderNode


class StartRouter:
    def __init__(self, starts: set[str]) -> None:
        self.starts = starts
        self.calls: list[str] = []

    async def aforward(self, *, target_block, **_kwargs) -> bool:
        self.calls.append(target_block.content)
        return target_block.content in self.starts


class BoundaryRouter:
    def __init__(self, boundaries: set[str]) -> None:
        self.boundaries = boundaries
        self.calls: list[str] = []

    async def aforward(self, *, candidate_block, **_kwargs) -> bool:
        self.calls.append(candidate_block.content)
        return candidate_block.content in self.boundaries


def _state(*contents: str) -> SourceState:
    blocks = [
        SourceBlock(
            uuid=f'block-{index}',
            block_type=BlockType.PARAGRAPH,
            content=content,
        )
        for index, content in enumerate(contents)
    ]
    return SourceState(
        pdf_path='source.pdf',
        source=Source(key='source-1'),
        split_pages=[SourcePage(index=0, blocks=blocks)],
    )


def test_instruction_finder_reconsiders_exclusive_boundary_candidate():
    start_router = StartRouter({'instruction one', 'instruction two'})
    boundary_router = BoundaryRouter({'instruction two'})
    node = InstructionFinderNode(
        start_router,
        boundary_router,
        InstructionFinderSettings(),
    )

    result = asyncio.run(
        node.run(
            _state(
                'instruction one',
                'instruction continuation',
                'instruction two',
                'instruction two continuation',
                'tail',
            )
        )
    )

    assert [item.member_block_uuids for item in result['instructions']] == [
        ['block-0', 'block-1'],
        ['block-2', 'block-3', 'block-4'],
    ]
    assert start_router.calls == ['instruction one', 'instruction two']
    assert boundary_router.calls == [
        'instruction continuation',
        'instruction two',
        'instruction two continuation',
        'tail',
    ]
