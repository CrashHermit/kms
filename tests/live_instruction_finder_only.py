"""Run the boolean instruction router and grower against a small fixture."""

import asyncio

from kms import config
from kms.construction import instruction_finder
from kms.core import llm, models


def _nodes() -> list[models.Node]:
    return [
        models.Node(
            index=0,
            type='paragraph',
            content='For the following exercises, find the gradient.',
        ),
        models.Node(
            index=1,
            type='paragraph',
            content='280. Find the gradient of $f(x, y) = x^2 + y^2$.',
        ),
        models.Node(
            index=2,
            type='paragraph',
            content='281. Find the gradient of $f(x, y) = xy$.',
        ),
        models.Node(
            index=3,
            type='paragraph',
            content='For the following exercises, find equations of:',
        ),
        models.Node(
            index=4,
            type='paragraph',
            content='a. the tangent plane and',
        ),
        models.Node(
            index=5,
            type='paragraph',
            content='b. the normal line to the given surface at the given point.',
        ),
        models.Node(
            index=6,
            type='paragraph',
            content='302. $z = 4x^2 + y^2$, point $P(2, 1, 8)$',
        ),
        models.Node(
            index=7,
            type='paragraph',
            content=(
                '282. Find the gradient of $f(x, y, z)$ at $P$ and the '
                'directional derivative in the direction of $\\mathbf{u}$'
            ),
        ),
    ]


async def main() -> None:
    router = instruction_finder.InstructionRouter(
        language_model=llm.module_lm('instruction_router')
    )
    grower = instruction_finder.InstructionGrower(
        language_model=llm.module_lm('instruction_grower')
    )
    spans = await instruction_finder.find_instruction_spans(
        _nodes(), router=router, grower=grower
    )
    print(
        f'context budget: '
        f'{config.get_settings().stages.finders.instruction_finder.context_budget} tokens'
    )
    print(f'\n=== {len(spans)} instruction span(s) ===')
    for span in spans:
        print(f'  {span}')


if __name__ == '__main__':
    asyncio.run(main())