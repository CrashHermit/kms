import asyncio
import os

os.environ['PIPELINE_API_BASE'] = 'http://localhost:8080/v1'
os.environ['PIPELINE_MODEL'] = 'openai/unsloth/gemma-4-e4b-it-GGUF'
os.environ['PIPELINE_API_KEY'] = 'not-needed'

from kms.core import llm
from kms.ingestion import instruction_finder


def _nodes():
    return [
        instruction_finder.WindowNode(
            position=0,
            type='paragraph',
            content='For the following exercises, find the gradient.',
        ),
        instruction_finder.WindowNode(
            position=1,
            type='paragraph',
            content='280. Find the gradient of $f(x, y) = x^2 + y^2$.',
        ),
        instruction_finder.WindowNode(
            position=2,
            type='paragraph',
            content='281. Find the gradient of $f(x, y) = xy$.',
        ),
        instruction_finder.WindowNode(
            position=3,
            type='paragraph',
            content='For the following exercises, find equations of:',
        ),
        instruction_finder.WindowNode(
            position=4,
            type='paragraph',
            content='a. the tangent plane and',
        ),
        instruction_finder.WindowNode(
            position=5,
            type='paragraph',
            content='b. the normal line to the given surface at the given point.',
        ),
        instruction_finder.WindowNode(
            position=6,
            type='paragraph',
            content='302. $z = 4x^2 + y^2$, point $P(2, 1, 8)$',
        ),
        instruction_finder.WindowNode(
            position=7,
            type='paragraph',
            content='282. Find the gradient of $f(x, y, z)$ at $P$ and the directional derivative in the direction of $\\mathbf{u}$',
        ),
    ]


async def main():
    finder = instruction_finder.InstructionFinder(language_model=llm.pipeline_lm())
    nodes = _nodes()
    spans = await finder.aforward(nodes)
    print(f'\n=== {len(spans)} instruction span(s) ===')
    for span in spans:
        members = list(range(span.start, span.end + 1))
        print(f'  span [{span.start}, {span.end}] -> {members}')
        for i in members:
            print(f'    [{i}] {nodes[i].content}')


if __name__ == '__main__':
    asyncio.run(main())
