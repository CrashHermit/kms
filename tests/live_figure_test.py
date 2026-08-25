import asyncio

from tests.helpers import (
    print_instructions,
    print_node_summary,
    print_statement_images,
)

from kms import runtime

PDF = 'tests/fixtures/books/ode_lebl_diffyqs.pdf'
SOURCE = 'ode_lebl_diffyqs'


def _elide(text, limit=80):
    text = (text or '').replace('\n', ' ')
    return text if len(text) <= limit else text[: limit - 3] + '...'


async def main():
    result = await runtime.ingest(
        PDF,
        output_dir='output/live_figure_test',
        source=SOURCE,
        pages=[2],
    )

    nodes = result.get('nodes') or []
    by_id = {node.id: node for node in nodes if node.id is not None}

    print_node_summary(nodes)

    images = [node for node in nodes if node.type == 'image']
    print(f'\n=== image nodes: {len(images)} ===')
    for node in images:
        print(f'  image id={node.id} path={node.image_path}')

    statements = result.get('statements') or []
    print_statement_images(statements, by_id)

    instructions = result.get('instructions') or []
    print_instructions(instructions, by_id)


if __name__ == '__main__':
    asyncio.run(main())