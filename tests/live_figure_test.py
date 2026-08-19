import asyncio
from collections import Counter

from kms import runtime


def _elide(text, limit=80):
    text = (text or '').replace('\n', ' ')
    return text if len(text) <= limit else text[: limit - 3] + '...'


async def main():
    result = await runtime.ingest(
        'tests/fixtures/books/ode_lebl_diffyqs.pdf',
        output_dir='output/live_figure_test',
        source='ode_lebl_diffyqs',
        pages=[2],
    )

    nodes = result.get('nodes') or []
    by_id = {node.id: node for node in nodes if node.id is not None}

    print(f'\n=== nodes: {len(nodes)} ===')
    print('by type:', dict(Counter(node.type for node in nodes)))

    images = [node for node in nodes if node.type == 'image']
    print(f'\n=== image nodes: {len(images)} ===')
    for node in images:
        print(f'  image id={node.id} path={node.image_path}')

    statements = result.get('statements') or []
    print(f'\n=== statements: {len(statements)} ===')
    for statement in statements:
        members = []
        for member_id in statement.members:
            node = by_id.get(member_id)
            if node is not None and node.type == 'image':
                members.append(f'[{member_id}] IMAGE')
            elif node is not None:
                members.append(f'[{member_id}] {_elide(node.content, 45)}')
        print(f'  statement: {members}')

    instructions = result.get('instructions') or []
    print(f'\n=== instructions: {len(instructions)} ===')
    for instruction in instructions:
        members = []
        for member_id in instruction.members:
            node = by_id.get(member_id)
            if node is not None:
                members.append(
                    f'[{member_id}] ({node.type}) {_elide(node.content, 45)}'
                )
        print(f'  instruction: {members}')


if __name__ == '__main__':
    asyncio.run(main())
