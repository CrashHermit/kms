import asyncio

from kms import runtime


def _elide(text: str | None, limit: int = 90) -> str:
    text = (text or '').replace('\n', ' ')
    return text if len(text) <= limit else text[: limit - 3] + '...'


async def main():
    result = await runtime.ingest(
        'tests/fixtures/books/calc3_gradients_exercises.pdf',
        output_dir='output/live_instr_test',
        source='calc3_gradients',
    )

    nodes = result.get('nodes') or []
    by_id = {node.id: node for node in nodes if node.id is not None}

    print(f'\n=== nodes: {len(nodes)} ===')
    from collections import Counter

    print('by type:', dict(Counter(node.type for node in nodes)))

    images = [node for node in nodes if node.type == 'image']
    for node in images:
        print(f'  image node id={node.id} path={node.image_path}')

    instructions = result.get('instructions') or []
    print(f'\n=== instructions: {len(instructions)} ===')
    for instruction in instructions:
        print(
            f'  instruction block={instruction.block} '
            f'member_positions={instruction.member_positions}'
        )
        for member_id in instruction.member_positions:
            node = by_id.get(member_id)
            if node:
                print(f'    [{member_id}] ({node.type}) {_elide(node.content)}')

    statements = result.get('statements') or []
    print(f'\n=== statements: {len(statements)} ===')
    for statement in statements:
        kinds = [
            f'{member_id}:'
            f'{by_id.get(member_id).type if by_id.get(member_id) else "?"}'
            for member_id in statement.member_positions
        ]
        print(f'  statement members=[{", ".join(kinds)}]')

    procedures = result.get('procedures') or []
    print(f'\n=== procedures: {len(procedures)} ===')

    spans = result.get('spans') or []
    print(f'\n=== spans (pedagogical): {len(spans)} ===')


if __name__ == '__main__':
    asyncio.run(main())
