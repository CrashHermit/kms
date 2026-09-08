import asyncio
import os

os.environ['KMS_MODELS__MODULES__FORMATTER__BASE_URL'] = (
    'http://localhost:8080/v1'
)
os.environ['KMS_MODELS__MODULES__FORMATTER__MODEL'] = 'openai/gemma-4-e4b-it'
os.environ['KMS_MODELS__MODULES__FORMATTER__API_KEY'] = 'not-needed'

from tests.helpers import (
    clear_neo4j,
    print_hubs,
    print_node_summary,
    print_triplet_summary,
)

from kms import runtime

PDF = 'tests/fixtures/books/logic_hammack_truthtables.pdf'
SOURCE = 'logic_hammack_truthtables'


async def main():
    await clear_neo4j()

    result = await runtime.ingest(
        PDF,
        output_dir='output/live_logic_hammack',
        source=SOURCE,
        pages=[0, 1, 2, 3],
    )

    nodes = result.get('nodes') or []
    print_node_summary(nodes)

    triplets = result.get('triplets') or []
    print_triplet_summary(triplets)

    statements = result.get('statements') or []
    print(f'\n=== statements: {len(statements)} ===')

    await print_hubs()

    print('\nDone.')


if __name__ == '__main__':
    asyncio.run(main())
