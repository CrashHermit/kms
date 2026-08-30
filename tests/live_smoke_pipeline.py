import asyncio
import logging
import os
import sys
import time
from collections import Counter

import dspy

os.environ['KMS_RECORDING__ENABLED'] = 'true'

# Set up detailed logging to see all pipeline stages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
    stream=sys.stdout,
)
# Keep third-party loggers at INFO to reduce noise
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('neo4j').setLevel(logging.WARNING)
logging.getLogger('aiohttp').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('litellm').setLevel(logging.WARNING)

from kms import runtime  # noqa: E402
from kms.core import loading  # noqa: E402
from kms.graph import db, schema  # noqa: E402

PDF = 'books/calc3_gradients_exercises.pdf'
SOURCE = 'calc3_gradients_exercises'
OUTPUT_DIR = f'output/live_smoke/run-{time.time_ns()}'
CACHE_DIR = f'output/live_smoke/cache-{time.time_ns()}'
dspy.configure_cache(disk_cache_dir=CACHE_DIR)


def elapsed(start):
    return time.perf_counter() - start


def _require_destructive_test_guard() -> None:
    """Requires explicit authorization before clearing the integration DB."""
    if os.getenv('KMS_NEO4J_IT') != '1':
        raise RuntimeError(
            'live smoke requires KMS_NEO4J_IT=1; refusing to clear Neo4j'
        )
    if os.getenv('KMS_LIVE_SMOKE_CLEAR') != 'I_UNDERSTAND_THIS_WIPES_NEO4J':
        raise RuntimeError(
            'set KMS_LIVE_SMOKE_CLEAR=I_UNDERSTAND_THIS_WIPES_NEO4J '
            'to authorize the destructive database reset'
        )


def _assert_recording_value(value: object) -> None:
    """Rejects legacy recording values while allowing text-only records."""
    if isinstance(value, list):
        if any(type(item).__name__ == 'WindowNode' for item in value):
            raise AssertionError('recording contains legacy WindowNode inputs')
        for item in value:
            _assert_recording_value(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_recording_value(item)


def verify_recordings(output_dir: str) -> list[loading.Dataset]:
    """Loads and validates replay datasets independently of ingestion."""
    datasets = loading.load_datasets(output_dir)
    if not datasets:
        raise AssertionError('live smoke produced no replay datasets')
    for dataset in datasets:
        if not dataset.examples:
            raise AssertionError(
                f'{dataset.stage} produced an empty replay dataset'
            )
        for example in dataset.examples:
            _assert_recording_value(example.toDict())
    return datasets


async def main():
    start_total = time.perf_counter()
    print(f'[{elapsed(start_total):.1f}s] Starting pipeline...', flush=True)

    if not db.is_configured():
        print('Neo4j not configured.', flush=True)
        return

    _require_destructive_test_guard()
    print(f'[{elapsed(start_total):.1f}s] Clearing DB...', flush=True)
    async with db.session() as session:
        result = await session.run('MATCH (n) DETACH DELETE n')
        await result.consume()
    print(f'[{elapsed(start_total):.1f}s] DB cleared, ensuring schema...', flush=True)
    await schema.ensure_schema(lambda: db.session())
    print(f'[{elapsed(start_total):.1f}s] Schema ensured, starting ingestion...', flush=True)

    result = await runtime.ingest(
        PDF,
        output_dir=OUTPUT_DIR,
        source=SOURCE,
        pages=[0, 1],
    )
    print(f'[{elapsed(start_total):.1f}s] Ingestion complete', flush=True)

    nodes = result.get('nodes') or []
    print(f'\n=== nodes: {len(nodes)} ===', flush=True)
    print('by type:', dict(Counter(node.type for node in nodes)), flush=True)

    triplets = result.get('triplets') or []
    print(f'\n=== triplets: {len(triplets)} ===', flush=True)
    for triplet in triplets:
        print(f'  {triplet.subject} | {triplet.predicate} | {triplet.object}', flush=True)

    print('\n=== hub building ===', flush=True)
    print(f'  entity assigned: {result.get("entity_assigned")}', flush=True)
    print(f'  predicate assigned: {result.get("predicate_assigned")}', flush=True)
    print(f'  procedures created: {result.get("procedures_created")}', flush=True)

    datasets = verify_recordings(f'{OUTPUT_DIR}/examples')
    print(f'\n=== recorded datasets: {len(datasets)} ===', flush=True)
    for dataset in datasets:
        print(
            f'  {dataset.stage}: {len(dataset.examples)} example(s), '
            f'signature={dataset.signature.__name__}', flush=True
        )

    print('\nDone.', flush=True)


if __name__ == '__main__':
    asyncio.run(main())