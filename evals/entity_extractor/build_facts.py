r"""
Build the fixed fact corpus the entity eval runs against.

The entity pass reads ``atomic_facts``, so measuring it needs facts — and
facts that stay put. This script produces them once from the 38 real pages
committed under ``data/gold/corrector`` and writes them to
``data/eval/entity_extractor/facts.json``, so every later entity run is
measured against identical input and a change in the numbers can only be the
entity prompt.

The chain picks up exactly where the corrector leaves off:

    corrected.md -> formatter -> extractor -> flatten -> atomic facts

``pipeline.build_graph`` runs corrector -> formatter -> extractor, and a
corrector record's ``corrected.md`` IS that first stage's output, hand-checked
against the page image. Starting there is therefore production-exact rather
than an approximation, and it needs no page images and no vision model.

The corrector set is used in preference to ``data/gold/extractor`` because of
what each contains. The extractor set is weighted toward front matter — title
pages, copyright, contents — since apparatus is what it exists to measure; run
through this chain it yields zero atomic facts, correctly, and so cannot
exercise the entity pass at all. The corrector set is 38 pages of body prose,
exercises and proofs across 12 books.

Only ``kind == 'real'`` records are used. The perturbed records share their
base page's gold verbatim, so including them would weight those pages double.

The formatter is kept, and keeping it matters for this particular eval: it is
the stage that wraps bare mathematics in ``$…$`` and rewrites Unicode notation
as LaTeX, so without it the corpus would carry ``x⁴`` and the entity pass
would be charged with a LaTeX violation made upstream of it.

Node ids restart at 0 per page rather than running global across the corpus.
Nothing in the entity pass reads them — facts carry them for provenance only —
and per-page ids keep each record independently regenerable.

Usage:
    uv run python evals/entity_extractor/build_facts.py [--limit N] [--force]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))

from kms.core import llm, models  # noqa: E402
from kms.ingestion import atomic_fact_extractor, extractor, formatter  # noqa: E402

GOLD = REPO / 'data' / 'gold' / 'corrector'
OUT = REPO / 'data' / 'eval' / 'entity_extractor' / 'facts.json'

logger = logging.getLogger('build_facts')


async def _build_one(
    record: dict,
    format_module: formatter.Formatter,
    extract_node: extractor.ExtractorNode,
    fact_module: atomic_fact_extractor.AtomicFactExtractor,
    gate: asyncio.Semaphore,
) -> dict:
    """Run one gold page through to atomic facts.

    Args:
        record: The gold corrector record.
        format_module: The formatting pass.
        extract_node: The extractor's LangGraph node (used for its worker, so
            furniture is dropped exactly as production drops it).
        fact_module: The atomic fact pass.
        gate: Bounds the page-level LM calls in flight.

    Returns:
        The corpus entry: the record's identity, its nodes, and its facts.
    """
    markdown = (GOLD / record['corrected']).read_text()

    async with gate:
        formatted = await format_module.aforward(markdown=markdown)

    segment = models.Segment(index=0, image_path='', content=formatted)
    async with gate:
        result = await extract_node.worker({'segment': segment})
    segment.nodes = result['extract_results'][0][1]
    nodes = models.flatten_segments([segment])

    facts = await atomic_fact_extractor.extract_atomic_facts(
        nodes, module=fact_module
    )

    logger.info(
        '%s: %d chars -> %d node(s) -> %d fact(s)',
        record['id'],
        len(markdown),
        len(nodes),
        len(facts),
    )
    return {
        'id': record['id'],
        'book': record['book'],
        'page': record['page'],
        'split': record['split'],
        'nodes': [
            {'id': node.id, 'type': node.type, 'content': node.content}
            for node in nodes
        ],
        'facts': [
            {'text': fact.text, 'node_ids': fact.node_ids} for fact in facts
        ],
    }


async def main() -> int:
    """Build the corpus.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--limit', type=int, default=0, help='only the first N gold pages'
    )
    parser.add_argument(
        '--split',
        choices=('train', 'dev'),
        help='only records from one split of the corrector set',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='rebuild even though the corpus already exists',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s %(message)s'
    )

    if OUT.exists() and not args.force:
        print(f'{OUT.relative_to(REPO)} already exists — pass --force to rebuild.')
        return 0

    index = json.loads((GOLD / 'index.json').read_text())
    # Real pages only: a perturbed record's gold is a verbatim copy of the
    # page it derives from, so including both would weight that page double.
    records = [record for record in index['records'] if record['kind'] == 'real']
    if args.split:
        records = [
            record for record in records if record['split'] == args.split
        ]
    if args.limit:
        records = records[: args.limit]

    language_model = llm.text_lm()
    format_module = formatter.Formatter(language_model=language_model)
    extract_node = extractor.ExtractorNode(
        module=extractor.Extractor(language_model=language_model)
    )
    fact_module = atomic_fact_extractor.AtomicFactExtractor(
        language_model=language_model
    )
    gate = llm.gate()

    started = time.time()
    entries = await asyncio.gather(
        *(
            _build_one(record, format_module, extract_node, fact_module, gate)
            for record in records
        )
    )
    elapsed = time.time() - started

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                'built_from': 'data/gold/corrector (kind=real)',
                'chain': 'corrected.md -> formatter -> extractor -> atomic_fact_extractor',
                'model': language_model.model,
                'seconds': round(elapsed, 1),
                'records': entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + '\n'
    )

    print(
        f'{len(entries)} page(s), '
        f'{sum(len(entry["facts"]) for entry in entries)} fact(s) '
        f'-> {OUT.relative_to(REPO)} in {elapsed:.0f}s'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
