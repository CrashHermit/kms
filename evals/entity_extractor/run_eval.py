r"""
Run the entity pass over the fixed fact corpus and score it against its own
stated rules.

Reads ``data/eval/entity_extractor/facts.json`` (built once by
``build_facts.py``), runs ``kms.ingestion.entity_extractor`` live over every
record, and reports what the run did: how many mentions per fact, how many
facts got nothing, and every finding from ``checks.py``.

The batching here deliberately mirrors ``entity_extractor.extract_entities``
rather than calling it — same ``_batch_facts`` cut, same ``llm.gate``, same
``aforward`` per batch — because the eval needs to know WHICH batch produced
each mention. That is the only way to catch a ``fact_index`` that is a valid
fact but was never in the batch the model was looking at, which is the
silent-corruption case: the mention lands on a real fact, the wrong one, and
nothing downstream can tell.

Each run is written to ``output/evals/entity_extractor/<timestamp>.json``.
Pass an earlier run to ``--baseline`` to print the deltas — the intended loop
after a prompt edit.

Usage:
    uv run python evals/entity_extractor/run_eval.py [--limit N]
    uv run python evals/entity_extractor/run_eval.py --baseline output/evals/entity_extractor/<file>.json
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import checks  # noqa: E402

from kms.core import llm, models  # noqa: E402
from kms.ingestion import entity_extractor  # noqa: E402

CORPUS = REPO / 'data' / 'eval' / 'entity_extractor' / 'facts.json'
REPORTS = REPO / 'output' / 'evals' / 'entity_extractor'

# Shorthands for --model. The full ids are what `llm.text_lm` reads from
# TEXT_MODEL; these just save typing at the prompt.
_MODEL_ALIASES = {
    'flash': 'deepseek/deepseek-v4-flash',
    'pro': 'deepseek/deepseek-v4-pro',
}

logger = logging.getLogger('run_eval')


async def _run_record(
    entry: dict,
    module: entity_extractor.EntityExtractor,
    gate: asyncio.Semaphore,
) -> dict:
    """Run the entity pass over one record's facts.

    Args:
        entry: The corpus entry.
        module: The entity extractor under test.
        gate: Bounds the LM calls in flight.

    Returns:
        The record's raw result: its batches and the mentions each produced.
    """
    facts = [
        models.AtomicFact(text=fact['text'], node_ids=fact['node_ids'])
        for fact in entry['facts']
    ]
    batches = entity_extractor._batch_facts(facts)

    async def _one(batch: list[tuple[int, str]]) -> list[tuple[int, str]]:
        async with gate:
            return await module.aforward(batch)

    per_batch = await asyncio.gather(*(_one(batch) for batch in batches))

    logger.info(
        '%s: %d fact(s) in %d batch(es) -> %d mention(s)',
        entry['id'],
        len(facts),
        len(batches),
        sum(len(mentions) for mentions in per_batch),
    )
    return {
        'id': entry['id'],
        'facts': [fact.text for fact in facts],
        'batches': [
            {
                'indices': [index for index, _ in batch],
                'mentions': [
                    {'fact_index': index, 'name': name}
                    for index, name in mentions
                ],
            }
            for batch, mentions in zip(batches, per_batch, strict=True)
        ],
    }


def _score(results: list[dict]) -> dict:
    """Apply every check to every mention and aggregate.

    Args:
        results: The raw per-record results.

    Returns:
        The scored report body.
    """
    findings: list[dict] = []
    counts: Counter[str] = Counter()
    severity_of: dict[str, str] = {}
    mentions_per_fact: list[int] = []
    total_mentions = 0
    total_facts = 0
    total_batches = 0
    names: Counter[str] = Counter()

    for result in results:
        facts = result['facts']
        total_facts += len(facts)
        total_batches += len(result['batches'])
        per_fact: defaultdict[int, int] = defaultdict(int)
        seen: defaultdict[int, set[str]] = defaultdict(set)

        for batch in result['batches']:
            batch_indices = set(batch['indices'])
            for mention in batch['mentions']:
                total_mentions += 1
                index = mention['fact_index']
                name = mention['name']
                names[name] += 1
                if 0 <= index < len(facts):
                    per_fact[index] += 1
                    fact_text = facts[index]
                else:
                    fact_text = None

                for check, severity, detail in checks.check_mention(
                    name=name,
                    fact_index=index,
                    fact_text=fact_text,
                    fact_count=len(facts),
                    batch_indices=batch_indices,
                    seen_in_fact=seen[index],
                ):
                    counts[check] += 1
                    severity_of[check] = severity
                    findings.append(
                        {
                            'record': result['id'],
                            'check': check,
                            'severity': severity,
                            'detail': detail,
                            'name': name,
                            'fact_index': index,
                            'fact': fact_text,
                        }
                    )

        mentions_per_fact.extend(per_fact.get(i, 0) for i in range(len(facts)))

    covered = [count for count in mentions_per_fact if count]
    # A triple needs two endpoints drawn from the same fact, so a fact with
    # fewer than two mentions cannot produce one however good the relation
    # pass is. This is the ceiling the mention pass sets on triple recall.
    triple_capable = [count for count in mentions_per_fact if count >= 2]
    histogram = Counter(min(count, 6) for count in mentions_per_fact)
    return {
        'totals': {
            'records': len(results),
            'facts': total_facts,
            'batches': total_batches,
            'mentions': total_mentions,
            'distinct_names': len(names),
        },
        'coverage': {
            'facts_with_mentions': len(covered),
            'facts_without_mentions': total_facts - len(covered),
            'facts_without_mentions_pct': round(
                100 * (total_facts - len(covered)) / total_facts, 1
            )
            if total_facts
            else 0.0,
            'mentions_per_fact_mean': round(
                statistics.mean(mentions_per_fact), 2
            )
            if mentions_per_fact
            else 0.0,
            'mentions_per_fact_max': max(mentions_per_fact, default=0),
            'triple_capable_facts': len(triple_capable),
            'triple_capable_pct': round(
                100 * len(triple_capable) / total_facts, 1
            )
            if total_facts
            else 0.0,
            'mentions_per_fact_histogram': {
                ('6+' if count == 6 else str(count)): facts
                for count, facts in sorted(histogram.items())
            },
            'repeat_name_rate_pct': round(
                100 * (total_mentions - len(names)) / total_mentions, 1
            )
            if total_mentions
            else 0.0,
        },
        'checks': {
            check: {
                'count': count,
                'severity': severity_of[check],
                'pct_of_mentions': round(100 * count / total_mentions, 1)
                if total_mentions
                else 0.0,
            }
            for check, count in counts.most_common()
        },
        'findings': findings,
    }


def _print_report(report: dict, baseline: dict | None) -> None:
    """Print the run as a readable summary.

    Args:
        report: The scored report.
        baseline: An earlier report to diff against, or None.
    """
    totals = report['totals']
    coverage = report['coverage']

    print()
    print('# Entity extractor eval')
    print()
    print(f'model     {report["model"]}')
    print(f'corpus    {totals["records"]} page(s), {totals["facts"]} fact(s)')
    print(
        f'run       {totals["batches"]} batch(es) -> '
        f'{totals["mentions"]} mention(s) in {report["seconds"]}s'
    )
    print()

    print('## Coverage')
    print()
    print(
        f'  mentions per fact      {coverage["mentions_per_fact_mean"]} mean, '
        f'{coverage["mentions_per_fact_max"]} max'
    )
    print(
        f'  facts with no mention  {coverage["facts_without_mentions"]} '
        f'({coverage["facts_without_mentions_pct"]}%)'
    )
    print(
        f'  triple-capable facts   {coverage["triple_capable_facts"]} '
        f'({coverage["triple_capable_pct"]}%) — 2+ mentions, the ceiling on '
        f'triple recall'
    )
    print(
        '  mentions per fact      '
        + ', '.join(
            f'{count}: {facts}'
            for count, facts in coverage['mentions_per_fact_histogram'].items()
        )
    )
    print(
        f'  distinct names         {totals["distinct_names"]} '
        f'({coverage["repeat_name_rate_pct"]}% of mentions repeat a name)'
    )
    print()

    for severity, heading in (
        (checks.VIOLATION, '## Violations — unambiguous breaches'),
        (checks.REVIEW, '## Review — heuristic, read before believing'),
    ):
        rows = [
            (check, data)
            for check, data in report['checks'].items()
            if data['severity'] == severity
        ]
        print(heading)
        print()
        if not rows:
            print('  none')
            print()
            continue
        for check, data in rows:
            delta = ''
            if baseline:
                was = baseline.get('checks', {}).get(check, {}).get('count', 0)
                change = data['count'] - was
                delta = f'  ({change:+d} vs baseline)' if change else '  (=)'
            print(
                f'  {check:<20} {data["count"]:>4}  '
                f'{data["pct_of_mentions"]:>5}% of mentions{delta}'
            )
        print()

    print('## Samples')
    print()
    shown: Counter[str] = Counter()
    for finding in report['findings']:
        if shown[finding['check']] >= 3:
            continue
        shown[finding['check']] += 1
        print(f'  [{finding["check"]}] {finding["name"]!r}')
        print(f'      {finding["detail"]}')
        if finding['fact']:
            fact = finding['fact']
            print(
                f'      fact {finding["fact_index"]}: '
                f'{fact[:110]}{"…" if len(fact) > 110 else ""}'
            )
    print()

    if baseline:
        print('## Baseline')
        print()
        was = baseline['totals']
        print(
            f'  mentions  {was["mentions"]} -> {totals["mentions"]} '
            f'({totals["mentions"] - was["mentions"]:+d})'
        )
        print(
            f'  facts with no mention  '
            f'{baseline["coverage"]["facts_without_mentions"]} -> '
            f'{coverage["facts_without_mentions"]}'
        )
        print()


async def main() -> int:
    """Run the eval.

    Returns:
        A process exit code — non-zero when a hard violation was found.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--limit', type=int, default=0, help='only the first N records'
    )
    parser.add_argument(
        '--baseline', type=Path, help='an earlier report JSON to diff against'
    )
    parser.add_argument(
        '--model',
        help=(
            "the text LM to run the pass on: 'flash', 'pro', or a full "
            'litellm id. Sets TEXT_MODEL, which llm.text_lm() reads.'
        ),
    )
    args = parser.parse_args()

    if args.model:
        os.environ['TEXT_MODEL'] = _MODEL_ALIASES.get(args.model, args.model)

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s %(message)s'
    )

    if not CORPUS.exists():
        print(
            f'No corpus at {CORPUS.relative_to(REPO)} — run '
            'evals/entity_extractor/build_facts.py first.'
        )
        return 2

    corpus = json.loads(CORPUS.read_text())
    entries = corpus['records']
    if args.limit:
        entries = entries[: args.limit]

    language_model = llm.text_lm()
    module = entity_extractor.EntityExtractor(language_model=language_model)
    gate = llm.gate()

    started = time.time()
    results = await asyncio.gather(
        *(_run_record(entry, module, gate) for entry in entries)
    )
    elapsed = time.time() - started

    report = _score(results)
    # The raw run, so a new metric can be computed over an old report instead
    # of paying for the model again.
    report['results'] = results
    report['model'] = language_model.model
    report['seconds'] = round(elapsed, 1)
    report['corpus'] = str(CORPUS.relative_to(REPO))
    report['ran_at'] = datetime.now(UTC).isoformat()

    REPORTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    path = REPORTS / f'{stamp}.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    _print_report(report, baseline)
    print(f'report written to {path.relative_to(REPO)}')

    violations = sum(
        data['count']
        for data in report['checks'].values()
        if data['severity'] == checks.VIOLATION
    )
    return 1 if violations else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
