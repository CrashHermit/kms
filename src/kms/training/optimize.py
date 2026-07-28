r"""Run MIPROv2 over one or more stages' captured traces.

Load every stage's examples from a trace directory (see ``core.tracing`` and
``core.datasets``), split train/holdout, compile the stage with ``MIPROv2``
using its corresponding discrete-error judge metric (from
``training.metrics``), save the compiled result, and print before/after scores.

MIPROv2 does instruction optimisation — it generates candidate prompts and
selects the one that scores highest on the held-out set. This is the right
optimiser for every stage: the corrector has too-large outputs for few-shot
demos to help, and the classification/extraction stages benefit from
instruction tuning more than demo selection.

Usage::

    PYTHONPATH=src uv run python -m kms.training.optimize \
        traces/stein-train --output optimized/

This optimises every stage that has examples in the trace directory. Pass
``--stage corrector`` to target a single stage, or
``--stage corrector extractor`` for a few.
"""

import argparse
import random
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import MIPROv2

from kms.core import datasets, llm
from kms.training import metrics

# Stage name -> (module path, class name) for dynamic import.
_STAGE_CLASS: dict[str, tuple[str, str]] = {
    'corrector': ('kms.ingestion.corrector', 'Corrector'),
    'extractor': ('kms.ingestion.extractor', 'Extractor'),
    'seam_merger': ('kms.ingestion.seam_merger', 'SeamMerger'),
    'splitter': ('kms.ingestion.splitter', 'Splitter'),
    'instruction_finder': (
        'kms.ingestion.instruction_finder',
        'InstructionFinder',
    ),
    'instruction_distributor': (
        'kms.ingestion.instruction_distributor',
        'InstructionDistributor',
    ),
    'pedagogical_component_finder': (
        'kms.ingestion.pedagogical_component_finder',
        'PedagogicalComponentFinder',
    ),
    'role_typer': ('kms.ingestion.role_typer', 'RoleTyper'),
}

_HOLDOUT_FRACTION = 0.3
_RANDOM_SEED = 42
_NUM_CANDIDATES = 10

# Stages whose outputs are too large for few-shot demos to help. For these
# MIPROv2 does pure instruction optimisation (no bootstrapped demos).
_NO_DEMO_STAGES = frozenset({'corrector'})


def _import_stage(stage: str) -> type:
    """Import and return a stage's ``dspy.Module`` subclass."""
    import importlib

    module_path, class_name = _STAGE_CLASS[stage]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not issubclass(cls, dspy.Module):
        raise TypeError(
            f'{stage}: {module_path}.{class_name} is not a dspy.Module'
        )
    return cls


def _split(
    examples: list[dspy.Example],
    holdout: float = _HOLDOUT_FRACTION,
    seed: int = _RANDOM_SEED,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Shuffle and split examples into train and holdout sets."""
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    boundary = max(1, int(len(shuffled) * (1 - holdout)))
    return shuffled[:boundary], shuffled[boundary:]


def _evaluate(
    program: dspy.Module,
    holdout: list[dspy.Example],
    metric: Any,
) -> float:
    """Return the mean metric score over *holdout*."""
    evaluator = dspy.Evaluate(
        devset=holdout,
        metric=metric,
        num_threads=1,
        display_progress=False,
        display_table=0,
        max_errors=5,
    )
    result = evaluator(program)
    return float(result.score)


def optimize_stage(
    stage: str,
    examples: list[dspy.Example],
    output_dir: str | Path,
) -> dict[str, float]:
    """Compile *stage* with MIPROv2, save, and return scores.

    Returns ``{'baseline': float, 'optimized': float}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module_cls = _import_stage(stage)
    metric = metrics.STAGE_METRICS[stage]
    trainset, holdout = _split(examples)

    baseline = _evaluate(module_cls(), holdout, metric)
    print(f'  {stage}: {len(trainset)} train / {len(holdout)} holdout')

    use_demos = stage not in _NO_DEMO_STAGES
    optimizer = MIPROv2(
        metric=metric,
        prompt_model=llm.prompt_optimizer_lm(),
        num_candidates=_NUM_CANDIDATES,
        max_bootstrapped_demos=4 if use_demos else 0,
        max_labeled_demos=4 if use_demos else 0,
        auto='light',
        seed=_RANDOM_SEED,
        verbose=False,
    )
    compiled = optimizer.compile(
        student=module_cls(),
        trainset=trainset,
        valset=holdout,
    )

    optimized = _evaluate(compiled, holdout, metric)
    delta = optimized - baseline

    path = output_dir / f'{stage}.json'
    compiled.save(str(path))
    print(
        f'  {stage}: {baseline:.2f} -> {optimized:.2f}  '
        f'(Delta {"+ " if delta >= 0 else ""}{delta:.2f})  '
        f'saved {path}'
    )

    return {'baseline': baseline, 'optimized': optimized}


def optimize_all(
    trace_dir: str,
    output_dir: str | Path = 'optimized',
    stages: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Optimise every stage that has examples in *trace_dir*.

    Args:
        trace_dir: The trace directory passed to ``tracing.enable`` (or
            ``KMS_TRACE_DIR``).
        output_dir: Where to write the compiled JSON files.
        stages: Which stages to optimise. ``None`` means every stage that
            has captured examples.
    """
    by_stage = datasets.examples_by_stage(trace_dir)
    candidates = stages or list(by_stage)

    results: dict[str, dict[str, float]] = {}
    total_examples = sum(len(v) for v in by_stage.values())
    print(
        f'{total_examples} example(s) across {len(by_stage)} stage(s) '
        f'in {trace_dir}'
    )

    for stage in candidates:
        examples = by_stage.get(stage, [])
        if not examples:
            print(f'  {stage}: no examples — skipping')
            continue
        results[stage] = optimize_stage(stage, examples, output_dir)

    if results:
        print()
        print('Summary:')
        for stage, scores in results.items():
            delta = scores['optimized'] - scores['baseline']
            print(
                f'  {stage:35s}  '
                f'{scores["baseline"]:.2f} -> '
                f'{scores["optimized"]:.2f}  '
                f'(Delta {"+ " if delta >= 0 else ""}{delta:.2f})'
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Optimise one or more DSPy stages from captured traces.',
    )
    parser.add_argument(
        'trace_dir',
        help='Trace directory (as passed to KMS_TRACE_DIR)',
    )
    parser.add_argument(
        '--output',
        default='optimized',
        help='Output directory for compiled programs (default: optimized/)',
    )
    parser.add_argument(
        '--stage',
        nargs='*',
        default=None,
        help='Stage(s) to optimise (default: all that have examples)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List which stages have examples without optimising',
    )
    args = parser.parse_args()

    if args.list:
        by_stage = datasets.examples_by_stage(args.trace_dir)
        if not by_stage:
            print(f'No examples in {args.trace_dir}')
            return
        for stage, examples in sorted(by_stage.items()):
            print(f'{stage:35s}  {len(examples)} example(s)')
        return

    optimize_all(args.trace_dir, args.output, args.stage)


if __name__ == '__main__':
    _main()
