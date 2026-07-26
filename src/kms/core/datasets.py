r"""
Turn captured traces into ``dspy.Example``\ s, grouped by the stage that produced them.

This is the read side of ``core.tracing``: capture writes MLflow spans during a book sweep,
and this loads them back in the one shape a DSPy optimiser consumes. An example *is* one
call's inputs paired with its outputs, so the conversion is mechanical — which is the whole
reason ``tracing`` records through MLflow's callback integration rather than an
OpenTelemetry-based one, where pydantic boundary DTOs arrive flattened to ``repr`` strings
and cannot be parsed back (``docs/TRACING-RESEARCH.md``, finding 1).

A trace is one stage call, and its spans nest::

    GroupFinder.forward       <- root: the stage, named after its dspy.Module subclass
      ChainOfThought.forward
        Predict.forward       <- the field-keyed inputs/outputs an Example is built from
          ChatAdapter.format
          LM.__call__
          ChatAdapter.parse

So the two questions have two different answers in the same trace: the **root** span names
the stage (``GroupFinder`` -> ``group_finder``), and the innermost ``Predict.forward`` span
carries the signature's fields. Taking exactly one ``Predict`` span per trace is what keeps
the adapter/LM fan-out — and ``ChainOfThought``'s duplicate wrapper — out of the dataset.

This depends on a naming contract: **a stage's ``dspy.Module`` subclass is named for its
stage**. It is free when honoured and silent when not, so ``tests/test_datasets.py`` asserts
every stage class round-trips to its own module name.

    from kms.core import datasets

    by_stage = datasets.examples_by_stage('traces/stein')
    trainset = by_stage['group_finder']

Outputs keep the ``reasoning`` field for ``ChainOfThought`` stages, which is signal worth
training on. Inputs become the example's input keys, so an example is ready for
``dspy.Evaluate`` and the optimisers without further shaping.
"""

from pathlib import Path

from kms.core import tracing


def _load_traces(source: str | Path) -> list:
    """Every trace in ``source``, which is either a trace directory or a tracking URI.

    A directory also names its experiment (``tracing.enable`` sets it that way), so the
    lookup is scoped to it; a bare URI is searched whole.
    """
    import mlflow

    text = str(source)
    if '://' in text:
        mlflow.set_tracking_uri(text)
        return list(mlflow.search_traces(return_type='list'))

    mlflow.set_tracking_uri(tracing.store_uri(source))
    # ``locations`` is keyed by experiment id, not name, so the directory's experiment has
    # to be resolved first. A directory never swept is simply empty, not an error.
    experiment = mlflow.get_experiment_by_name(Path(text).name or 'kms')
    if experiment is None:
        return []
    return list(
        mlflow.search_traces(
            return_type='list', locations=[experiment.experiment_id]
        )
    )


def stage_name(class_name: str) -> str:
    """The stage a dspy.Module subclass belongs to: ``'GroupFinder'`` -> ``'group_finder'``.

    The inverse of the naming contract, and the only place it is encoded.
    """
    out = []
    for index, char in enumerate(class_name):
        if char.isupper() and index:
            out.append('_')
        out.append(char.lower())
    return ''.join(out)


def _calls(traces: list) -> list[tuple[str, object]]:
    """One ``(stage, span)`` per traced stage call.

    The stage comes from the trace's root span — named after the ``dspy.Module`` subclass —
    and the span is the innermost ``Predict.forward``, the only one whose inputs and outputs
    are keyed by the signature's fields. A trace missing either is skipped rather than
    guessed at: a bare predictor call made outside a stage has no stage to name.
    """
    calls = []
    for trace in traces:
        spans = list(getattr(trace.data, 'spans', []) or [])
        root = next((s for s in spans if s.parent_id is None), None)
        predicts = [s for s in spans if s.name == 'Predict.forward']
        if root is None or len(predicts) != 1:
            continue
        stage = stage_name(root.name.split('.', 1)[0])
        calls.append((stage, predicts[0]))
    return calls


def examples_by_stage(
    source: str | Path,
) -> dict[str, list]:
    """Load captured traces and group them into ``dspy.Example``\\ s per stage.

    Args:
        source: A trace directory (as passed to ``tracing.enable`` / ``KMS_TRACE_DIR``) or
            an MLflow tracking URI.

    Returns:
        Stage name -> the examples that stage produced, each with the call's inputs marked
        as input keys. Stages that made no calls are absent rather than empty.
    """
    import dspy

    by_stage: dict[str, list] = {}
    for stage, span in _calls(_load_traces(source)):
        inputs = span.inputs if isinstance(span.inputs, dict) else {}
        outputs = span.outputs if isinstance(span.outputs, dict) else {}
        if not inputs or not outputs:
            continue  # a failed or half-recorded call trains nothing
        example = dspy.Example(**inputs, **outputs).with_inputs(*inputs)
        by_stage.setdefault(stage, []).append(example)
    return by_stage


def stage_counts(source: str | Path) -> dict[str, int]:
    """How many examples each stage contributed — a quick check that a sweep captured."""
    return {
        stage: len(examples)
        for stage, examples in sorted(examples_by_stage(source).items())
    }
