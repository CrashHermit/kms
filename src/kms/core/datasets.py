r"""
Turn captured traces into ``dspy.Example``\ s, grouped by the stage that produced them.

This is the read side of ``core.tracing``: capture writes MLflow spans during a book sweep,
and this loads them back in the one shape a DSPy optimiser consumes. An example *is* one
call's inputs paired with its outputs, so the conversion is mechanical — which is the whole
reason ``tracing`` records through MLflow's callback integration rather than an
OpenTelemetry-based one, where pydantic boundary DTOs arrive flattened to ``repr`` strings
and cannot be parsed back (``docs/TRACING-RESEARCH.md``, finding 1).

Selection is by the ``kms.stage`` attribute that ``tracing._StageTagger`` writes, which lands
on exactly one span per logical call. That is what keeps the fan-out out of the dataset: a
single ``Predict`` call also produces ``ChatAdapter.format``, ``LM.__call__`` and
``ChatAdapter.parse`` spans, and a ``ChainOfThought`` adds an outer wrapper span on top —
none of which are tagged.

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


def _tagged_spans(traces: list) -> list:
    """The one span per logical DSPy call that carries a resolved stage."""
    spans = []
    for trace in traces:
        for span in getattr(trace.data, 'spans', []) or []:
            attributes = span.attributes or {}
            if attributes.get(tracing.STAGE_ATTR):
                spans.append(span)
    return spans


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
    for span in _tagged_spans(_load_traces(source)):
        stage = (span.attributes or {})[tracing.STAGE_ATTR]
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
