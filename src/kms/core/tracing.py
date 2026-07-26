r"""
Trace capture — records every DSPy call through MLflow, for prompt optimisation, offline
inspection, and run-to-run comparison.

WHY MLFLOW. The stage logs (``core.logs``) are for reading; this is for *data*. Prompt tuning
without captured I/O is guesswork — the 2026-07-26 session tuned three Signatures by hand
across six full book sweeps and hit visible whack-a-mole (fixing one book's cut broke
another's), which is the classic symptom of tuning with no held-out set. A DSPy optimiser
needs ``dspy.Example``\ s, and an example is exactly one call's inputs paired with its outputs.

This replaced a hand-rolled JSONL recorder. MLflow is DSPy's officially documented integration
and, unlike the OpenTelemetry-based platforms, it hooks DSPy's *callback* system — so pydantic
boundary DTOs arrive as structured dicts rather than ``repr`` strings and survive the round
trip back into a ``dspy.Example`` losslessly. It needs no server (SQLite is enough), no
account, and it tracks optimiser runs, which is what the traces are collected *for*.
``docs/TRACING-RESEARCH.md`` has the measurements behind that choice.

STILL NOTHING IS INSTRUMENTED. ``mlflow.dspy.autolog()`` plus the one hook below are global,
so **every stage is captured automatically and no stage module imports or mentions tracing at
all**. A stage added tomorrow is traced the day it is written.

STAGE IDENTITY IS FREE, AND IT IS A NAMING CONTRACT. MLflow names each trace's root span
after the ``dspy.Module`` subclass that was called — ``GroupFinder.forward`` — so the stage
*is* the class name, and nothing here has to track it. That only works because every stage's
module class is named for its stage: they were uniformly called ``Module`` until this was
adopted, which made every root span read ``Module.forward``. ``core.datasets`` maps the name
back to a stage name, and ``tests/test_datasets.py`` pins the convention so a future stage
cannot quietly break it. An earlier revision tagged spans from a DSPy callback instead; that
is deleted — naming the classes properly does the same job with no code.

ONE THING MLFLOW DOES NOT SOLVE. Autolog embeds a page image's whole base64 payload (~1.2 MB
per corrected page), and MLflow ships no built-in redactors — a custom span processor is the
documented way to mask trace data. ``_strip_images`` is that processor: it swaps the payload
for ``'<image>'``. The bytes are reconstructable from the input PDF, and the trainable text
signal (transcription -> corrected) is kept in full.

USAGE. Set ``KMS_TRACE_DIR`` and run the pipeline — ``run()`` enables capture when the variable
is set, so both the CLI and library callers get it:

    KMS_TRACE_DIR=traces/stein PYTHONPATH=src uv run --extra mlflow --extra mistral \
        python -m kms.cli book.pdf out/

Traces land in ``<dir>/mlruns.db`` under an experiment named after the directory. To read
them, either point the UI at the same store::

    mlflow ui --backend-store-uri sqlite:///traces/stein/mlruns.db

or load them straight into ``dspy.Example``\ s with ``core.datasets.examples_by_stage``.
``KMS_MLFLOW_URI`` overrides the store outright (any MLflow tracking URI, including a remote
server), in which case ``KMS_TRACE_DIR`` only names the experiment.

Traces export asynchronously, so a short run can exit before its queue drains — ``flush()``
is called at the end of ``run()``. Query a store without flushing first and you get nothing.
"""

import contextlib
import os
from pathlib import Path

TRACE_DIR_ENV = 'KMS_TRACE_DIR'
URI_ENV = 'KMS_MLFLOW_URI'

_enabled = False


def _is_image(value: object) -> bool:
    """Whether a traced input value carries base64 image bytes.

    Matched on content as well as type, because by the time autolog serialises the call a
    ``dspy.Image`` has already become the string form of its chat-content payload. The
    patterns are deliberately narrow — a bare ``'base64,'`` would also match a page of a
    textbook *about* base64, and silently destroy that training example.
    """
    if type(value).__name__ == 'Image':  # dspy.Image, before serialisation
        return True
    text = str(value)
    if 'CUSTOM-TYPE-START' in text and 'image_url' in text:
        return True
    return 'data:image/' in text and ';base64,' in text


def _strip_images(span: object) -> None:
    """Replace base64 image payloads in a span's inputs with a ``'<image>'`` placeholder.

    Registered as an MLflow span processor, so it runs on every span with no stage-side
    involvement. Only inputs are rewritten: no stage returns an image.
    """
    with contextlib.suppress(Exception):  # tracing must never break a run
        inputs = span.inputs
        if not isinstance(inputs, dict):
            return
        cleaned = {
            key: '<image>' if _is_image(value) else value
            for key, value in inputs.items()
        }
        if cleaned != inputs:
            span.set_inputs(cleaned)


def store_uri(directory: str | Path) -> str:
    """The MLflow tracking URI backing ``directory`` — a SQLite file inside it.

    SQLite rather than the default ``./mlruns`` file store because MLflow's tracing tables
    need a database backend; this keeps a book's traces self-contained in one directory.
    """
    path = Path(directory).resolve()
    return f'sqlite:///{path / "mlruns.db"}'


def enable(directory: str | Path) -> None:
    """Start capturing every DSPy call into ``directory``'s MLflow store.

    Idempotent: calling it twice keeps a single set of hooks, so a library caller that
    enables tracing and then runs several books does not double every span.

    Args:
        directory: Where the trace store lives, and the experiment name. Created on first
            write. Overridden as a *store* by ``KMS_MLFLOW_URI``, which still takes the
            experiment name from here.
    """
    global _enabled
    if _enabled:
        return
    try:
        import mlflow
    except ImportError:  # pragma: no cover - the mlflow extra is optional
        return

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(
        (os.environ.get(URI_ENV) or '').strip() or store_uri(directory)
    )
    mlflow.set_experiment(directory.name or 'kms')
    mlflow.tracing.configure(span_processors=[_strip_images])
    mlflow.dspy.autolog()
    _enabled = True


def enable_from_env() -> None:
    """Enable tracing if ``KMS_TRACE_DIR`` is set; otherwise do nothing.

    This is the pipeline's hook, so capture is opt-in per run with no code change and no
    argument threading."""
    directory = (os.environ.get(TRACE_DIR_ENV) or '').strip()
    if directory:
        enable(directory)


def flush() -> None:
    """Drain MLflow's async export queue so a just-finished run is queryable.

    A no-op when capture is off. Without this a short run can exit — or a caller can query
    the store — before the queue has drained, and the traces silently appear to be missing.
    """
    if not _enabled:
        return
    with contextlib.suppress(Exception):
        import mlflow

        mlflow.flush_trace_async_logging(terminate=True)
