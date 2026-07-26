r"""
Trace capture — records every DSPy call's inputs and outputs as JSONL, for prompt
optimisation and offline inspection.

WHY IT EXISTS. The stage logs (``core.logs``) are for reading; this is for *data*. Prompt
tuning without captured I/O is guesswork — the 2026-07-26 session tuned three Signatures by
hand across six full book sweeps and hit visible whack-a-mole (fixing one book's cut broke
another's), which is the classic symptom of tuning with no held-out set. A DSPy optimiser
needs ``dspy.Example``\ s, and an example is exactly one call's inputs paired with its
outputs.

STREAMLINED: NOTHING IS INSTRUMENTED. The retired ``tracing.record()`` scheme needed an
explicit call inside every module, so a new stage silently produced no traces until someone
remembered to wire it up. This hooks DSPy's own callback system instead
(``dspy.settings.callbacks``), so **every stage is captured automatically and no stage module
imports or mentions tracing at all**. A stage added tomorrow is traced the day it is written.

Only ``dspy.Predict`` instances are recorded. A ``ChainOfThought`` fires a callback too and
wraps exactly one inner ``Predict``, so recording both would double every line; the inner
predictor is the one that carries the real signature, and it is the one kept.

USAGE. Set ``KMS_TRACE_DIR`` and run the pipeline — ``run()`` enables capture when the
variable is set, so both the CLI and library callers get it:

    KMS_TRACE_DIR=traces/stein PYTHONPATH=src uv run --extra mistral \
        python -m kms.cli book.pdf out/

Output is one file per stage, ``<dir>/<stage>.jsonl``, each line::

    {"stage": "group_finder", "inputs": {...}, "outputs": {...}}

``outputs`` includes the ``reasoning`` field for ``ChainOfThought`` stages, which is signal
worth keeping. Page images are recorded as a ``'<image>'`` placeholder rather than their
bytes: the image is large and reconstructable from the input PDF, and the trainable text
signal (transcription -> corrected) is captured in full.
"""

import contextlib
import json
import os
import sys
import threading
from pathlib import Path

# DSPy's callback base. Guarded like the optional imports in ``core.llm`` — the test suite
# may run against a stub that has no ``dspy.utils`` package, and tracing is a convenience
# that must never make the package unimportable.
try:
    from dspy.utils.callback import BaseCallback
except ImportError:  # pragma: no cover - only when dspy is stubbed/absent
    BaseCallback = object

TRACE_DIR_ENV = 'KMS_TRACE_DIR'

_recorder: '_Recorder | None' = None


def _plain(value: object) -> object:
    """Convert one traced value into something ``json.dumps`` accepts.

    Pydantic boundary models (the window/member DTOs every stage passes) become plain
    dicts, images become a placeholder, and anything else unrecognised falls back to its
    string form so a trace line is never lost to a serialisation error.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if type(value).__name__ == 'Image':  # dspy.Image — bytes are not trace data
        return '<image>'
    if hasattr(value, 'model_dump'):  # pydantic BaseModel
        with contextlib.suppress(Exception):
            return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return str(value)


def _signature_of(instance: object) -> object | None:
    """The signature a ``dspy.Predict`` was built from, or None for anything else.

    ``Predict`` exposes ``.signature`` directly; ``ChainOfThought`` deliberately does not
    (it holds an inner ``Predict``), which is what makes this the filter that keeps one
    record per logical call.
    """
    return getattr(instance, 'signature', None)


def _stage_of(signature: object) -> str:
    """The stage name a signature belongs to — the last part of its defining module.

    A plain ``Predict`` keeps the original signature class, so its ``__module__`` answers
    directly. ``ChainOfThought`` rebuilds the signature to add its ``reasoning`` field, and
    the rebuilt class reports dspy's own module — but it copies the instructions verbatim,
    so the docstring identifies the origin. Falls back to ``'unknown'`` rather than raising.
    """
    module = getattr(signature, '__module__', '') or ''
    if module.startswith('kms.'):
        return module.rsplit('.', 1)[-1]
    instructions = (getattr(signature, 'instructions', '') or '').strip()
    return _by_instructions().get(instructions, 'unknown')


def _by_instructions() -> dict[str, str]:
    """Map every loaded ``kms.*`` signature's instructions to its stage name.

    Built by scanning ``sys.modules`` rather than importing the stages: ``core`` depends on
    no stage (``docs/ARCHITECTURE.md``, backward-only rule), and by the time a DSPy call
    happens the stage that made it is necessarily imported.
    """
    registry: dict[str, str] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith('kms.') or module is None:
            continue
        for attr in list(vars(module).values()):
            if not isinstance(attr, type):
                continue
            if getattr(attr, '__module__', None) != name:
                continue  # imported into this module, defined in another
            with contextlib.suppress(Exception):
                if not hasattr(attr, 'input_fields'):
                    continue
                key = (attr.instructions or '').strip()
                if key:
                    registry[key] = name.rsplit('.', 1)[-1]
    return registry


class _Recorder(BaseCallback):
    """Writes one JSONL line per DSPy predictor call, grouped into a file per stage."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._pending: dict[str, tuple[str, dict]] = {}
        self._lock = threading.Lock()
        self.written = 0

    def on_module_start(
        self, call_id: str, instance: object, inputs: dict
    ) -> None:
        """Hold a predictor call's inputs until its outputs arrive."""
        signature = _signature_of(instance)
        if signature is None:
            return  # a ChainOfThought wrapper; its inner Predict is the one recorded
        # DSPy hands the call through as {'args': (...), 'kwargs': {...}}; every stage in
        # this package calls its predictor with keyword arguments only.
        named = inputs.get('kwargs') if isinstance(inputs, dict) else None
        with self._lock:
            self._pending[call_id] = (
                _stage_of(signature),
                named if isinstance(named, dict) else inputs,
            )

    def on_module_end(
        self,
        call_id: str,
        outputs: object | None,
        exception: Exception | None = None,
    ) -> None:
        """Pair the outputs with their held inputs and append the trace line."""
        with self._lock:
            held = self._pending.pop(call_id, None)
        if held is None or exception is not None or outputs is None:
            return
        stage, inputs = held
        keys = getattr(outputs, 'keys', None)
        record = {
            'stage': stage,
            'inputs': {
                str(key): _plain(value) for key, value in inputs.items()
            },
            'outputs': {key: _plain(outputs[key]) for key in keys()}
            if callable(keys)
            else _plain(outputs),
        }
        self._append(stage, record)

    def _append(self, stage: str, record: dict) -> None:
        """Append one record to the stage's JSONL file, creating it if needed."""
        with contextlib.suppress(Exception):  # tracing must never break a run
            self.directory.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False)
            with self._lock:
                with open(
                    self.directory / f'{stage}.jsonl', 'a', encoding='utf-8'
                ) as handle:
                    handle.write(line + '\n')
                self.written += 1


def enable(directory: str | Path) -> None:
    """Start capturing every DSPy call into ``directory`` as one JSONL file per stage.

    Idempotent: calling it twice keeps a single recorder, so a library caller that enables
    tracing and then runs several books does not double every line.

    Args:
        directory: Where the per-stage JSONL files are written. Created on first write.
    """
    global _recorder
    if _recorder is not None:
        return
    if BaseCallback is object:  # dspy stubbed or absent — nothing to hook
        return
    import dspy

    _recorder = _Recorder(Path(directory))
    dspy.settings.configure(
        callbacks=[*(dspy.settings.callbacks or []), _recorder]
    )


def enable_from_env() -> None:
    """Enable tracing if ``KMS_TRACE_DIR`` is set; otherwise do nothing.

    This is the pipeline's hook, so capture is opt-in per run with no code change and no
    argument threading."""
    directory = (os.environ.get(TRACE_DIR_ENV) or '').strip()
    if directory:
        enable(directory)


def written() -> int:
    """How many trace lines the active recorder has written (0 when disabled)."""
    return _recorder.written if _recorder else 0
