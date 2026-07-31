"""Record DSPy module calls as JSONL examples for training.

Each pipeline run creates ``<output_dir>/<module>/<run_id>/`` containing:

* ``examples.jsonl`` — one JSON object per call (append-only, crash-safe)
* ``meta.json`` — run context (book, pages, model)
* ``images/`` — sidecar PNGs for ``dspy.Image`` fields

Run IDs are deterministic ``uuid5`` hashes of the source name, so
re-running the same book appends to the same corpus.
"""

import base64
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import dspy
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_NAMESPACE_KMS = uuid5(NAMESPACE_URL, 'kms')

_run_id: str | None = None
_run_meta: dict = {}
_output_dir: str = 'output/examples'


def set_run(source: str, output_dir: str = 'output/examples', **meta) -> None:
    """Set the current run identity, derived from the source name.

    Args:
        source: The book identity (PDF filename or Neo4j key).
        output_dir: Root directory for recorded examples.
        **meta: Extra metadata written to ``meta.json``.
    """
    global _run_id, _output_dir, _run_meta  # noqa PLW0603
    _run_id = str(uuid5(_NAMESPACE_KMS, source))
    _output_dir = output_dir
    _run_meta = dict(meta, source=source)


def _ensure_run_dir(module_name: str) -> Path:
    """Return the run directory for *module_name*, creating it lazily."""
    global _run_id, _output_dir, _run_meta  # noqa: PLW0603
    if _run_id is None:
        _run_id = str(uuid5(_NAMESPACE_KMS, 'default'))
    run_dir = Path(_output_dir) / module_name / _run_id
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / 'meta.json'
        if not meta_path.exists() and _run_meta:
            meta_path.write_text(
                json.dumps(
                    {
                        'run_id': _run_id,
                        'created': datetime.now(UTC).isoformat(),
                        **_run_meta,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
    return run_dir


def _is_data_url(value: str) -> bool:
    """Return True if *value* is a base64 data URL."""
    return bool(re.match(r'^data:[^;]+;base64,', value))


def _jsonable(value: object) -> object:
    """Coerce one recorded value into something ``json.dumps`` accepts.

    Most stages declare their DSPy fields as Pydantic models — a single
    ``DSPyModel``, or a ``list[DSPyModel]`` for the stages that emit a whole
    page of nodes — so both the inputs and the prediction routinely arrive as
    models nested inside lists. Recursion is what makes those cases work:
    coercing only the top level (the previous behaviour) served the corrector
    and formatter, whose fields are plain strings, and raised ``TypeError`` on
    every other stage.

    Args:
        value: A recorded input or output value.

    Returns:
        The same value with Pydantic models dumped to dicts, containers
        rebuilt from coerced members, and anything else JSON cannot represent
        rendered as its string form.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)


def _serialize_images(inputs: dict, images_dir: Path) -> dict:
    """Replace ``dspy.Image`` values with sidecar file paths.

    Every other value is coerced by ``_jsonable``, so a model-valued input
    field records as a dict rather than aborting the write.
    """
    index = 0
    serialized: dict = {}
    for name, value in inputs.items():
        if isinstance(value, dspy.Image):
            images_dir.mkdir(parents=True, exist_ok=True)
            filename = f'{name}_{index}.png'
            sidecar = images_dir / filename
            _write_image_sidecar(value, sidecar)
            serialized[name] = f'images/{filename}'
            index += 1
        else:
            serialized[name] = _jsonable(value)
    return serialized


def _write_image_sidecar(image: dspy.Image, path: Path) -> None:
    """Write a ``dspy.Image`` to disk from its URL."""
    url = image.url or ''
    if _is_data_url(url):
        path.write_bytes(base64.b64decode(url.split(',', 1)[1]))
    else:
        source = Path(url)
        if source.exists():
            path.write_bytes(source.read_bytes())


def _deserialize_images(inputs: dict, run_dir: Path, image_fields: frozenset[str]) -> dict:
    """Rebuild ``dspy.Image`` values from sidecar paths."""
    deserialized: dict = {}
    for name, value in inputs.items():
        if name in image_fields and isinstance(value, str):
            deserialized[name] = dspy.Image(url=str(run_dir / value))
        else:
            deserialized[name] = value
    return deserialized


def record_example(
    module_name: str,
    inputs: dict,
    prediction: dspy.Prediction,
) -> None:
    """Append one prediction to the module's JSONL corpus.

    Called inside each module's ``aforward`` — a recording side effect
    that does not change the return type. No-op when ``set_run`` has not
    been called (recording is opt-in).

    Failures are logged and swallowed: recording is a side channel for
    training data, so a bad value or an unwritable corpus must cost that one
    example, never the document being ingested. Raising here aborts the whole
    LangGraph run from inside a worker.

    Args:
        module_name: Stable key for the module (e.g. ``'corrector'``).
        inputs: The keyword arguments the module was called with.
        prediction: The ``dspy.Prediction`` returned by ``acall``.
    """
    global _run_id  # noqa PLW0603
    if _run_id is None:
        return
    try:
        run_dir = _ensure_run_dir(module_name)
        images_dir = run_dir / 'images'
        jsonl = run_dir / 'examples.jsonl'

        record = {
            'inputs': _serialize_images(inputs, images_dir),
            'outputs': _jsonable(dict(prediction)),
        }
        with jsonl.open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    except (TypeError, ValueError, OSError):
        logger.warning('recorder: failed to record a %s example', module_name, exc_info=True)


def load_examples(
    module_name: str,
    run_name: str | None = None,
    *,
    output_dir: str = 'output/examples',
    image_fields: frozenset[str] = frozenset(),
) -> list[dspy.Example]:
    """Read recorded examples back as ``dspy.Example`` objects.

    Args:
        module_name: The key the examples were recorded under.
        run_name: A specific run id to load, or None for the latest.
        output_dir: Root directory of recorded examples.
        image_fields: Names of input fields that were image sidecars.

    Returns:
        One example per recorded call, each with ``with_inputs`` set so
        a consumer can split ``inputs()`` from ``labels()``. Empty if
        nothing was recorded.
    """
    base = Path(output_dir) / module_name
    if not base.exists():
        return []

    if run_name:
        run_dir = base / run_name
    else:
        run_dirs = sorted(entry for entry in base.iterdir() if entry.is_dir())
        run_dir = run_dirs[-1] if run_dirs else None

    if not run_dir or not run_dir.exists():
        return []

    jsonl = run_dir / 'examples.jsonl'
    if not jsonl.exists():
        return []

    examples: list[dspy.Example] = []
    for line in jsonl.read_text().strip().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        inputs = record['inputs']
        outputs = record['outputs']

        inputs = _deserialize_images(inputs, run_dir, image_fields)

        example = dspy.Example(**inputs, **outputs)
        example = example.with_inputs(*inputs.keys())
        examples.append(example)

    return examples
