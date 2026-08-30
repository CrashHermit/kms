"""Record per-module LLM call examples, capturing images as sidecars."""

import base64
import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import dspy
from pydantic import BaseModel

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1

IMAGE_FIELD = 'image_path'


class Recorder:
    """Persists LLM inputs and predictions as replayable examples.

    Each module gets its own run directory under the output directory,
    keyed by a fresh run id. Records capture the
    signature-form inputs the LM actually saw, the model that produced
    them, and per-call timing. Images are written once as
    content-addressed sidecars under a shared ``images/`` directory, so
    the same image referenced by many windows is stored once. Each
    signature's field schema is kept in a ``manifest.json`` so records
    can be replayed without re-deriving the prompt.
    """

    def __init__(
        self,
        source: str,
        output_dir: str = 'output/examples',
        **meta: object,
    ) -> None:
        self._run_id = uuid4().hex
        self._output_dir = Path(output_dir)
        self._run_meta = dict(meta, source=source)
        self._manifest: dict[str, dict] = {}
        self._manifest_loaded = False

    def record(
        self,
        module_name: str,
        signature: type[dspy.Signature],
        inputs: dict,
        prediction: dspy.Prediction,
        *,
        model: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Appends one example record for a module call.

        Serialization failures are logged and swallowed so recording
        never breaks the pipeline.

        Args:
            module_name: The module that produced the prediction.
            signature: The signature class the module uses.
            inputs: The signature-form inputs the LM was called with.
            prediction: The prediction to record.
            model: The language model name, when known.
            duration_ms: Wall-clock duration of the call in milliseconds.
        """
        try:
            images_dir = self._output_dir / 'images'
            run_dir = self._ensure_run_dir(module_name, signature, model)
            prediction_fields = dict(prediction)
            record = {
                'schema': _qualified_name(signature),
                'inputs': _serialize(inputs, images_dir),
                'outputs': _serialize(prediction_fields, images_dir),
                'timestamp': datetime.now(UTC).isoformat(),
            }
            if model is not None:
                record['model'] = model
            if duration_ms is not None:
                record['duration_ms'] = duration_ms
            with (run_dir / 'examples.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            self._record_schema(signature, prediction_fields)
        except (TypeError, ValueError, OSError):
            logger.warning(
                'recorder: failed to record a %s example',
                module_name,
                exc_info=True,
            )

    def record_progress(
        self,
        stage: str,
        *,
        status: str,
        duration_ms: float,
        output_keys: list[str] | None = None,
    ) -> None:
        """Records one workflow-stage timing event."""
        try:
            progress_path = self._output_dir / 'progress.jsonl'
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                'stage': stage,
                'status': status,
                'duration_ms': duration_ms,
                'timestamp': datetime.now(UTC).isoformat(),
            }
            if output_keys is not None:
                record['output_keys'] = sorted(output_keys)
            with progress_path.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        except (TypeError, ValueError, OSError):
            logger.warning(
                'recorder: failed to record progress for %s',
                stage,
                exc_info=True,
            )

    def _ensure_run_dir(
        self,
        module_name: str,
        signature: type[dspy.Signature],
        model: str | None,
    ) -> Path:
        """Returns the run directory, creating it with metadata if new."""
        run_dir = self._output_dir / module_name / self._run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            meta_path = run_dir / 'meta.json'
            if not meta_path.exists():
                meta = {
                    'format_version': FORMAT_VERSION,
                    'run_id': self._run_id,
                    'stage': module_name,
                    'schema': _qualified_name(signature),
                    'created': datetime.now(UTC).isoformat(),
                    **self._run_meta,
                }
                if model is not None:
                    meta['model'] = model
                meta_path.write_text(
                    json.dumps(meta, indent=2, ensure_ascii=False)
                )
        return run_dir

    def _record_schema(
        self,
        signature: type[dspy.Signature],
        prediction: dict[str, object],
    ) -> None:
        """Registers the signature and observed prediction fields."""
        if not self._manifest_loaded:
            self._load_manifest()
        key = _qualified_name(signature)
        schema = self._manifest.get(key)
        schema_changed = schema is None
        if schema is None:
            schema = _signature_schema(signature)
            self._manifest[key] = schema
        for name, value in prediction.items():
            if name not in schema['outputs']:
                schema['outputs'][name] = _value_schema(value)
                schema_changed = True
        if not schema_changed:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        (self._output_dir / 'manifest.json').write_text(
            json.dumps(
                {
                    'format_version': FORMAT_VERSION,
                    'signatures': self._manifest,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    def _load_manifest(self) -> None:
        """Loads a pre-existing manifest, if any, for cumulative runs."""
        self._manifest_loaded = True
        manifest_path = self._output_dir / 'manifest.json'
        if not manifest_path.exists():
            return
        data = json.loads(manifest_path.read_text())
        self._manifest = data.get('signatures', {})


def _qualified_name(signature: type[dspy.Signature]) -> str:
    """Returns the qualified class name identifying a signature."""
    return f'{signature.__module__}.{signature.__qualname__}'


def _signature_schema(signature: type[dspy.Signature]) -> dict:
    """Dumps a signature's prompt and field descriptions."""
    return {
        'docstring': signature.__doc__ or '',
        'inputs': {
            name: _field_schema(field)
            for name, field in signature.input_fields.items()
        },
        'outputs': {
            name: _field_schema(field)
            for name, field in signature.output_fields.items()
        },
    }


def _field_schema(field: object) -> dict:
    """Dumps one signature field's description and type name."""
    return {
        'description': getattr(field, 'description', None) or '',
        'type': _type_name(getattr(field, 'annotation', None)),
    }


def _value_schema(value: object) -> dict:
    """Dumps the basic schema for an undeclared prediction field."""
    return {
        'description': 'Observed auxiliary prediction field.',
        'type': _type_name(type(value)),
    }


def _type_name(annotation: object) -> str:
    """Renders a field annotation as a readable type name."""
    if annotation is None:
        return 'Any'
    name = getattr(annotation, '__name__', None)
    return name or str(annotation)


def _serialize(value: object, images_dir: Path) -> object:
    """Serializes a value, capturing images as content-addressed sidecars.

    ``dspy.Image`` values and ``image_path`` strings are written into
    ``images_dir`` under their SHA-256 digest and replaced by their
    relative path, so recorded examples are self-contained and the same
    image is stored once.
    """
    if isinstance(value, dspy.Image):
        return _capture_image(value.url or '', images_dir)
    if isinstance(value, BaseModel):
        return {
            name: _serialize_field(name, getattr(value, name), images_dir)
            for name in type(value).model_fields
        }
    if isinstance(value, dict):
        return {
            str(key): _serialize_field(key, item, images_dir)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize(item, images_dir) for item in value]
    return _jsonable(value)


def _serialize_field(key: str, value: object, images_dir: Path) -> object:
    """Serializes one dict entry, sidecaring image_path references."""
    if key == IMAGE_FIELD and isinstance(value, str) and value:
        return _capture_image(value, images_dir)
    return _serialize(value, images_dir)


def _capture_image(source: str, images_dir: Path) -> str:
    """Writes an image to a content-addressed sidecar.

    Args:
        source: A data URL or a file path.
        images_dir: The shared directory holding sidecar files.

    Returns:
        The sidecar's relative path, or the original source when the
        image bytes cannot be read.
    """
    data = _image_bytes(source)
    if data is None:
        return source
    digest = hashlib.sha256(data).hexdigest()
    filename = f'{digest}.png'
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / filename
    if not path.exists():
        path.write_bytes(data)
    return f'images/{filename}'


def _image_bytes(source: str) -> bytes | None:
    """Returns the raw bytes for a data URL or file path, or None."""
    if _is_data_url(source):
        return base64.b64decode(source.split(',', 1)[1])
    path = Path(source)
    if path.exists():
        return path.read_bytes()
    return None


def _is_data_url(value: str) -> bool:
    """True if the string is a base64 data URL."""
    return bool(re.match(r'^data:[^;]+;base64,', value))


def _jsonable(value: object) -> object:
    """Converts a value into a JSON-serializable shape.

    Pydantic models are dumped, dicts and lists are converted
    recursively, and anything else becomes its string form.
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
