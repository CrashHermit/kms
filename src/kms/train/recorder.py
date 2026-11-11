"""Persist DSPy calls as one-record JSONL training examples."""

import base64
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import dspy
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from kms.concurrency import run_blocking_work
from kms.media.images import load_content_images_sync
from kms.media.models import ImageAsset


def _qualified_name(value: type[object]) -> str:
    """Return the import-qualified name for a class."""
    return f'{value.__module__}.{value.__qualname__}'


def _serialize(value: object, images_directory: Path) -> object:
    """Convert a prediction value into JSON data and image sidecars."""
    if isinstance(value, dspy.Image):
        return _serialize_image(value, images_directory)
    if isinstance(value, ImageAsset):
        reference = _write_image(
            load_content_images_sync([value], budget=None)[0], images_directory
        )
        return {'key': list(value.key), 'payload': {'$image': reference}}
    if isinstance(value, BaseModel):
        return {
            name: _serialize(getattr(value, name), images_directory)
            for name, field in type(value).model_fields.items()
            if not field.exclude
        }
    if isinstance(value, Mapping):
        return {
            key: _serialize(item, images_directory)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize(item, images_directory) for item in value]
    return to_jsonable_python(value)


def _serialize_image(image: dspy.Image, images_directory: Path) -> str:
    """Write an image once and return its run-relative sidecar reference."""
    if image.url.startswith('data:'):
        _, encoded = image.url.split(',', 1)
        data = base64.b64decode(encoded)
    else:
        data = Path(image.url).read_bytes()
    return _write_image(data, images_directory)


def _write_image(data: bytes, images_directory: Path) -> str:
    """Store exact native bytes once and return their run-relative reference."""
    digest = hashlib.sha256(data).hexdigest()
    path = images_directory / f'{digest}.png'
    if not path.exists():
        images_directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return f'images/{digest}.png'


class Recorder:
    """Persist one successful DSPy prediction per UUID4-named JSONL file."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        timestamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')
        self._run_directory = directory / timestamp

    def record(
        self,
        module: type[dspy.Module],
        signature: type[dspy.Signature],
        inputs: dict[str, object],
        prediction: dspy.Prediction,
    ) -> None:
        """Persist one successful prediction in the current recording run."""
        record = {
            'signature': _qualified_name(signature),
            'inputs': _serialize(
                {
                    name: value
                    for name, value in inputs.items()
                    if name in signature.input_fields
                },
                self._run_directory / 'images',
            ),
            'outputs': _serialize(
                dict(prediction),
                self._run_directory / 'images',
            ),
        }

        module_directory = self._run_directory / _qualified_name(module)
        module_directory.mkdir(parents=True, exist_ok=True)
        path = module_directory / f'{uuid4()}.jsonl'
        path.write_text(
            f'{json.dumps(record, ensure_ascii=False)}\n',
            encoding='utf-8',
        )


class RecordingModule(dspy.Module):
    """Record calls to a DSPy module without changing its predictions."""

    def __init__(
        self,
        predictor: dspy.Module,
        module: type[dspy.Module],
        signature: type[dspy.Signature],
        recorder: Recorder,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.module = module
        self.signature = signature
        self.recorder = recorder

    def forward(self, **inputs: object) -> dspy.Prediction:
        """Run and record one synchronous module call."""
        prediction = self.predictor(**inputs)
        self.recorder.record(self.module, self.signature, inputs, prediction)
        return prediction

    async def aforward(
        self,
        **inputs: object,
    ) -> dspy.Prediction:
        """Run and record one asynchronous module call."""
        prediction = await self.predictor.acall(**inputs)
        await run_blocking_work(
            self.recorder.record,
            self.module,
            self.signature,
            inputs,
            prediction,
        )
        return prediction
