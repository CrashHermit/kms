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


class Recorder:
    def __init__(
        self,
        source: str,
        output_dir: str = 'output/examples',
        **meta: object,
    ) -> None:
        self._run_id = str(uuid5(_NAMESPACE_KMS, source))
        self._output_dir = output_dir
        self._run_meta = dict(meta, source=source)

    def record(
        self,
        module_name: str,
        inputs: dict,
        prediction: dspy.Prediction,
    ) -> None:
        try:
            run_dir = self._ensure_run_dir(module_name)
            images_dir = run_dir / 'images'
            jsonl = run_dir / 'examples.jsonl'

            record = {
                'inputs': _serialize_images(inputs, images_dir),
                'outputs': _jsonable(dict(prediction)),
            }
            with jsonl.open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        except (TypeError, ValueError, OSError):
            logger.warning(
                'recorder: failed to record a %s example',
                module_name,
                exc_info=True,
            )

    def _ensure_run_dir(self, module_name: str) -> Path:
        run_dir = Path(self._output_dir) / module_name / self._run_id
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            meta_path = run_dir / 'meta.json'
            if not meta_path.exists() and self._run_meta:
                meta_path.write_text(
                    json.dumps(
                        {
                            'run_id': self._run_id,
                            'created': datetime.now(UTC).isoformat(),
                            **self._run_meta,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
        return run_dir


def _is_data_url(value: str) -> bool:
    return bool(re.match(r'^data:[^;]+;base64,', value))


def _jsonable(value: object) -> object:
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
    url = image.url or ''
    if _is_data_url(url):
        path.write_bytes(base64.b64decode(url.split(',', 1)[1]))
    else:
        source = Path(url)
        if source.exists():
            path.write_bytes(source.read_bytes())


def _deserialize_images(
    inputs: dict, run_dir: Path, image_fields: frozenset[str]
) -> dict:
    deserialized: dict = {}
    for name, value in inputs.items():
        if name in image_fields and isinstance(value, str):
            deserialized[name] = dspy.Image(url=str(run_dir / value))
        else:
            deserialized[name] = value
    return deserialized


def load_examples(
    module_name: str,
    run_name: str | None = None,
    *,
    output_dir: str = 'output/examples',
    image_fields: frozenset[str] = frozenset(),
) -> list[dspy.Example]:
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
