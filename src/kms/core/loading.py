"""Load recorded pipeline runs back into DSPy examples.

The inverse of :mod:`kms.core.recording`: reads the on-disk format and
reconstructs each recorded call as a :class:`dspy.Example` in the
signature's field space, with images re-hydrated from their
content-addressed sidecars.
"""

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Union, get_args, get_origin

import dspy
from pydantic import BaseModel

from kms.core import recording

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """One stage's recorded examples together with its signature."""

    stage: str
    signature: type[dspy.Signature]
    examples: list[dspy.Example]


def load_datasets(output_dir: str | Path) -> list[Dataset]:
    """Loads recorded examples grouped by stage.

    Each returned :class:`Dataset` carries the stage's signature (the
    field-space authority) and its examples, marked with ``with_inputs``
    so they drop into a DSPy optimizer or evaluator directly.
    """
    root = Path(output_dir)
    if not root.exists():
        return []
    datasets: list[Dataset] = []
    for stage_dir in sorted(root.iterdir()):
        if stage_dir.name == 'images' or not stage_dir.is_dir():
            continue
        records = _read_records(stage_dir)
        if not records:
            continue
        signature = _import_signature(records[0]['schema'])
        datasets.append(
            Dataset(
                stage=stage_dir.name,
                signature=signature,
                examples=[
                    _to_example(record, signature, root) for record in records
                ],
            )
        )
    return datasets


def _read_records(stage_dir: Path) -> list[dict]:
    """Reads every record line across a stage's run directories."""
    records: list[dict] = []
    for jsonl in sorted(stage_dir.rglob('examples.jsonl')):
        for line in jsonl.read_text().splitlines():
            records.append(json.loads(line))
    return records


def _import_signature(qualified: str) -> type[dspy.Signature]:
    """Imports a signature class by its qualified name."""
    module_name, _, class_name = qualified.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _to_example(
    record: dict, signature: type[dspy.Signature], root: Path
) -> dspy.Example:
    """Reconstructs one record as a signature-space example."""
    inputs = {
        name: _hydrate(
            record['inputs'][name],
            signature.input_fields[name].annotation,
            root,
        )
        for name in record['inputs']
    }
    outputs = {
        name: _hydrate(
            record['outputs'][name],
            signature.output_fields[name].annotation
            if name in signature.output_fields
            else None,
            root,
        )
        for name in record['outputs']
    }
    return dspy.Example(**inputs, **outputs).with_inputs(
        *tuple(signature.input_fields)
    )


def _hydrate(value: object, annotation: object, root: Path) -> object:
    """Reconstructs a recorded value according to its type annotation."""
    if annotation is None:
        return value
    if annotation is dspy.Image:
        return _hydrate_image(value, root)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is UnionType or origin is Union:
        return _hydrate_union(value, args, root)
    if origin is list:
        elem = args[0] if args else None
        return [_hydrate(item, elem, root) for item in value]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _hydrate_model(annotation, value, root)
    return value


def _hydrate_union(value: object, args: tuple, root: Path) -> object:
    """Reconstructs a union value by matching its runtime shape."""
    if value is None:
        return None
    non_none = [arg for arg in args if arg is not NoneType]
    if len(non_none) == 1:
        return _hydrate(value, non_none[0], root)
    for candidate in non_none:
        if _matches(candidate, value):
            return _hydrate(value, candidate, root)
    return value


def _matches(annotation: object, value: object) -> bool:
    """Whether a union branch's annotation fits a recorded value."""
    if annotation is dspy.Image:
        return _is_sidecar_ref(value)
    if annotation is str:
        return isinstance(value, str) and not _is_sidecar_ref(value)
    if annotation in (int, float, bool):
        return isinstance(value, annotation)
    origin = get_origin(annotation)
    if origin in (list, tuple, set):
        return isinstance(value, (list, tuple))
    if origin is dict:
        return isinstance(value, dict)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return isinstance(value, dict) and set(value) == set(
            annotation.model_fields
        )
    return True


def _hydrate_model(
    cls: type[BaseModel], value: object, root: Path
) -> BaseModel | None:
    """Reconstructs a pydantic model, resolving image_path sidecars."""
    if value is None:
        return None
    if isinstance(value, cls):
        return value
    fields = {}
    for name, field in cls.model_fields.items():
        if name not in value:
            continue
        item = value[name]
        if (
            name == recording.IMAGE_FIELD
            and isinstance(item, str)
            and _is_sidecar_ref(item)
        ):
            fields[name] = str(root / item)
        else:
            fields[name] = _hydrate(item, field.annotation, root)
    return cls.model_validate(fields)


def _hydrate_image(value: object, root: Path) -> dspy.Image:
    """Reconstructs a dspy.Image from a sidecar reference or URL."""
    if value is None:
        return None
    url = str(value)
    if _is_sidecar_ref(url):
        url = str(root / url)
    return dspy.Image(url=url)


def _is_sidecar_ref(value: object) -> bool:
    """Whether a value is a recorded sidecar reference."""
    return isinstance(value, str) and value.startswith('images/')
