"""Load recorded KMS2 calls into typed DSPy examples."""

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, get_origin

import dspy
from pydantic import TypeAdapter


@dataclass(frozen=True, slots=True)
class Dataset:
    """Recorded examples for one module and signature pair."""

    module: str
    signature: type[dspy.Signature]
    examples: list[dspy.Example]


def load_datasets(directory: Path) -> list[Dataset]:
    """Load all recorded examples grouped by module and signature."""
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, object]]]] = {}
    for run_directory in sorted(directory.iterdir()):
        if not run_directory.is_dir():
            continue
        for module_directory in sorted(run_directory.iterdir()):
            if not module_directory.is_dir():
                continue
            for record_path in sorted(module_directory.glob('*.jsonl')):
                for line in record_path.read_text(
                    encoding='utf-8'
                ).splitlines():
                    record = json.loads(line)
                    key = (module_directory.name, record['signature'])
                    grouped.setdefault(key, []).append((run_directory, record))

    datasets = []
    for (module, qualified_signature), records in sorted(grouped.items()):
        signature = _import_signature(qualified_signature)
        datasets.append(
            Dataset(
                module=module,
                signature=signature,
                examples=[
                    _to_example(record, signature, run_directory)
                    for run_directory, record in records
                ],
            )
        )
    return datasets


def _import_signature(qualified_name: str) -> type[dspy.Signature]:
    """Import a signature class from its qualified name."""
    module_name, _, class_name = qualified_name.rpartition('.')
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _to_example(
    record: dict[str, object],
    signature: type[dspy.Signature],
    run_directory: Path,
) -> dspy.Example:
    """Reconstruct one recording as a typed signature-space example."""
    raw_inputs = record['inputs']
    raw_outputs = record['outputs']
    inputs = {
        name: _hydrate(
            value,
            signature.input_fields[name].annotation,
            run_directory,
        )
        for name, value in raw_inputs.items()
    }
    outputs = {
        name: (
            _hydrate(
                value,
                signature.output_fields[name].annotation,
                run_directory,
            )
            if name in signature.output_fields
            else value
        )
        for name, value in raw_outputs.items()
    }
    return dspy.Example(**inputs, **outputs).with_inputs(
        *signature.input_fields
    )


def _hydrate(value: object, annotation: object, run_directory: Path) -> object:
    """Reconstruct one declared field according to its annotation."""
    if annotation is dspy.Image:
        return dspy.Image(url=str(run_directory / value))

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and args and args[0] is dspy.Image:
        return [dspy.Image(url=str(run_directory / item)) for item in value]

    return TypeAdapter(annotation).validate_python(value)
