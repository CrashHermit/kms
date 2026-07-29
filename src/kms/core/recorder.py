"""
Store DSPy module inputs and outputs as examples in per-module JSON files,
reloadable as ``dspy.Example`` objects for optimiser training sets.
"""

import json
from pathlib import Path

import dspy


def _serialize(value):
    return value.url if isinstance(value, dspy.Image) else value


def _deserialize(value, deserialize_images: bool):
    return dspy.Image(url=value) if deserialize_images and isinstance(value, str) else value


def record_example(
    module_name: str,
    inputs: dict,
    outputs: dict,
    output_dir: str = 'output/examples',
) -> None:
    """Append one (inputs, outputs) pair to a module's examples file.

    Args:
        module_name: Stable key for the module (e.g. ``"extractor"``).
        inputs: Keyword args the module was called with.
        outputs: Keyword args the module returned.
        output_dir: Directory to write ``<module_name>.json`` files into.
    """
    path = Path(output_dir) / f'{module_name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    if path.exists():
        examples = json.loads(path.read_text())

    examples.append({
        'inputs': {k: _serialize(v) for k, v in inputs.items()},
        'outputs': {k: _serialize(v) for k, v in outputs.items()},
    })
    path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))


def load_examples(
    module_name: str,
    output_dir: str = 'output/examples',
    image_fields: frozenset[str] = frozenset(),
) -> list[dspy.Example]:
    """Read recorded examples back as ``dspy.Example`` objects.

    Each example has ``with_inputs`` set from the recorded input keys, so
    DSPy optimisers can split ``inputs()`` from ``labels()``.
    """
    path = Path(output_dir) / f'{module_name}.json'
    if not path.exists():
        return []

    data: list[dict] = json.loads(path.read_text())
    return [
        dspy.Example(
            **{
                **{
                    k: _deserialize(v, k in image_fields)
                    for k, v in entry['inputs'].items()
                },
                **{
                    k: _deserialize(v, k in image_fields)
                    for k, v in entry['outputs'].items()
                },
            }
        ).with_inputs(*entry['inputs'].keys())
        for entry in data
    ]
