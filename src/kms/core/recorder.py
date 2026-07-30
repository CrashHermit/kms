"""
Record DSPy module inputs and outputs as examples in per-module JSON files.

Each call a stage chooses to record appends one ``(inputs, outputs)`` pair to
``<output_dir>/<module_name>.json``. ``load_examples`` reads them back as
``dspy.Example`` objects, so a recorded run is a reusable corpus of the
material the pipeline actually produced.

``dspy.Image`` values are stored as their data/URL string and rebuilt on load
for the fields the caller names, since JSON has no image type.
"""

import json
from pathlib import Path

import dspy


def _serialize(value: object) -> object:
    """The JSON-safe form of one recorded value.

    Args:
        value: The value a module was called with or returned.

    Returns:
        The image's url for a ``dspy.Image``, otherwise the value unchanged.
    """
    return value.url if isinstance(value, dspy.Image) else value


def _deserialize(value: object, deserialize_images: bool) -> object:
    """The in-memory form of one recorded value.

    Args:
        value: The value as it was stored in JSON.
        deserialize_images: Whether this field holds an image url.

    Returns:
        A ``dspy.Image`` when the field is an image url, otherwise the value
        unchanged.
    """
    return (
        dspy.Image(url=value)
        if deserialize_images and isinstance(value, str)
        else value
    )


def record_example(
    module_name: str,
    inputs: dict,
    outputs: dict,
    output_dir: str = 'output/examples',
) -> None:
    """Append one (inputs, outputs) pair to a module's examples file.

    Args:
        module_name: Stable key for the module (e.g. ``'extractor'``).
        inputs: Keyword args the module was called with.
        outputs: Keyword args the module returned.
        output_dir: Directory to write ``<module_name>.json`` files into.
    """
    path = Path(output_dir) / f'{module_name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    if path.exists():
        examples = json.loads(path.read_text())

    examples.append(
        {
            'inputs': {
                name: _serialize(value) for name, value in inputs.items()
            },
            'outputs': {
                name: _serialize(value) for name, value in outputs.items()
            },
        }
    )
    path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))


def load_examples(
    module_name: str,
    output_dir: str = 'output/examples',
    image_fields: frozenset[str] = frozenset(),
) -> list[dspy.Example]:
    """Read recorded examples back as ``dspy.Example`` objects.

    Args:
        module_name: The key the examples were recorded under.
        output_dir: Directory the ``<module_name>.json`` file lives in.
        image_fields: Names of the fields holding image urls, rebuilt as
            ``dspy.Image`` values.

    Returns:
        One example per recorded call, each with ``with_inputs`` set from the
        recorded input keys so a consumer can split ``inputs()`` from
        ``labels()``. Empty if nothing was recorded for the module.
    """
    path = Path(output_dir) / f'{module_name}.json'
    if not path.exists():
        return []

    data: list[dict] = json.loads(path.read_text())
    return [
        dspy.Example(
            **{
                name: _deserialize(value, name in image_fields)
                for section in ('inputs', 'outputs')
                for name, value in entry[section].items()
            }
        ).with_inputs(*entry['inputs'].keys())
        for entry in data
    ]
