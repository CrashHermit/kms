"""Load captured trainsets as ``dspy.Example`` lists for DSPy optimization.

The JSONL files under ``data/trainsets/`` (produced by running the pipeline with
``KMS_CAPTURE_DIR`` set — see ``capture.py``) hold plain ``{inputs, outputs}`` dicts.
This module reconstructs the signature-typed fields (WindowNode / EntitySpan / the
extractor's node model) and returns ``dspy.Example`` objects with the right input keys
marked, ready to hand to an optimizer's trainset.
"""

import json
from pathlib import Path

import dspy

from .entity_grouper import WindowNode, EntitySpan
from .extractor import DSPyModel

# data/trainsets at the repo root (this file lives at src/module/trainsets.py).
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "trainsets"


def _read(signature: str, directory: Path) -> list[dict]:
    path = Path(directory) / f"{signature}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extractor_examples(directory: Path = DEFAULT_DIR) -> list[dspy.Example]:
    out = []
    for rec in _read("extractor", directory):
        i, o = rec["inputs"], rec["outputs"]
        nodes = [DSPyModel(type=n["type"], content=n["content"]) for n in o["nodes"]]
        out.append(
            dspy.Example(**i, nodes=nodes).with_inputs(*i.keys())
        )
    return out


def entity_grouper_examples(directory: Path = DEFAULT_DIR) -> list[dspy.Example]:
    out = []
    for rec in _read("entity_grouper", directory):
        i, o = rec["inputs"], rec["outputs"]
        current = [WindowNode(**w) for w in i["current_nodes"]]
        spans = [EntitySpan(**s) for s in o["entities"]]
        out.append(
            dspy.Example(
                previous_context=i["previous_context"],
                current_nodes=current,
                next_context=i["next_context"],
                entities=spans,
            ).with_inputs("previous_context", "current_nodes", "next_context")
        )
    return out


def entity_attributor_examples(directory: Path = DEFAULT_DIR) -> list[dspy.Example]:
    out = []
    for rec in _read("entity_attributor", directory):
        i, o = rec["inputs"], rec["outputs"]
        out.append(
            dspy.Example(entity_type=i["entity_type"], members=i["members"], roles=o["roles"])
            .with_inputs("entity_type", "members")
        )
    return out


LOADERS = {
    "extractor": extractor_examples,
    "entity_grouper": entity_grouper_examples,
    "entity_attributor": entity_attributor_examples,
}


def load(signature: str, directory: Path = DEFAULT_DIR) -> list[dspy.Example]:
    """Load one signature's trainset. Raises KeyError for an unknown signature."""
    return LOADERS[signature](directory)
