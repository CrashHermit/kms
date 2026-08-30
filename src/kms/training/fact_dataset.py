"""Load curated natural-language fact examples into their DSPy shape."""

from __future__ import annotations

from pathlib import Path

import dspy
from pydantic import BaseModel, ConfigDict, Field

from kms.core import models


class FactTrainingRecord(BaseModel):
    """One persisted natural-language fact-training record."""

    model_config = ConfigDict(extra='forbid')

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    request: models.FactExtractionInput
    facts: list[models.AtomicFact]


def load_examples(path: str | Path) -> list[dspy.Example]:
    """Load strict, self-contained natural-language fact examples."""
    examples: list[dspy.Example] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = FactTrainingRecord.model_validate_json(line)
        except ValueError as error:
            raise ValueError(
                f'invalid fact training record on line {line_number}'
            ) from error
        examples.append(
            dspy.Example(
                request=record.request,
                facts=record.facts,
            ).with_inputs('request')
        )
    return examples
