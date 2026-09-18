"""Persist DSPy calls as flat JSONL training examples."""

import json
from pathlib import Path

import dspy
from pydantic_core import to_jsonable_python


class Recorder:
    """Append named DSPy inputs and outputs to one file per signature."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def record(
        self,
        signature: type[dspy.Signature],
        inputs: dict[str, object],
        prediction: dspy.Prediction,
    ) -> None:
        """Append one successful prediction as a flat JSONL record."""
        values = {
            **{name: inputs[name] for name in signature.input_fields},
            **prediction.toDict(),
        }
        record = to_jsonable_python(values)

        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f'{signature.__name__}.jsonl'
        with path.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False))
            output.write('\n')


class RecordingModule(dspy.Module):
    """Record calls to a DSPy module without changing its predictions."""

    def __init__(
        self,
        module: dspy.Module,
        signature: type[dspy.Signature],
        recorder: Recorder,
    ) -> None:
        super().__init__()
        self.module = module
        self.signature = signature
        self.recorder = recorder

    def forward(self, **inputs: object) -> dspy.Prediction:
        """Run and record one synchronous module call."""
        prediction = self.module(**inputs)
        self.recorder.record(self.signature, inputs, prediction)
        return prediction

    async def aforward(
        self,
        **inputs: object,
    ) -> dspy.Prediction:
        """Run and record one asynchronous module call."""
        prediction = await self.module.acall(**inputs)
        self.recorder.record(self.signature, inputs, prediction)
        return prediction
