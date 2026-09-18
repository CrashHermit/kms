import asyncio
import json
from pathlib import Path

import dspy
from pydantic import BaseModel

from kms2.train.recorder import Recorder, RecordingModule


class _Request(BaseModel):
    index: int
    text: str


class _Signature(dspy.Signature):
    requests: list[_Request] = dspy.InputField()
    decisions: list[bool] = dspy.OutputField()


class _Module(dspy.Module):
    def forward(self, *, requests: list[_Request]) -> dspy.Prediction:
        return dspy.Prediction(
            decisions=[request.index > 0 for request in requests]
        )

    async def aforward(
        self,
        *,
        requests: list[_Request],
    ) -> dspy.Prediction:
        return self.forward(requests=requests)


def test_recorder_writes_flat_named_jsonl_example(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)
    recorder.record(
        _Signature,
        {'requests': [_Request(index=1, text='claim')]},
        dspy.Prediction(decisions=[True]),
    )

    path = tmp_path / '_Signature.jsonl'
    record = json.loads(path.read_text(encoding='utf-8'))

    assert record == {
        'requests': [{'index': 1, 'text': 'claim'}],
        'decisions': [True],
    }


def test_recorder_appends_records_for_one_signature(tmp_path: Path) -> None:
    recorder = Recorder(tmp_path)

    recorder.record(
        _Signature,
        {'requests': []},
        dspy.Prediction(decisions=[]),
    )
    recorder.record(
        _Signature,
        {'requests': [_Request(index=2, text='second')]},
        dspy.Prediction(decisions=[False]),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / '_Signature.jsonl')
        .read_text(encoding='utf-8')
        .splitlines()
    ]

    assert records == [
        {'requests': [], 'decisions': []},
        {
            'requests': [{'index': 2, 'text': 'second'}],
            'decisions': [False],
        },
    ]


def test_recording_module_preserves_sync_prediction(tmp_path: Path) -> None:
    module = RecordingModule(
        _Module(),
        _Signature,
        Recorder(tmp_path),
    )

    prediction = module(
        requests=[_Request(index=1, text='claim')],
    )

    assert prediction.decisions == [True]
    record = json.loads(
        (tmp_path / '_Signature.jsonl').read_text(encoding='utf-8')
    )
    assert record['decisions'] == [True]


def test_recording_module_preserves_async_prediction(tmp_path: Path) -> None:
    module = RecordingModule(
        _Module(),
        _Signature,
        Recorder(tmp_path),
    )

    async def run() -> dspy.Prediction:
        return await module.acall(
            requests=[_Request(index=0, text='claim')],
        )

    prediction = asyncio.run(run())

    assert prediction.decisions == [False]
    record = json.loads(
        (tmp_path / '_Signature.jsonl').read_text(encoding='utf-8')
    )
    assert record['decisions'] == [False]
