import asyncio
import base64
import hashlib
import json
from pathlib import Path
from uuid import UUID

import dspy
from pydantic import BaseModel

from kms2.train.recorder import Recorder, RecordingModule


class _Request(BaseModel):
    index: int
    text: str


class _Signature(dspy.Signature):
    requests: list[_Request] = dspy.InputField()
    decisions: list[bool] = dspy.OutputField()


class _ImageSignature(dspy.Signature):
    images: list[dspy.Image] = dspy.InputField()
    result: dspy.Image = dspy.OutputField()


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


def _recording_files(directory: Path) -> list[Path]:
    run_directories = list(directory.iterdir())
    assert len(run_directories) == 1
    module_directory = (
        run_directories[0] / f'{_Module.__module__}.{_Module.__qualname__}'
    )
    return list(module_directory.glob('*.jsonl'))


def test_recorder_writes_qualified_one_record_jsonl_file(
    tmp_path: Path,
) -> None:
    recorder = Recorder(tmp_path)
    recorder.record(
        _Module,
        _Signature,
        {'requests': [_Request(index=1, text='claim')]},
        dspy.Prediction(decisions=[True]),
    )

    [path] = _recording_files(tmp_path)
    assert path.parent.name == f'{_Module.__module__}.{_Module.__qualname__}'
    assert UUID(path.stem).version == 4
    assert len(path.read_text(encoding='utf-8').splitlines()) == 1
    record = json.loads(path.read_text(encoding='utf-8'))

    assert record == {
        'signature': f'{_Signature.__module__}.{_Signature.__qualname__}',
        'inputs': {'requests': [{'index': 1, 'text': 'claim'}]},
        'outputs': {'decisions': [True]},
    }


def test_recorder_writes_each_prediction_to_a_distinct_file(
    tmp_path: Path,
) -> None:
    recorder = Recorder(tmp_path)

    recorder.record(
        _Module,
        _Signature,
        {'requests': []},
        dspy.Prediction(decisions=[]),
    )
    recorder.record(
        _Module,
        _Signature,
        {'requests': [_Request(index=2, text='second')]},
        dspy.Prediction(decisions=[False]),
    )

    paths = _recording_files(tmp_path)
    assert len(paths) == 2
    records = [json.loads(path.read_text(encoding='utf-8')) for path in paths]

    assert sorted(record['outputs']['decisions'] for record in records) == [
        [],
        [False],
    ]
    assert all(
        len(path.read_text(encoding='utf-8').splitlines()) == 1
        for path in paths
    )


def test_recorder_stores_images_as_run_sidecars(
    tmp_path: Path,
) -> None:
    data = b'png bytes'
    digest = hashlib.sha256(data).hexdigest()
    image_path = tmp_path / 'source.png'
    image_path.write_bytes(data)
    data_url = 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')
    recorder = Recorder(tmp_path / 'recordings')

    recorder.record(
        _Module,
        _ImageSignature,
        {
            'images': [
                dspy.Image(url=data_url),
                dspy.Image(url=str(image_path)),
            ]
        },
        dspy.Prediction(result=dspy.Image(url=data_url)),
    )

    [recording_path] = _recording_files(tmp_path / 'recordings')
    run_directory = recording_path.parents[1]
    record = json.loads(recording_path.read_text(encoding='utf-8'))
    reference = f'images/{digest}.png'

    assert record['inputs']['images'] == [reference, reference]
    assert record['outputs']['result'] == reference
    assert (run_directory / reference).read_bytes() == data
    assert len(list((run_directory / 'images').iterdir())) == 1


def test_recording_module_preserves_sync_prediction(tmp_path: Path) -> None:
    module = RecordingModule(
        _Module(),
        _Module,
        _Signature,
        Recorder(tmp_path),
    )

    prediction = module(
        requests=[_Request(index=1, text='claim')],
    )

    assert prediction.decisions == [True]
    [path] = _recording_files(tmp_path)
    record = json.loads(path.read_text(encoding='utf-8'))
    assert record['outputs']['decisions'] == [True]


def test_recording_module_preserves_async_prediction(tmp_path: Path) -> None:
    module = RecordingModule(
        _Module(),
        _Module,
        _Signature,
        Recorder(tmp_path),
    )

    async def run() -> dspy.Prediction:
        return await module.acall(
            requests=[_Request(index=0, text='claim')],
        )

    prediction = asyncio.run(run())

    assert prediction.decisions == [False]
    [path] = _recording_files(tmp_path)
    record = json.loads(path.read_text(encoding='utf-8'))
    assert record['outputs']['decisions'] == [False]
