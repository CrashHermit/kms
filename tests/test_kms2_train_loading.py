import base64
import hashlib
from pathlib import Path

import dspy
from pydantic import BaseModel

from kms2.train.loading import load_datasets
from kms2.train.recorder import Recorder


class _Request(BaseModel):
    index: int
    text: str


class _Decision(BaseModel):
    accepted: bool


class _LoaderSignature(dspy.Signature):
    requests: list[_Request] = dspy.InputField()
    images: list[dspy.Image] = dspy.InputField()
    decisions: list[_Decision] = dspy.OutputField()
    preview: dspy.Image = dspy.OutputField()


class _OtherSignature(dspy.Signature):
    text: str = dspy.InputField()
    answer: str = dspy.OutputField()


class _LoaderModule(dspy.Module):
    pass


def _data_url(data: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def test_loader_reconstructs_typed_examples_and_run_relative_images(
    tmp_path: Path,
) -> None:
    data = b'loader image'
    digest = hashlib.sha256(data).hexdigest()
    recorder = Recorder(tmp_path / 'recordings')
    recorder.record(
        _LoaderModule,
        _LoaderSignature,
        {
            'requests': [_Request(index=1, text='claim')],
            'images': [dspy.Image(url=_data_url(data))],
        },
        dspy.Prediction(
            decisions=[_Decision(accepted=True)],
            preview=dspy.Image(url=_data_url(data)),
            auxiliary={'source': 'recorded'},
        ),
    )

    [dataset] = load_datasets(tmp_path / 'recordings')
    [example] = dataset.examples
    run_directory = next((tmp_path / 'recordings').iterdir())

    assert dataset.module == (
        f'{_LoaderModule.__module__}.{_LoaderModule.__qualname__}'
    )
    assert dataset.signature is _LoaderSignature
    assert set(example.inputs()) == {'requests', 'images'}
    assert example.requests == [_Request(index=1, text='claim')]
    assert isinstance(example.images[0], dspy.Image)
    sidecar = run_directory / f'images/{digest}.png'
    assert sidecar.read_bytes() == data
    assert example.images[0].url == _data_url(data)
    assert sidecar.is_file()
    assert isinstance(example.preview, dspy.Image)
    assert example.preview.url == _data_url(data)
    assert example.decisions == [_Decision(accepted=True)]
    assert example.auxiliary == {'source': 'recorded'}


def test_loader_groups_signatures_within_a_module_separately(
    tmp_path: Path,
) -> None:
    recorder = Recorder(tmp_path / 'recordings')
    recorder.record(
        _LoaderModule,
        _LoaderSignature,
        {'requests': [], 'images': []},
        dspy.Prediction(
            decisions=[],
            preview=dspy.Image(url=_data_url(b'group image')),
        ),
    )
    recorder.record(
        _LoaderModule,
        _OtherSignature,
        {'text': 'question'},
        dspy.Prediction(answer='response'),
    )

    datasets = load_datasets(tmp_path / 'recordings')

    assert [dataset.module for dataset in datasets] == [
        f'{_LoaderModule.__module__}.{_LoaderModule.__qualname__}',
        f'{_LoaderModule.__module__}.{_LoaderModule.__qualname__}',
    ]
    assert {dataset.signature for dataset in datasets} == {
        _LoaderSignature,
        _OtherSignature,
    }
