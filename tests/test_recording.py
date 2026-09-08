import base64
import hashlib
import io
import json
from pathlib import Path

import dspy
from PIL import Image
from pydantic import BaseModel

from kms.core import recording


class _Sig(dspy.Signature):
    """A test signature for recording."""

    text: str = dspy.InputField(description='Some input text.')
    ok: str = dspy.OutputField(description='An answer.')


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new('RGBA', (3, 2), (255, 0, 0, 255)).save(buf, format='PNG')
    return buf.getvalue()


def _data_url(data: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def _records(tmp_path, module: str) -> tuple[list[dict], Path]:
    jsonl = next((tmp_path / 'ex' / module).rglob('examples.jsonl'))
    records = [json.loads(line) for line in jsonl.read_text().splitlines()]
    return records, jsonl.parent


def _record(recorder, *, inputs, prediction, **kwargs) -> None:
    recorder.record('corrector', _Sig, inputs, prediction, **kwargs)


def test_records_dspy_image_as_a_sidecar(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    data = _png_bytes()
    _record(
        recorder,
        inputs={'page_image': dspy.Image(url=_data_url(data)), 'text': 'hi'},
        prediction=dspy.Prediction(ok='yes'),
        model='openai/dummy',
        duration_ms=12.5,
    )

    records, _ = _records(tmp_path, 'corrector')
    digest = hashlib.sha256(data).hexdigest()
    assert records[0]['inputs'] == {
        'page_image': f'images/{digest}.png',
        'text': 'hi',
    }
    assert records[0]['outputs'] == {'ok': 'yes'}
    assert records[0]['model'] == 'openai/dummy'
    assert records[0]['duration_ms'] == 12.5
    assert (tmp_path / 'ex' / 'images' / f'{digest}.png').read_bytes() == data


def test_records_image_path_fields_as_sidecars(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    data = _png_bytes()
    image_file = tmp_path / 'fig.png'
    image_file.write_bytes(data)

    class _Node(BaseModel):
        position: int
        type: str
        content: str
        image_path: str | None = None

    _record(
        recorder,
        inputs={
            'current_nodes': [
                _Node(
                    position=0,
                    type='image',
                    content='',
                    image_path=str(image_file),
                ),
                _Node(position=1, type='text', content='hi'),
            ]
        },
        prediction=dspy.Prediction(splits=[]),
    )

    records, _ = _records(tmp_path, 'corrector')
    digest = hashlib.sha256(data).hexdigest()
    nodes = records[0]['inputs']['current_nodes']
    assert nodes[0]['image_path'] == f'images/{digest}.png'
    assert nodes[1]['image_path'] is None
    assert nodes[1]['content'] == 'hi'
    assert (tmp_path / 'ex' / 'images' / f'{digest}.png').read_bytes() == data


def test_the_same_image_is_stored_once(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    data = _png_bytes()
    image = dspy.Image(url=_data_url(data))
    _record(recorder, inputs={'a': image}, prediction=dspy.Prediction(ok=True))
    _record(recorder, inputs={'a': image}, prediction=dspy.Prediction(ok=True))

    digest = hashlib.sha256(data).hexdigest()
    images = list((tmp_path / 'ex' / 'images').iterdir())
    assert [p.name for p in images] == [f'{digest}.png']


def test_distinct_images_get_distinct_sidecars(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    a = _png_bytes()
    buf = io.BytesIO()
    Image.new('RGBA', (4, 2), (0, 0, 255, 255)).save(buf, format='PNG')
    b = buf.getvalue()
    _record(
        recorder,
        inputs={
            'a': dspy.Image(url=_data_url(a)),
            'b': dspy.Image(url=_data_url(b)),
        },
        prediction=dspy.Prediction(ok=True),
    )

    records, _ = _records(tmp_path, 'corrector')
    inputs = records[0]['inputs']
    assert inputs['a'] == f'images/{hashlib.sha256(a).hexdigest()}.png'
    assert inputs['b'] == f'images/{hashlib.sha256(b).hexdigest()}.png'
    assert len(list((tmp_path / 'ex' / 'images').iterdir())) == 2


def test_manifest_records_the_signature_schema(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    _record(
        recorder,
        inputs={'text': 'hi'},
        prediction=dspy.Prediction(
            ok='yes', analysis='I checked the input before answering.'
        ),
    )

    manifest = json.loads((tmp_path / 'ex' / 'manifest.json').read_text())
    assert manifest['format_version'] == recording.FORMAT_VERSION
    key = f'{_Sig.__module__}.{_Sig.__qualname__}'
    schema = manifest['signatures'][key]
    assert schema['docstring'] == _Sig.__doc__
    assert schema['inputs']['text']['description'] == 'Some input text.'
    assert schema['outputs']['ok']['description'] == 'An answer.'
    assert schema['outputs']['analysis']['type'] == 'str'


def test_records_auxiliary_prediction_fields(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    _record(
        recorder,
        inputs={'text': 'hi'},
        prediction=dspy.Prediction(
            ok='yes', analysis='I checked the input before answering.'
        ),
    )

    records, _ = _records(tmp_path, 'corrector')
    assert records[0]['outputs']['analysis'] == (
        'I checked the input before answering.'
    )


def test_same_source_gets_a_fresh_run_namespace(tmp_path):
    output_dir = tmp_path / 'ex'
    first = recording.Recorder('src', output_dir=str(output_dir))
    second = recording.Recorder('src', output_dir=str(output_dir))
    assert first._run_id != second._run_id

    _record(first, inputs={'text': 'one'}, prediction=dspy.Prediction(ok='yes'))
    _record(
        second, inputs={'text': 'two'}, prediction=dspy.Prediction(ok='yes')
    )

    runs = list((output_dir / 'corrector').iterdir())
    assert len(runs) == 2


def test_run_metadata_records_source_stage_and_model(tmp_path):
    recorder = recording.Recorder(
        'src', output_dir=str(tmp_path / 'ex'), title='A Book'
    )
    _record(
        recorder,
        inputs={'text': 'hi'},
        prediction=dspy.Prediction(ok='yes'),
        model='openai/dummy',
    )

    meta = json.loads(
        next((tmp_path / 'ex' / 'corrector').rglob('meta.json')).read_text()
    )
    assert meta['source'] == 'src'
    assert meta['title'] == 'A Book'
    assert meta['stage'] == 'corrector'
    assert meta['model'] == 'openai/dummy'
    assert meta['format_version'] == recording.FORMAT_VERSION
