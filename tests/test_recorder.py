"""Recorder corpus — JSONL round-trip, image sidecars, append-only. No
network/LLM."""

import base64
import json
from pathlib import Path

import dspy
import pytest

from kms.core import recorder

MODEL = 'openrouter/qwen/qwen3-vl-235b-a22b-instruct'
PNG = b'\x89PNG\r\n\x1a\nfake page render'
OTHER_PNG = b'\x89PNG\r\n\x1a\na different page'


def _image(payload=PNG):
    """A dspy.Image carrying inlined bytes, as the corrector builds one."""
    encoded = base64.b64encode(payload).decode('utf-8')
    return dspy.Image(url=f'data:image/png;base64,{encoded}')


def _record(tmp_path, inputs=None, outputs=None, module='corrector', **kwargs):
    """Record one call, defaulting everything the test does not care about."""
    return recorder.record_example(
        module,
        {'transcription': 'orig'} if inputs is None else inputs,
        {'corrected': 'fixed'} if outputs is None else outputs,
        model=kwargs.pop('model', MODEL),
        corpus_root=tmp_path,
        run_id=kwargs.pop('run_id', 'run1'),
        **kwargs,
    )


def _lines(tmp_path, module='corrector', run='run1'):
    """The raw records as written, straight off disk."""
    path = Path(tmp_path) / module / run / recorder.EXAMPLES_FILENAME
    return [
        json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


def _images(tmp_path, module='corrector', run='run1'):
    return sorted(
        (Path(tmp_path) / module / run / recorder.IMAGES_DIRNAME).iterdir()
    )


def test_record_writes_three_sections_with_metadata(tmp_path):
    record_id = _record(tmp_path)
    (record,) = _lines(tmp_path)
    assert set(record) == {'metadata', 'inputs', 'outputs'}
    assert record['metadata'] == {'id': record_id, 'model': MODEL}
    assert record['inputs'] == {'transcription': 'orig'}
    assert record['outputs'] == {'corrected': 'fixed'}


def test_appends_rather_than_rewrites(tmp_path):
    for text in ('one', 'two', 'three'):
        _record(tmp_path, {'transcription': text})
    records = _lines(tmp_path)
    assert [r['inputs']['transcription'] for r in records] == [
        'one',
        'two',
        'three',
    ]
    # Distinct ids, so an adjudication can address one record.
    assert len({r['metadata']['id'] for r in records}) == 3


def test_image_is_written_beside_the_corpus_not_inlined(tmp_path):
    _record(tmp_path, {'transcription': 'orig', 'page_image': _image()})
    (record,) = _lines(tmp_path)
    reference = record['inputs']['page_image'][recorder.IMAGE_KEY]

    # Reference is relative to the corpus root, so concatenating runs keeps it
    # resolvable.
    assert reference.startswith('corrector/run1/images/')
    assert (tmp_path / reference).read_bytes() == PNG
    # No base64 payload survives in the line itself.
    assert 'base64' not in json.dumps(record)


def test_same_image_shares_one_sidecar_across_records(tmp_path):
    for _ in range(3):
        _record(tmp_path, {'page_image': _image()})
    _record(tmp_path, {'page_image': _image(OTHER_PNG)})
    # Content-hash naming: three samples of one page, one file; plus the other.
    assert len(_images(tmp_path)) == 2


def test_image_provenance_rides_alongside_the_reference(tmp_path):
    _record(
        tmp_path,
        {'page_image': _image()},
        image_provenance={
            'page_image': {
                'source_pdf': 'books/stein.pdf',
                'page_index': 39,
                'render_scale': 2.5,
            }
        },
    )
    (record,) = _lines(tmp_path)
    entry = record['inputs']['page_image']
    assert recorder.IMAGE_KEY in entry
    assert entry['source_pdf'] == 'books/stein.pdf'
    assert entry['page_index'] == 39
    assert entry['render_scale'] == 2.5


def test_round_trips_to_examples_with_inputs_split(tmp_path):
    _record(tmp_path, {'transcription': 'orig', 'page_image': _image()})
    (example,) = recorder.load_examples('corrector', corpus_root=tmp_path)

    assert example.transcription == 'orig'
    assert example.corrected == 'fixed'
    # The image comes back as a dspy.Image carrying the sidecar's bytes.
    assert isinstance(example.page_image, dspy.Image)
    assert example.page_image.url == _image().url
    # Provenance never enters the field space DSPy sees.
    assert 'model' not in example.toDict()
    assert set(example.inputs().toDict()) == {'transcription', 'page_image'}
    assert set(example.labels().toDict()) == {'corrected'}


def test_missing_sidecar_raises_rather_than_substituting(tmp_path):
    _record(tmp_path, {'page_image': _image()})
    for image in _images(tmp_path):
        image.unlink()

    # A pruned sidecar is data loss; a stand-in would quietly feed a training
    # run an image that is not the page.
    with pytest.raises(FileNotFoundError, match='images/'):
        recorder.load_examples('corrector', corpus_root=tmp_path)

    # The record itself still loads — only image rehydration fails.
    assert len(recorder.load_records('corrector', corpus_root=tmp_path)) == 1


def test_torn_line_is_skipped_not_fatal(tmp_path):
    _record(tmp_path, {'transcription': 'good'})
    path = tmp_path / 'corrector' / 'run1' / recorder.EXAMPLES_FILENAME
    with path.open('a', encoding='utf-8') as handle:
        handle.write('{"metadata": {"id": "trunc"}, "inputs": {"transcr')

    records = recorder.load_records('corrector', corpus_root=tmp_path)
    assert [r['inputs']['transcription'] for r in records] == ['good']


def test_runs_are_separate_and_loadable_together_or_singly(tmp_path):
    for run, text in (('run1', 'first'), ('run2', 'second')):
        _record(tmp_path, {'transcription': text}, run_id=run)

    every = recorder.load_records('corrector', corpus_root=tmp_path)
    assert [r['inputs']['transcription'] for r in every] == ['first', 'second']

    only = recorder.load_records(
        'corrector', corpus_root=tmp_path, run_id='run2'
    )
    assert [r['inputs']['transcription'] for r in only] == ['second']


def test_modules_get_their_own_directories(tmp_path):
    _record(tmp_path)
    _record(
        tmp_path,
        {'segment_markdown': 'b'},
        {'nodes': []},
        module='extractor',
        model='deepseek/deepseek-v4-flash',
    )
    assert {p.name for p in tmp_path.iterdir()} == {'corrector', 'extractor'}

    (record,) = recorder.load_records('extractor', corpus_root=tmp_path)
    assert record['metadata']['model'] == 'deepseek/deepseek-v4-flash'


def test_missing_corpus_loads_empty(tmp_path):
    assert recorder.load_records('corrector', corpus_root=tmp_path) == []
    assert recorder.load_examples('corrector', corpus_root=tmp_path) == []


def test_run_ids_are_unique_and_sortable():
    ids = [recorder.new_run_id() for _ in range(5)]
    assert len(set(ids)) == 5
    assert all(id_[:4].isdigit() for id_ in ids)


@pytest.mark.parametrize('url', ['https://example.com/page.png', 'data:,'])
def test_non_inlined_image_urls_are_kept_verbatim(tmp_path, url):
    _record(tmp_path, {'page_image': dspy.Image(url=url)})
    (record,) = _lines(tmp_path)
    assert record['inputs']['page_image'][recorder.IMAGE_KEY] == url
