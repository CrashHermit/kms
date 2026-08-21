import asyncio
import base64
import io
from pathlib import Path

import dspy
from PIL import Image

from kms.construction import (
    block_corrector,
    entity_enrichment,
    triplet_extractor,
)
from kms.core import content, loading, recording, walker


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new('RGBA', (3, 2), (255, 0, 0, 255)).save(buf, format='PNG')
    return buf.getvalue()


def _data_url(data: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def _by_stage(output_dir: Path) -> dict[str, loading.Dataset]:
    return {
        dataset.stage: dataset for dataset in loading.load_datasets(output_dir)
    }


def test_loads_a_dspy_image_input(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    recorder.record(
        'corrector_block_editor',
        block_corrector.BlockCorrectionSignature,
        {
            'block_crop': dspy.Image(url=_data_url(_png_bytes())),
            'block_type': 'text',
            'lines': '[1] hi',
        },
        dspy.Prediction(edits=[]),
    )

    dataset = _by_stage(tmp_path / 'ex')['corrector_block_editor']
    assert dataset.signature is block_corrector.BlockCorrectionSignature
    example = dataset.examples[0]
    assert set(example.inputs().keys()) == {
        'block_crop',
        'block_type',
        'lines',
    }
    assert isinstance(example.block_crop, dspy.Image)
    assert 'base64' in example.block_crop.url


def test_loads_auxiliary_prediction_fields(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    recorder.record(
        'corrector_block_editor',
        block_corrector.BlockCorrectionSignature,
        {
            'block_crop': None,
            'block_type': 'text',
            'lines': '[1] hi',
        },
        dspy.Prediction(edits=[]),
    )


def test_loads_a_content_parts_input(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    passage = content.ContentParts(
        content=content.Content(
            parts=[
                content.TextPart(text='hi'),
                content.ImagePart(
                    image=dspy.Image(url=_data_url(_png_bytes()))
                ),
            ]
        )
    )
    recorder.record(
        'component_enrichment',
        entity_enrichment.EntityEnrichmentSignature,
        {'passage': passage, 'terms': ['vector space']},
        dspy.Prediction(descriptions=[]),
    )

    example = _by_stage(tmp_path / 'ex')['component_enrichment'].examples[0]
    assert isinstance(example.passage, content.ContentParts)
    parts = example.passage.content.parts
    assert parts[0].text == 'hi'
    assert isinstance(parts[1].image, dspy.Image)
    assert example.terms == ['vector space']


def test_resolves_image_path_sidecars(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    image_file = tmp_path / 'fig.png'
    image_file.write_bytes(_png_bytes())
    node = walker.WindowNode(
        position=0,
        id=1,
        type='image',
        content='',
        image_path=str(image_file),
    )
    recorder.record(
        'atomic_fact_extractor',
        triplet_extractor._FactSignature,
        {'current_nodes': [node]},
        dspy.Prediction(facts=[]),
    )

    example = _by_stage(tmp_path / 'ex')['atomic_fact_extractor'].examples[0]
    loaded = example.current_nodes[0]
    assert isinstance(loaded, walker.WindowNode)
    assert Path(loaded.image_path).exists()
    assert loaded.image_path != str(image_file)


def test_round_trip_through_a_module(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))

    class _Fake:
        def __init__(self, **values):
            self.values = values

        async def acall(self, **kwargs):
            assert kwargs['lines'] == '[1] hi'
            return dspy.Prediction(**self.values)

    module = block_corrector.BlockCorrector(
        language_model=dspy.LM('openai/dummy', api_key='x'),
        recorder=recorder,
    )
    module.router.predictor = _Fake(needs_correction=True)
    module.editor.predictor = _Fake(edits=[])
    image_path = tmp_path / 'block.png'
    image_path.write_bytes(_png_bytes())
    asyncio.run(
        module.acorrect(
            type(
                'Region',
                (),
                {
                    'crop_path': str(image_path),
                    'block': type(
                        'Block', (), {'type': 'text', 'content': 'hi'}
                    )(),
                },
            )()
        )
    )

    example = _by_stage(tmp_path / 'ex')['corrector_block_editor'].examples[0]
    assert isinstance(example.block_crop, dspy.Image)
    assert example.lines == '[1] hi'
