import asyncio
import base64
import io
from pathlib import Path

import dspy
import pytest
from PIL import Image

from kms.construction import (
    block_corrector,
    entity_enrichment,
    triplet_extractor,
)
from kms.core import loading, models, recording


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new('RGBA', (3, 2), (255, 0, 0, 255)).save(buf, format='PNG')
    return buf.getvalue()


def _data_url(data: bytes) -> str:
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


class _OtherSignature(dspy.Signature):
    text: str = dspy.InputField()
    answer: str = dspy.OutputField()


def _by_stage(output_dir: Path) -> dict[str, loading.Dataset]:
    return {
        dataset.stage: dataset for dataset in loading.load_datasets(output_dir)
    }


def test_rejects_mixed_schemas_in_one_stage(tmp_path):
    output_dir = tmp_path / 'ex'
    first = recording.Recorder('src-a', output_dir=str(output_dir))
    second = recording.Recorder('src-b', output_dir=str(output_dir))
    first.record(
        'stage',
        block_corrector.BlockCorrectionSignature,
        {'block_crop': None, 'block_type': 'text', 'lines': '[1] hi'},
        dspy.Prediction(edits=[]),
    )
    second.record(
        'stage',
        _OtherSignature,
        {'text': 'hi'},
        dspy.Prediction(answer='ok'),
    )

    with pytest.raises(ValueError, match='mixed recorded schemas'):
        loading.load_datasets(output_dir)


def test_loads_a_dspy_image_input(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    recorder.record(
        'corrector_block_editor',
        block_corrector.BlockCorrectionSignature,
        {
            'block_crop': dspy.Image(url=_data_url(_png_bytes())),
            'block_type': 'text',
            'lines': [{'index': 1, 'text': 'hi'}],
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


def test_loads_structured_entity_context_input(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    recorder.record(
        'entity_enrichment',
        entity_enrichment.EntityEnrichmentSignature,
        {
            'request': models.TermEnrichmentInput(
                context_before=[
                    models.NodeInput(
                        index=1,
                        node_type='paragraph',
                        text='Before',
                    )
                ],
                target_node=models.NodeInput(
                    index=1,
                    node_type='image',
                    text='A diagram',
                ),
                context_after=[
                    models.NodeInput(
                        index=1,
                        node_type='paragraph',
                        text='After',
                    )
                ],
                terms=['vector space'],
            )
        },
        dspy.Prediction(description='A directed quantity.'),
    )

    example = _by_stage(tmp_path / 'ex')['entity_enrichment'].examples[0]
    assert example.request.target_node.text == 'A diagram'
    assert example.request.context_before[0].text == 'Before'
    assert example.request.context_after[0].text == 'After'
    assert example.request.terms == ['vector space']


def test_loads_structured_fact_context_inputs(tmp_path):
    recorder = recording.Recorder('src', output_dir=str(tmp_path / 'ex'))
    recorder.record(
        'atomic_fact_extractor',
        triplet_extractor._FactSignature,
        {
            'request': models.FactExtractionInput(
                context_before=[
                    models.NodeInput(
                        index=1, node_type='paragraph', text='Before'
                    )
                ],
                target_node=models.NodeInput(
                    index=1, node_type='image', text='A diagram'
                ),
                context_after=[],
            )
        },
        dspy.Prediction(facts=[]),
    )

    example = _by_stage(tmp_path / 'ex')['atomic_fact_extractor'].examples[0]
    request = example.request
    assert request.context_before[0].text == 'Before'
    assert request.target_node.node_type == 'image'
    assert request.target_node.text == 'A diagram'
    assert request.context_after == []

    class _Fake:
        def __init__(self, **values):
            self.values = values

        async def acall(self, **kwargs):
            assert kwargs['lines'] == [models.LineInput(index=1, text='hi')]
            return dspy.Prediction(**self.values)

    module = block_corrector.BlockCorrector(
        language_model=dspy.LM('openai/dummy', api_key='x'),
        recorder=recorder,
    )
    module.router.predictor = _Fake(needs_correction=True)
    module.locator.predictor = _Fake(locations=[models.LineSelection(index=1)])
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
    assert example.lines == [models.LineInput(index=1, text='hi')]
