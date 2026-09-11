import asyncio

import dspy

from kms2.module.source import content_correction


class _Predictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(corrected_content='corrected')

    async def acall(self, **kwargs: object) -> dspy.Prediction:
        self.calls.append(kwargs)
        return dspy.Prediction(corrected_content='corrected async')


def _module() -> tuple[content_correction.ContentCorrectorModule, _Predictor]:
    predictor = _Predictor()
    module = content_correction.ContentCorrectorModule(predictor)
    return module, predictor


def test_signature_exposes_full_block_contract():
    assert set(content_correction.ContentCorrectorSignature.input_fields) == {
        'block_crop',
        'block_type',
        'content',
    }
    assert set(content_correction.ContentCorrectorSignature.output_fields) == {
        'corrected_content',
    }


def test_forward_passes_full_block_inputs():
    module, predictor = _module()

    result = module(
        block_crop=dspy.Image(url='data:image/png;base64,AA=='),
        block_type='text',
        content='original',
    )

    assert result == 'corrected'
    assert predictor.calls == [
        {
            'block_crop': dspy.Image(url='data:image/png;base64,AA=='),
            'block_type': 'text',
            'content': 'original',
        }
    ]


def test_aforward_passes_full_block_inputs():
    module, predictor = _module()

    result = asyncio.run(
        module.aforward(
            block_crop=dspy.Image(url='data:image/png;base64,AA=='),
            block_type='equation',
            content='original',
        )
    )

    assert result == 'corrected async'
    assert predictor.calls[0]['block_type'] == 'equation'
    assert predictor.calls[0]['content'] == 'original'
