"""Behavioral coverage for native evidence preparation and request capacity."""

import asyncio
import base64
import threading
from dataclasses import replace
from io import BytesIO

import dspy
import pytest
from dspy.utils import DummyLM
from PIL import Image

from kms.config.inference import VisionInferenceSettings
from kms.config.settings import Settings
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import (
    NativePredictor,
    fits_generation_request,
    generation_prompt,
    generation_signature,
)
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.media.images import load_content_images
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.source.context import PositionedSourceBlock, SourceBlockContext
from kms.source.models.source import BlockType
from kms.source.processing.exercise_splitter.inference import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
)
from kms.source.processing.exercise_splitter.models import SplitCandidate
from kms.train.loading import load_datasets
from kms.train.recorder import Recorder


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Signature(dspy.Signature):
    """Use this request's native evidence without borrowing from another task."""

    material: Material = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    answer: str = dspy.OutputField()


class _MustNotRun(dspy.Module):
    async def aforward(self, **inputs: object) -> object:
        raise AssertionError(
            'over-capacity or unsupported request reached backend'
        )


def _asset(key: str = 'figure') -> ImageAsset:
    output = BytesIO()
    with Image.new('RGB', (4, 4), 'red') as image:
        image.save(output, format='PNG')
    return ImageAsset(('private-asset-key', key), output.getvalue())


def _budget() -> TokenBudget:
    return TokenBudget(MaterialTokenEstimator(_Counter()), 10_000)


def test_actual_dspy_prompt_keeps_part_order_and_one_native_shared_image():
    asset = _asset()
    material = Material(
        parts=(
            TextPart(text='before-the-figure'),
            ImagePart(asset=asset),
            TextPart(text='after-the-figure'),
            ImagePart(asset=asset),
        )
    )
    backend = dspy.Predict(generation_signature(_Signature))
    lm = DummyLM([{'answer': 'native evidence received'}])
    backend.set_lm(lm)
    native = NativePredictor(
        backend,
        _Signature,
        budget=_budget(),
        image_budget=ImagePayloadBudget(1, 100_000, 100_000),
        supports_images=True,
    )

    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        asyncio.run(native.acall(material=material))

    [entry] = lm.history
    message = [item for item in entry['messages'] if item['role'] == 'user'][-1]
    image_parts = [
        part for part in message['content'] if part['type'] == 'image_url'
    ]
    assert len(image_parts) == 1
    assert (
        base64.b64decode(image_parts[0]['image_url']['url'].split(',', 1)[1])
        == asset.payload
    )
    text = '\n'.join(
        part['text'] for part in message['content'] if part['type'] == 'text'
    )
    assert (
        text.index('before-the-figure')
        < text.index('image_position')
        < text.index('after-the-figure')
    )
    assert text.count('image_position') == 2
    assert 'private-asset-key' not in text
    assert base64.b64encode(asset.payload).decode() not in text


def test_splitter_prompt_includes_reference_text_and_native_image():
    asset = _asset()
    reference = PositionedSourceBlock(
        position=0,
        source_block=SourceBlockContext(
            block_type=BlockType.PARAGRAPH,
            material=Material(
                parts=(
                    TextPart(text='reference diagram'),
                    ImagePart(asset=asset),
                )
            ),
        ),
    )
    candidate = SplitCandidate(
        position=1,
        source_block=SourceBlockContext(
            block_type=BlockType.PARAGRAPH,
            material=Material.from_text('1. Find area. 2. Find perimeter.'),
        ),
    )
    backend = dspy.Predict(generation_signature(ExerciseSplitterSignature))
    lm = DummyLM(
        [
            {
                'splits': [
                    {
                        'position': 1,
                        'pieces': [
                            {'content': '1. Find area. '},
                            {'content': '2. Find perimeter.'},
                        ],
                    }
                ]
            }
        ]
    )
    backend.set_lm(lm)
    native = NativePredictor(
        backend,
        ExerciseSplitterSignature,
        budget=_budget(),
        image_budget=ImagePayloadBudget(1, 100_000, 100_000),
        supports_images=True,
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        asyncio.run(
            ExerciseSplitterModule(native).acall(
                context_before=[],
                candidates=[candidate],
                reference_blocks=[reference],
                context_after=[],
            )
        )

    [entry] = lm.history
    message = [item for item in entry['messages'] if item['role'] == 'user'][-1]
    text = '\n'.join(
        part['text'] for part in message['content'] if part['type'] == 'text'
    )
    assert 'reference diagram' in text
    assert 'image_position' in text
    [image] = [
        part for part in message['content'] if part['type'] == 'image_url'
    ]
    assert base64.b64decode(image['image_url']['url'].split(',', 1)[1]) == (
        asset.payload
    )


def test_live_workload_and_complete_prompt_have_independent_limits():
    examples = [
        dspy.Example(
            material=Material.from_text('demonstration ' * 200),
            answer='example',
        ).with_inputs('material')
    ]
    prompt, images = generation_prompt(_Signature, examples)
    budget = replace(
        _budget(),
        format_request=prompt,
        context_token_limit=20_000,
        prompt_images=images,
    )
    inputs = {'material': Material.from_text('short live evidence')}
    assert fits_generation_request(inputs, budget=budget, image_budget=None)
    assert not fits_generation_request(
        inputs, budget=replace(budget, token_limit=1), image_budget=None
    )
    assert not fits_generation_request(
        inputs,
        budget=replace(budget, context_token_limit=1_000),
        image_budget=None,
    )


@pytest.mark.parametrize('constraint', ['count', 'bytes', 'pixels'])
def test_demo_and_live_catalogs_share_whole_request_image_capacity(constraint):
    asset = _asset()
    example = dspy.Example(
        material=Material.from_image(asset), answer='example'
    ).with_inputs('material')
    prompt, images = generation_prompt(_Signature, [example])
    budget = replace(
        _budget(),
        format_request=prompt,
        context_token_limit=20_000,
        prompt_images=images,
    )
    inputs = {'material': Material.from_image(asset)}
    limits = {
        'count': ImagePayloadBudget(1, 100_000, 100_000),
        'bytes': ImagePayloadBudget(2, len(asset.payload) * 2 - 1, 100_000),
        'pixels': ImagePayloadBudget(2, 100_000, 31),
    }
    assert not fits_generation_request(
        inputs, budget=budget, image_budget=limits[constraint]
    )
    assert fits_generation_request(
        inputs,
        budget=budget,
        image_budget=ImagePayloadBudget(2, len(asset.payload) * 2, 32),
    )


def test_full_prompt_overflow_fails_before_loading_or_model_dispatch(
    monkeypatch,
):
    example = dspy.Example(
        material=Material.from_text('large demonstration ' * 200),
        answer='example',
    ).with_inputs('material')
    prompt, images = generation_prompt(_Signature, [example])
    budget = replace(
        _budget(),
        format_request=prompt,
        context_token_limit=1_000,
        prompt_images=images,
    )

    async def unexpected_read(*args, **kwargs):
        raise AssertionError('overflow reached native image loading')

    monkeypatch.setattr(
        'kms.inference.generation.load_content_images', unexpected_read
    )
    native = NativePredictor(
        _MustNotRun(),
        _Signature,
        budget=budget,
        image_budget=ImagePayloadBudget(4, 100_000, 100_000),
        supports_images=True,
    )
    with pytest.raises(InputBudgetExceeded, match='Complete generation prompt'):
        asyncio.run(native.acall(material=Material.from_image(_asset())))


def test_text_only_profile_rejects_native_demonstrations_even_for_text_live_input():
    budget = replace(_budget(), prompt_images=(_asset(),))
    native = NativePredictor(
        _MustNotRun(),
        _Signature,
        budget=budget,
        image_budget=ImagePayloadBudget(4, 100_000, 100_000),
        supports_images=False,
    )
    with pytest.raises(ValueError, match='does not support native images'):
        asyncio.run(
            native.acall(material=Material.from_text('text-only live input'))
        )


def test_cancelled_image_preparation_joins_active_file_work(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    asset = _asset()

    def read_image(_asset):
        started.set()
        assert release.wait(5)
        completed.set()
        return asset.payload

    monkeypatch.setattr('kms.media.images._read_image', read_image)

    async def run():
        task = asyncio.create_task(load_content_images([asset], budget=None))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert completed.is_set()

    asyncio.run(run())


def test_factory_recording_reloads_ordered_native_material_for_replay(
    monkeypatch, tmp_path
):
    lm = DummyLM([{'answer': 'recorded native evidence'}])
    monkeypatch.setattr(
        'kms.inference.predictors.dspy.LM', lambda *args, **kwargs: lm
    )
    settings = Settings(_env_file=None)
    factory = PredictorFactory(
        settings,
        object(),
        Recorder(tmp_path),
        tokenizers=LanguageModelTokenizers(settings),
    )
    native = factory.create(
        dspy.Predict,
        VisionInferenceSettings(),
        _Signature,
        budget=_budget(),
        image_budget=ImagePayloadBudget(2, 100_000, 100_000),
    )
    asset = _asset('recorded')
    original = Material(
        parts=(
            TextPart(text='before'),
            ImagePart(asset=asset),
            TextPart(text='after'),
            ImagePart(asset=asset),
        )
    )
    with dspy.context(
        adapter=dspy.ChatAdapter(use_json_adapter_fallback=False)
    ):
        asyncio.run(native.acall(material=original))

    [dataset] = load_datasets(tmp_path)
    [example] = dataset.examples
    restored = example.material
    assert isinstance(restored, Material)
    assert [part.type for part in restored.parts] == [
        'text',
        'image',
        'text',
        'image',
    ]
    assert restored.text == 'beforeafter'
    assert restored.images[0].payload.read_bytes() == asset.payload
    assert restored.images[0] == restored.images[1]
    assert set(example.inputs().keys()) == {'material'}
    assert example.answer == 'recorded native evidence'
