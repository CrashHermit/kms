"""Provider-aware token semantics used by inference sizing."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from kms.composition.predictors import build_token_budget
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.inference import TextInferenceSettings
from kms.config.llm import ChatGPTLLMProfile, LLMSettings, LocalLLMProfile
from kms.config.settings import Settings
from kms.core.content import ContentImage
from kms.core.tokenization.images import ImageMetadataCache
from kms.core.tokenization.material import (
    MaterialTokenEstimator,
    estimate_image_tokens,
)
from kms.inference.tokenization.text import HuggingFaceTextTokenCounter


class _WordCounter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def count_texts(self, texts: list[str]) -> list[int]:
        self.calls.extend(texts)
        return [len(text.split()) for text in texts]


def _image_bytes(size: tuple[int, int]) -> bytes:
    output = BytesIO()
    Image.new('RGB', size, color='white').save(output, format='PNG')
    return output.getvalue()


def test_image_token_estimates_use_integer_32_pixel_patches():
    assert estimate_image_tokens(1, 1) == 2
    assert estimate_image_tokens(32, 32) == 2
    assert estimate_image_tokens(33, 32) == 3
    assert estimate_image_tokens(1024, 1024) == 1229
    assert estimate_image_tokens(2048, 2048) == 4916
    assert estimate_image_tokens(1, 1024) == 39


def test_combined_counter_counts_text_and_native_image_material(tmp_path):
    payload = _image_bytes((64, 64))
    path = tmp_path / 'figure.png'
    path.write_bytes(payload)
    path_image = ContentImage(('path', str(path)), path)
    bytes_image = ContentImage(('bytes', 'figure'), payload)
    image_metadata = ImageMetadataCache()
    text_counter = _WordCounter()
    counter = MaterialTokenEstimator(text_counter, image_metadata)

    assert image_metadata.measure(path_image).byte_count == len(payload)
    assert image_metadata.measure(bytes_image).width == 64
    assert counter.estimate() == 0
    assert counter.estimate(images=[bytes_image]) == 5
    assert text_counter.calls == []
    assert counter.estimate(text='one two') == 2
    assert counter.estimate(text='one two', images=[bytes_image]) == 7
    assert counter.estimate(images=[bytes_image, bytes_image]) == 10


def test_image_token_counter_propagates_missing_and_invalid_image_errors(
    tmp_path,
):
    counter = MaterialTokenEstimator(_WordCounter())

    with pytest.raises(FileNotFoundError):
        counter.estimate_images(
            [
                ContentImage(
                    ('path', str(tmp_path / 'missing.png')),
                    tmp_path / 'missing.png',
                )
            ]
        )
    with pytest.raises(OSError):
        counter.estimate_images(
            [ContentImage(('bytes', 'invalid'), b'not an image')]
        )


def _write_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                '[UNK]': 0,
                'alpha': 1,
                'beta': 2,
                'γ': 3,
                '[BOS]': 4,
                '[EOS]': 5,
                '[PAD]': 6,
            },
            unk_token='[UNK]',
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single='[BOS] $A [EOS]',
        pair='[BOS] $A [EOS] $B:1 [EOS]:1',
        special_tokens=[('[BOS]', 4), ('[EOS]', 5)],
    )
    tokenizer.enable_truncation(max_length=2)
    tokenizer.enable_padding(length=6, pad_id=6, pad_token='[PAD]')
    tokenizer.save(str(path))


def test_local_counter_uses_raw_unpadded_untruncated_counts(tmp_path):
    tokenizer_path = tmp_path / 'tokenizer.json'
    _write_tokenizer(tokenizer_path)

    counter = HuggingFaceTextTokenCounter(
        str(tokenizer_path), add_special_tokens=False
    )

    assert counter.count_texts(['alpha beta', '', 'γ', 'alpha beta γ']) == [
        2,
        0,
        1,
        3,
    ]


def test_huggingface_counter_can_count_special_tokens(tmp_path):
    tokenizer_path = tmp_path / 'tokenizer.json'
    _write_tokenizer(tokenizer_path)
    counter = HuggingFaceTextTokenCounter(
        str(tokenizer_path), add_special_tokens=True
    )

    assert counter.count_texts(['alpha beta', '', 'γ', 'alpha beta γ']) == [
        4,
        2,
        3,
        5,
    ]


def test_huggingface_counter_loads_only_for_nonempty_text(tmp_path):
    missing_path = tmp_path / 'missing-tokenizer.json'
    counter = HuggingFaceTextTokenCounter(
        str(missing_path), add_special_tokens=False
    )

    assert counter.count_texts([]) == []
    assert counter.count_texts(['']) == [0]
    with pytest.raises(Exception, match='No such file or directory'):
        counter.count_texts(['text'])


def test_chatgpt_counter_and_budget_do_not_load_local_tokenizers():
    settings = Settings(
        llm=LLMSettings(
            default_text_profile='subscription',
            profiles={
                'subscription': ChatGPTLLMProfile(
                    context_size=1_000,
                    max_input_tokens=300,
                    tokenizer_encoding='o200k_base',
                )
            },
        )
    )
    inference = TextInferenceSettings(profile='subscription', max_tokens=100)
    tokenizers = LanguageModelTokenizers(settings)

    counter: MaterialTokenEstimator = tokenizers.for_inference(inference)
    assert (
        counter.count_texts(['This is an estimated cloud-token count.'])[0] > 0
    )
    assert tokenizers.counters(inference) == (counter,)

    budget = build_token_budget(
        settings,
        tokenizers,
        inference,
        input_token_budget=10_000,
        safety_margin_tokens=50,
    )
    assert budget.counter is counter
    assert budget.token_limit == 300


def test_unknown_stage_profile_is_a_configuration_error():
    settings = Settings(
        llm=LLMSettings(
            default_text_profile='known',
            profiles={'known': LocalLLMProfile(model_server_profile='known')},
        )
    )

    with pytest.raises(ValueError, match='Unknown LLM profile: missing'):
        LanguageModelTokenizers(settings).for_inference(
            TextInferenceSettings(profile='missing')
        )
