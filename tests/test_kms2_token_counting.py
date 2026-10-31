"""Provider-aware token semantics used by inference sizing."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from kms2.composition.predictors import build_token_budget
from kms2.config.inference import TextInferenceSettings
from kms2.config.llm import ChatGPTLLMProfile, LLMSettings, LocalLLMProfile
from kms2.config.settings import Settings
from kms2.core.windowing import TextTokenCounter
from kms2.inference.embedding import VoyageEmbeddingTokenCounter
from kms2.inference.token_counting import InferenceTokenizers
from kms2.local_models.token_counting import LocalTokenCounter


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

    counter = LocalTokenCounter(tokenizer_path)

    assert counter.count_texts(['alpha beta', '', 'γ', 'alpha beta γ']) == [
        2,
        0,
        1,
        3,
    ]


def test_hosted_embedding_counts_special_tokens_without_truncation(tmp_path):
    tokenizer_path = tmp_path / 'tokenizer.json'
    _write_tokenizer(tokenizer_path)
    counter = VoyageEmbeddingTokenCounter(str(tokenizer_path))

    assert counter.count_texts(['alpha beta', '', 'γ', 'alpha beta γ']) == [
        4,
        2,
        3,
        5,
    ]


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
    tokenizers = InferenceTokenizers(settings)

    counter: TextTokenCounter = tokenizers.for_inference(inference)
    assert (
        counter.count_texts(['This is an estimated cloud-token count.'])[0] > 0
    )
    assert tokenizers.text_counters(inference) == (counter,)

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
        InferenceTokenizers(settings).for_inference(
            TextInferenceSettings(profile='missing')
        )
