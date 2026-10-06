"""Local tokenizer semantics used by token sizing."""

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, processors

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
