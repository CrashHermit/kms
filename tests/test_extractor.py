"""Extractor — the LLM block type -> ASTNode mapping, and the furniture
discard. No network/LLM."""

import asyncio

import pytest

from kms.core import models
from kms.ingestion import extractor


def _block(node_type, content='text'):
    return extractor.DSPyModel(type=node_type, content=content)


class _Module:
    """Stands in for the LLM: returns a fixed block list for any page."""

    def __init__(self, blocks):
        self.blocks = blocks

    async def aforward(self, segment_markdown):
        return list(self.blocks)


def _worker(blocks, index=0):
    node = extractor.ExtractorNode(module=_Module(blocks))
    segment = models.Segment(index=index, image_path='', content='markdown')
    out = asyncio.run(node.worker({'segment': segment}))
    return out['extract_results'][0][1]


def test_every_prompt_type_is_in_the_valid_set():
    # The taxonomy the Signature names and the validation set the consumer
    # uses must agree, or a type the model is told to emit silently raises.
    described = extractor.DSPyModel.model_fields['type'].description
    for node_type in extractor._VALID_TYPES:
        assert node_type in described, (
            f'{node_type!r} missing from field description'
        )
        assert f'- {node_type}:' in extractor.Signature.__doc__, (
            f'{node_type!r} missing from Signature docstring'
        )


def test_bibliographic_block_becomes_a_bibliographic_node():
    entry = (
        'Kurt D. Bollacker et al. 2008. Freebase. In SIGMOD, pages 1247-1250.'
    )
    node = extractor._node_for('bibliographic', entry)
    assert node.type == 'bibliographic'
    assert node.content == entry
    assert node.type == 'bibliographic'


def test_note_block_becomes_a_note_node():
    footnote = '$^2$A *lemma* is a mathematical statement of lesser importance.'
    node = extractor._node_for('note', footnote)
    assert node.type == 'note'
    assert node.content == footnote
    assert node.type == 'note'


def test_notes_are_kept_not_discarded():
    # A note says something about the subject, so unlike furniture it stays in
    # the stream and remains eligible for the semantic chain.
    nodes = _worker(
        [
            _block('paragraph', 'body'),
            _block('note', '$^1$Named after Charles Émile Picard.'),
            _block('furniture', '42'),
        ]
    )
    assert [node.type for node in nodes] == ['paragraph', 'note']


def test_furniture_is_known_to_the_model_but_not_a_node_type():
    # furniture is a vocabulary the model needs, but it must never become an
    # ASTNode — if it ever joined _VALID_TYPES the discard would silently stop
    # working and the blocks would flow downstream.
    assert 'furniture' in extractor.DSPyModel.model_fields['type'].description
    assert '- furniture:' in extractor.Signature.__doc__
    assert 'furniture' not in extractor._VALID_TYPES


def test_unknown_type_raises():
    with pytest.raises(ValueError, match='Unknown block type'):
        extractor._node_for('footnote', 'text')


def test_furniture_never_leaves_the_stage():
    nodes = _worker(
        [
            _block('header', '## 1.2 Slope Fields'),
            _block('paragraph', 'body text'),
            _block('furniture', 'Access for free at openstax.org'),
        ]
    )
    assert [node.content for node in nodes] == [
        '## 1.2 Slope Fields',
        'body text',
    ]
    # Not merely untyped: it is gone, so no later stage has to know about it.
    assert not any(node.type == 'furniture' for node in nodes)


def test_furniture_is_matched_regardless_of_case_or_padding():
    # The type comes back as free text from the model.
    nodes = _worker([_block(' Furniture ', 'chrome'), _block('paragraph', 'a')])
    assert [node.content for node in nodes] == ['a']


def test_a_page_of_pure_furniture_yields_no_nodes():
    nodes = _worker([_block('furniture', '42'), _block('furniture', 'Logic')])
    assert nodes == []


def test_a_page_with_no_furniture_is_untouched():
    nodes = _worker(
        [
            _block('paragraph', 'a'),
            _block('math', '$$x$$'),
            _block('bibliographic', 'Polya, 1970.'),
        ]
    )
    assert [node.type for node in nodes] == [
        'paragraph',
        'math',
        'bibliographic',
    ]


def test_discarded_blocks_are_logged_for_audit(caplog):
    # The discard is unrecoverable, so the log is the only record that it
    # happened — a false positive has to be findable after the fact.
    with caplog.at_level('DEBUG', logger='kms.ingestion.extractor'):
        _worker([_block('furniture', 'Richard Hammack Book of Proof')], index=7)
    assert 'Richard Hammack Book of Proof' in caplog.text
    assert 'page 7' in caplog.text
