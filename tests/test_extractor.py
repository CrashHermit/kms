import asyncio
from types import SimpleNamespace

import pytest

from kms.construction import extractor
from kms.core import models


def _span(start, end, node_type):
    return extractor.LineSpan(start=start, end=end, type=node_type)


def _block(node_type, content='text'):
    return extractor.DSPyModel(type=node_type, content=content)


class _Module:
    def __init__(self, blocks):
        self.blocks = blocks

    async def aforward(self, segment_markdown):
        return list(self.blocks)


def _worker(blocks, index=0):
    node = extractor.ExtractorNode(module=_Module(blocks))
    segment = models.Segment(index=index, image_path='', content='markdown')
    out = asyncio.run(node.worker({'segment': segment}))
    return out['extract_results'][0][1]


def _decode(spans, markdown='heading\n\nbody'):
    module = extractor.Extractor.__new__(extractor.Extractor)
    return module.decode(
        SimpleNamespace(spans=spans), segment_markdown=markdown
    )


def test_prompt_describes_line_addressed_output():
    prompt = extractor.Signature.__doc__
    assert 'original markdown as a list of lines' in prompt
    assert 'Never' in prompt
    assert 'copy, rewrite, summarize' in prompt
    assert 'Do not create a span containing only blank lines' in prompt
    assert 'Return the ordered list of spans and nothing else.' in prompt
    assert 'COVERAGE IS MANDATORY' in prompt
    assert 'SCAN THEN CLASSIFY' in prompt
    assert 'Never skip ordinary prose' in prompt


def test_every_prompt_type_is_in_the_valid_set():
    described = extractor.LineSpan.model_fields['type'].description
    for node_type in extractor._VALID_TYPES:
        assert node_type in described, (
            f'{node_type!r} missing from field description'
        )
        assert f'`{node_type}`' in extractor.Signature.__doc__, (
            f'{node_type!r} missing from Signature docstring'
        )


def test_line_spans_reconstruct_verbatim_source_blocks():
    blocks = _decode(
        [
            _span(1, 1, 'header'),
            _span(2, 2, 'paragraph'),
            _span(3, 3, 'math'),
        ],
        markdown='## Heading\nbody\n$$x$$',
    )

    assert [(block.type, block.content) for block in blocks] == [
        ('header', '## Heading'),
        ('paragraph', 'body'),
        ('math', '$$x$$'),
    ]


def test_line_spans_omit_blank_separator_lines():
    blocks = _decode(
        [_span(1, 1, 'paragraph'), _span(3, 3, 'header')],
        markdown='first\n\nsecond',
    )

    assert [(block.type, block.content) for block in blocks] == [
        ('paragraph', 'first'),
        ('header', 'second'),
    ]


@pytest.mark.parametrize(
    ('spans', 'markdown', 'message'),
    [
        ([], 'text', 'no line spans'),
        ([_span(2, 2, 'paragraph')], 'one\ntwo', 'nonblank source lines'),
        (
            [_span(1, 2, 'paragraph'), _span(2, 2, 'paragraph')],
            'a\nb',
            'overlaps',
        ),
        ([_span(1, 3, 'paragraph')], 'a\nb', 'exceeds source line count'),
        ([_span(1, 1, 'paragraph')], '\n', 'only blank lines'),
        ([_span(1, 1, 'paragraph')], 'a\nb', 'nonblank source lines'),
    ],
)
def test_line_spans_fail_fast_on_invalid_partition(spans, markdown, message):
    with pytest.raises(ValueError, match=message):
        _decode(spans, markdown=markdown)


def test_bibliographic_block_becomes_a_bibliographic_node():
    entry = (
        'Kurt D. Bollacker et al. 2008. Freebase. In SIGMOD, pages 1247-1250.'
    )
    node = extractor._node_for('bibliographic', entry)
    assert node.type == 'bibliographic'
    assert node.content == entry


def test_note_block_becomes_a_note_node():
    footnote = '$^2$A *lemma* is a mathematical statement of lesser importance.'
    node = extractor._node_for('note', footnote)
    assert node.type == 'note'
    assert node.content == footnote


def test_notes_are_kept_not_discarded():
    nodes = _worker(
        [
            _block('paragraph', 'body'),
            _block('note', '$^1$Named after Charles Émile Picard.'),
            _block('furniture', '42'),
        ]
    )
    assert [node.type for node in nodes] == ['paragraph', 'note']


def test_furniture_is_known_to_the_model_but_not_a_node_type():
    assert 'furniture' in extractor.LineSpan.model_fields['type'].description
    assert '`furniture`' in extractor.Signature.__doc__
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
    assert not any(node.type == 'furniture' for node in nodes)


def test_furniture_is_matched_regardless_of_case_or_padding():
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


def test_image_markers_are_removed_from_image_nodes():
    node = extractor._node_for('image', '* a) ![1]()')
    assert node.content == '![1]()'


def test_discarded_blocks_are_logged_for_audit(caplog):
    with caplog.at_level('DEBUG', logger='kms.construction.extractor'):
        _worker([_block('furniture', 'Richard Hammack Book of Proof')], index=7)
    assert 'Richard Hammack Book of Proof' in caplog.text
    assert 'page 7' in caplog.text
