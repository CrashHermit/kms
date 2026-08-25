import asyncio

from kms.construction import text_seam_merger
from kms.core import models


def _segment(index, nodes):
    return models.Document(index=index, image_path='', nodes=nodes)


def _para(content):
    return models.SourceNode(type='paragraph', content=content)


def _ref(content):
    return models.SourceNode(type='bibliographic', content=content)


def _note(content):
    return models.SourceNode(type='note', content=content)


def test_seam_encoder_uses_plain_text_fields():
    node = models.SourceNode(type='paragraph', content='tail text')

    encoded = text_seam_merger.TextSeamMerger.encode(object(), node, node)

    assert encoded == {
        'top_node_context': '',
        'top_bottom_edge_node': 'tail text',
        'bottom_top_edge_node': 'tail text',
        'bottom_node_context': '',
    }


def _shown(
    top_bottom_edge_node,
    bottom_top_edge_node,
    top_node_context,
    bottom_node_context,
):
    def text(node):
        if node is None:
            return ''
        return getattr(node, 'content', node)

    return (
        text(top_bottom_edge_node),
        text(bottom_top_edge_node),
        text(top_node_context),
        text(bottom_node_context),
    )


class _Merger:
    def __init__(self):
        self.seen = []

    async def aforward(
        self,
        top_bottom_edge_node,
        bottom_top_edge_node,
        top_node_context=None,
        bottom_node_context=None,
    ):
        self.seen.append(
            _shown(
                top_bottom_edge_node,
                bottom_top_edge_node,
                top_node_context,
                bottom_node_context,
            )
        )
        return True


class _NeverMerges:
    async def aforward(self, **_kwargs):
        return False


class _Rewriter:
    def __init__(self, merged='MERGED'):
        self.merged = merged
        self.seen = []

    async def aforward(
        self,
        top_bottom_edge_node,
        bottom_top_edge_node,
        top_node_context=None,
        bottom_node_context=None,
    ):
        self.seen.append(
            _shown(
                top_bottom_edge_node,
                bottom_top_edge_node,
                top_node_context,
                bottom_node_context,
            )
        )
        return self.merged


class _NeverRewrites:
    async def aforward(self, **_kwargs):
        raise AssertionError('the rewriter ran on a seam the judge declined')


def _merge(top, bottom, module, rewriter=None):
    return dict(
        asyncio.run(
            text_seam_merger._merge_pair(
                module, rewriter or _Rewriter(), top, bottom
            )
        )
    )


def test_edges_skip_a_trailing_citation_and_heal_the_real_tail():
    top = _segment(
        0,
        [
            _para('intro'),
            _para('a sentence cut off mid-'),
            _ref('Pólya, 1970.'),
        ],
    )
    bottom = _segment(1, [_para('way through it.'), _para('next')])

    module = _Merger()
    rewriter = _Rewriter('a sentence cut off mid-way through it.')
    result = _merge(top, bottom, module, rewriter)
    assert module.seen[0][0] == 'a sentence cut off mid-'
    assert module.seen[0][1] == 'way through it.'
    assert [node.content for node in result[0]] == [
        'intro',
        'a sentence cut off mid-way through it.',
        'Pólya, 1970.',
    ]
    assert [node.content for node in result[1]] == ['next']


def test_a_leading_citation_on_the_bottom_page_is_passed_over():
    top = _segment(0, [_para('a sentence cut off mid-')])
    bottom = _segment(1, [_ref('Stein, 2009.'), _para('way through it.')])

    module = _Merger()
    result = _merge(top, bottom, module)

    assert module.seen[0][1] == 'way through it.'
    assert [node.content for node in result[1]] == ['Stein, 2009.']


def test_a_trailing_note_also_displaces_nothing():
    top = _segment(
        0,
        [
            _para('a sentence cut off mid-'),
            _note('$^2$You are reminded that "or" is not exclusive.'),
        ],
    )
    bottom = _segment(1, [_para('way through it.')])

    module = _Merger()
    rewriter = _Rewriter('a sentence cut off mid-way through it.')
    result = _merge(top, bottom, module, rewriter)

    assert module.seen[0][0] == 'a sentence cut off mid-'
    assert [node.content for node in result[0]] == [
        'a sentence cut off mid-way through it.',
        '$^2$You are reminded that "or" is not exclusive.',
    ]
    assert result[1] == []


def test_a_seam_between_two_notes_is_never_judged():
    top = _segment(0, [_note('$^1$One note.')])
    bottom = _segment(1, [_note('$^2$A different note.')])

    module = _Merger()
    result = _merge(top, bottom, module)

    assert module.seen == []
    assert [node.content for node in result[0]] == ['$^1$One note.']
    assert [node.content for node in result[1]] == ['$^2$A different note.']


def test_context_nodes_also_skip_citations():
    top = _segment(
        0, [_para('context above'), _ref('a footnote'), _para('tail')]
    )
    bottom = _segment(
        1, [_para('head'), _ref('another footnote'), _para('context below')]
    )

    module = _Merger()
    _merge(top, bottom, module)

    _, _, top_context, bottom_context = module.seen[0]
    assert top_context == 'context above'
    assert bottom_context == 'context below'


def test_image_at_top_edge_is_skipped_for_text_seam_dispatch():
    documents = [
        _segment(
            0,
            [
                _para('text before image'),
                models.SourceNode(
                    type='image',
                    assets=[models.VisualAsset(path='bottom.png')],
                ),
            ],
        ),
        _segment(1, [_para('next-page text')]),
    ]

    pairs = text_seam_merger._pairs(documents, parity=0)

    assert [(top.index, bottom.index) for top, bottom in pairs] == [(0, 1)]


def test_image_at_bottom_edge_is_skipped_for_text_seam_dispatch():
    documents = [
        _segment(0, [_para('previous-page text')]),
        _segment(
            1,
            [
                models.SourceNode(
                    type='image',
                    assets=[models.VisualAsset(path='top.png')],
                ),
                _para('text after image'),
            ],
        ),
    ]

    pairs = text_seam_merger._pairs(documents, parity=0)

    assert [(top.index, bottom.index) for top, bottom in pairs] == [(0, 1)]


def test_image_before_tail_is_skipped_for_farther_context():
    top = _segment(
        0,
        [
            _para('farther text'),
            models.SourceNode(
                type='image',
                assets=[models.VisualAsset(path='before-tail.png')],
            ),
            _para('tail'),
        ],
    )
    bottom = _segment(1, [_para('head')])
    merger = _Merger()

    _merge(top, bottom, merger)

    assert merger.seen[0][2] == 'farther text'


def test_image_after_head_is_skipped_for_farther_context():
    top = _segment(0, [_para('tail')])
    bottom = _segment(
        1,
        [
            _para('head'),
            models.SourceNode(
                type='image',
                assets=[models.VisualAsset(path='after-head.png')],
            ),
            _para('farther text'),
        ],
    )
    merger = _Merger()

    _merge(top, bottom, merger)

    assert merger.seen[0][3] == 'farther text'


def test_a_seam_between_two_citations_is_never_judged():
    top = _segment(0, [_ref('Agirre et al. 2000.')])
    bottom = _segment(1, [_ref('Bollacker et al. 2008.')])

    module = _Merger()
    result = _merge(top, bottom, module)

    assert module.seen == []
    assert [node.content for node in result[0]] == ['Agirre et al. 2000.']
    assert [node.content for node in result[1]] == ['Bollacker et al. 2008.']


def test_unhealed_seam_leaves_both_pages_untouched():
    top = _segment(0, [_para('complete.'), _ref('a footnote')])
    bottom = _segment(1, [_para('Also complete.')])

    result = _merge(top, bottom, _NeverMerges(), _NeverRewrites())

    assert [node.content for node in result[0]] == ['complete.', 'a footnote']
    assert [node.content for node in result[1]] == ['Also complete.']


def test_a_declined_seam_never_overwrites_the_tail():
    top = _segment(0, [_para('**965.** (4² + 5²)²')])
    bottom = _segment(1, [_para('**966.** Simplify: $3(x+2)$')])

    result = _merge(top, bottom, _NeverMerges(), _NeverRewrites())

    assert [node.content for node in result[0]] == ['**965.** (4² + 5²)²']
    assert [node.content for node in result[1]] == [
        '**966.** Simplify: $3(x+2)$'
    ]


def test_a_healed_seam_takes_the_rewriter_s_text():
    top = _segment(0, [_para('a sentence cut off mid-')])
    bottom = _segment(1, [_para('way through it.')])

    rewriter = _Rewriter('a sentence cut off mid-way through it.')
    result = _merge(top, bottom, _Merger(), rewriter)

    assert [node.content for node in result[0]] == [
        'a sentence cut off mid-way through it.'
    ]
    assert result[1] == []


def test_an_image_seam_is_left_separate(tmp_path):
    top_path = tmp_path / 'top.png'
    bottom_path = tmp_path / 'bottom.png'
    top = _segment(
        0,
        [
            models.SourceNode(
                type='image',
                assets=[models.VisualAsset(path=str(top_path))],
            )
        ],
    )
    bottom = _segment(
        1,
        [
            models.SourceNode(
                type='image',
                assets=[models.VisualAsset(path=str(bottom_path))],
            )
        ],
    )

    merger = _Merger()
    result = _merge(top, bottom, merger, _NeverRewrites())

    assert merger.seen == []
    assert [asset.path for asset in result[0][0].assets] == [str(top_path)]
    assert [asset.path for asset in result[1][0].assets] == [str(bottom_path)]


def test_an_asset_bearing_text_seam_is_left_separate():
    asset = models.VisualAsset(path='figure.png')
    top = _segment(
        0,
        [
            models.SourceNode(
                type='paragraph', content='caption', assets=[asset]
            )
        ],
    )
    bottom = _segment(
        1,
        [
            models.SourceNode(
                type='paragraph',
                content='continuation',
                assets=[asset],
            )
        ],
    )

    merger = _Merger()
    rewriter = _Rewriter('should not be used')
    result = _merge(top, bottom, merger, rewriter)

    assert merger.seen == []
    assert rewriter.seen == []
    assert result[0][0].content == 'caption'
    assert result[1][0].content == 'continuation'


def test_the_rewriter_sees_the_same_nodes_as_the_judge():
    top = _segment(0, [_para('context above'), _para('tail')])
    bottom = _segment(1, [_para('head'), _para('context below')])

    module, rewriter = _Merger(), _Rewriter()
    _merge(top, bottom, module, rewriter)

    assert rewriter.seen == module.seen
    assert rewriter.seen[0] == (
        'tail',
        'head',
        'context above',
        'context below',
    )


def test_pairs_skip_a_page_with_nothing_mergeable():
    documents = [
        _segment(0, [_para('body')]),
        _segment(1, [_ref('one'), _ref('two')]),
        _segment(2, [_para('body')]),
    ]
    assert text_seam_merger._pairs(documents, parity=0) == []
    assert text_seam_merger._pairs(documents, parity=1) == []


def test_pairs_still_fan_out_over_ordinary_neighbours():
    documents = [
        _segment(0, [_para('a')]),
        _segment(1, [_para('b')]),
        _segment(2, [_para('c')]),
    ]
    assert [
        (top.index, bottom.index)
        for top, bottom in text_seam_merger._pairs(documents, parity=0)
    ] == [(0, 1)]
    assert [
        (top.index, bottom.index)
        for top, bottom in text_seam_merger._pairs(documents, parity=1)
    ] == [(1, 2)]
