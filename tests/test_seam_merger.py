import asyncio

from kms.construction import seam_merger
from kms.core import models


def _segment(index, nodes):
    return models.Segment(index=index, image_path='', nodes=nodes)


def _para(content):
    return models.ASTNode(type='paragraph', content=content)


def _ref(content):
    return models.ASTNode(type='bibliographic', content=content)


def _note(content):
    return models.ASTNode(type='note', content=content)


def _shown(
    top_bottom_edge_node,
    bottom_top_edge_node,
    top_node_context,
    bottom_node_context,
):
    return (
        top_bottom_edge_node.content,
        bottom_top_edge_node.content,
        top_node_context.content,
        bottom_node_context.content,
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
            seam_merger._merge_pair(
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
    segments = [
        _segment(0, [_para('body')]),
        _segment(1, [_ref('one'), _ref('two')]),
        _segment(2, [_para('body')]),
    ]
    assert seam_merger._pairs(segments, parity=0) == []
    assert seam_merger._pairs(segments, parity=1) == []


def test_pairs_still_fan_out_over_ordinary_neighbours():
    segments = [
        _segment(0, [_para('a')]),
        _segment(1, [_para('b')]),
        _segment(2, [_para('c')]),
    ]
    assert [
        (top.index, bottom.index)
        for top, bottom in seam_merger._pairs(segments, parity=0)
    ] == [(0, 1)]
    assert [
        (top.index, bottom.index)
        for top, bottom in seam_merger._pairs(segments, parity=1)
    ] == [(1, 2)]
