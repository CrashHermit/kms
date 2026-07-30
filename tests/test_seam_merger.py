"""Seam merger — edge selection and healing. No network/LLM.

Covers the rule that bibliographic references are passed over when the seam's
edges are chosen, so a page ending in a footnote citation still gets its real
tail healed and no citation is ever welded onto a neighbour.
"""

import asyncio

from kms.core import models
from kms.ingestion import seam_merger


def _segment(index, nodes):
    return models.Segment(index=index, image_path='', nodes=nodes)


def _para(content):
    return models.ParagraphNode(content=content)


def _ref(content):
    return models.BibliographicNode(content=content)


def _note(content):
    return models.NoteNode(content=content)


class _Merger:
    """Stands in for the LLM: always merges, recording what it was asked.

    Returns the merged CONTENT, like ``SeamMerger.aforward`` does — the healed
    tail takes the string straight onto its ``content``.
    """

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
            (
                top_bottom_edge_node.content,
                bottom_top_edge_node.content,
                top_node_context.content,
                bottom_node_context.content,
            )
        )
        return self.merged


class _NeverMerges:
    async def aforward(self, **_kwargs):
        return None


def _merge(top, bottom, module):
    return dict(asyncio.run(seam_merger._merge_pair(module, top, bottom)))


def test_edges_skip_a_trailing_citation_and_heal_the_real_tail():
    # The footer lands after the paragraph that actually runs onto the next
    # page, so the last node is not the tail to merge.
    top = _segment(
        0,
        [
            _para('intro'),
            _para('a sentence cut off mid-'),
            _ref('Pólya, 1970.'),
        ],
    )
    bottom = _segment(1, [_para('way through it.'), _para('next')])

    module = _Merger('a sentence cut off mid-way through it.')
    result = _merge(top, bottom, module)

    # The paragraph, not the citation, was offered to the model.
    assert module.seen[0][0] == 'a sentence cut off mid-'
    assert module.seen[0][1] == 'way through it.'
    # ... and the merged content landed on it, with the citation left in place.
    assert [node.content for node in result[0]] == [
        'intro',
        'a sentence cut off mid-way through it.',
        'Pólya, 1970.',
    ]
    assert [node.content for node in result[1]] == ['next']


def test_a_leading_citation_on_the_bottom_page_is_passed_over():
    top = _segment(0, [_para('a sentence cut off mid-')])
    bottom = _segment(1, [_ref('Stein, 2009.'), _para('way through it.')])

    module = _Merger('a sentence cut off mid-way through it.')
    result = _merge(top, bottom, module)

    assert module.seen[0][1] == 'way through it.'
    # The healed head is dropped by position — the citation is not the head,
    # and it survives.
    assert [node.content for node in result[1]] == ['Stein, 2009.']


def test_a_trailing_note_also_displaces_nothing():
    # Same displacement as a citation: the appended footnote is last, but the
    # paragraph before it is what runs onto the next page.
    top = _segment(
        0,
        [
            _para('a sentence cut off mid-'),
            _note('$^2$You are reminded that "or" is not exclusive.'),
        ],
    )
    bottom = _segment(1, [_para('way through it.')])

    module = _Merger('a sentence cut off mid-way through it.')
    result = _merge(top, bottom, module)

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
    # A reference list split across pages: two distinct works, no continuation.
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

    result = _merge(top, bottom, _NeverMerges())

    assert [node.content for node in result[0]] == ['complete.', 'a footnote']
    assert [node.content for node in result[1]] == ['Also complete.']


def test_pairs_skip_a_page_with_nothing_mergeable():
    # A page of pure bibliography has no seam to heal on either side, so no
    # worker is spawned for it.
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
