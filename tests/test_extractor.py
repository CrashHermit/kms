"""Extractor — the LLM block type -> ASTNode mapping, and the furniture
discard. No network/LLM."""

import asyncio

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


def test_every_prompt_type_maps_to_its_node_class():
    # The taxonomy the Signature names and the map the worker uses have to
    # agree, or a type the model is told to emit silently becomes a paragraph.
    assert extractor._TYPE_MAP == {
        'paragraph': models.ParagraphNode,
        'math': models.MathNode,
        'code': models.CodeNode,
        'list': models.ListNode,
        'table': models.TableNode,
        'image': models.ImageNode,
        'caption': models.CaptionNode,
        'header': models.HeaderNode,
        'bibliographic': models.BibliographicNode,
    }
    described = extractor.DSPyModel.model_fields['type'].description
    for node_type in extractor._TYPE_MAP:
        assert node_type in described
        assert f'- {node_type}:' in extractor.Signature.__doc__


def test_bibliographic_block_becomes_a_bibliographic_node():
    entry = (
        'Kurt D. Bollacker et al. 2008. Freebase. In SIGMOD, pages 1247-1250.'
    )
    node = extractor._node_for('bibliographic', entry)
    assert isinstance(node, models.BibliographicNode)
    assert node.content == entry
    # The graph tier derives its label and `type` property from the class name.
    assert node.kind == 'bibliographic'


def test_discarded_types_are_offered_to_the_model_but_have_no_node_class():
    # furniture is a vocabulary the model needs and a node class it must never
    # produce — if it ever gained one, the discard would silently stop working
    # and the blocks would flow downstream as nodes.
    for node_type in extractor._DISCARDED_TYPES:
        assert node_type in extractor.DSPyModel.model_fields['type'].description
        assert f'- {node_type}:' in extractor.Signature.__doc__
        assert node_type not in extractor._TYPE_MAP


def test_unknown_type_still_falls_back_to_paragraph():
    node = extractor._node_for('footnote', 'text')
    assert isinstance(node, models.ParagraphNode)


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
    assert not any(node.kind == 'furniture' for node in nodes)


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
    assert [node.kind for node in nodes] == [
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
