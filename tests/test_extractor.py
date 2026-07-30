"""Extractor — the LLM block type -> ASTNode mapping. No network/LLM."""

from kms.core import models
from kms.ingestion import extractor


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


def test_unknown_type_still_falls_back_to_paragraph():
    node = extractor._node_for('footnote', 'text')
    assert isinstance(node, models.ParagraphNode)
