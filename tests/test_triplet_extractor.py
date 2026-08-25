import asyncio
from types import SimpleNamespace

from kms.construction import triplet_extractor
from kms.core import context_window, models


def _fact_module() -> triplet_extractor._FactExtractor:
    return triplet_extractor._FactExtractor.__new__(
        triplet_extractor._FactExtractor
    )


def test_fact_decode_returns_text_without_model_provenance() -> None:
    prediction = SimpleNamespace(facts=['A is related to B'])

    assert _fact_module().decode(prediction, current_nodes=[]) == [
        {'text': 'A is related to B'}
    ]


def test_fact_signature_has_no_provenance_output() -> None:
    assert 'node_ids' not in triplet_extractor._FactSignature.output_fields
def test_fact_encode_uses_document_ordered_target_context_fields():
    encoded = _fact_module().encode(
        context_before=[
            context_window.ContextNode(
                position=0, type='paragraph', content='Before'
            )
        ],
        target_node=context_window.ContextNode(
            position=4,
            type='image',
            content='A diagram',
            assets=[models.VisualAsset(path='figure.png')],
        ),
        context_after=[
            context_window.ContextNode(
                position=0, type='paragraph', content='After'
            )
        ],
    )

    assert list(encoded) == [
        'context_before',
        'target_node',
        'context_after',
    ]
    assert encoded['target_node'].node_text == 'A diagram'
    assert encoded['target_node'].local_index == 0
    assert encoded['context_before'][0].node_text == 'Before'
    assert encoded['context_after'][0].node_text == 'After'
    assert not hasattr(encoded['target_node'], 'assets')
    assert not hasattr(encoded['target_node'], 'marker')


class _FactModule:
    async def aforward(self, **kwargs):
        self.calls.append(kwargs)
        return [{'text': 'The anchor states a fact.'}]

    def __init__(self) -> None:
        self.calls = []


class _TripletModule:
    async def aforward(self, *, fact_text):
        return [
            SimpleNamespace(
                subject='anchor',
                predicate='states',
                object='fact',
                node_ids=[],
                fact_text=fact_text,
            )
        ]


def test_extract_triplets_assigns_only_anchor_provenance(monkeypatch) -> None:
    nodes = [
        SimpleNamespace(type='paragraph', content='Anchor A', assets=[]),
        SimpleNamespace(type='paragraph', content='Context B', assets=[]),
    ]
    fact_module = _FactModule()
    triplet_module = _TripletModule()
    monkeypatch.setattr(
        triplet_extractor.config.get_settings().stages.triplet,
        'backward_context_budget',
        100,
    )

    result = asyncio.run(
        triplet_extractor._extract_triplets(
            nodes,
            fact_module=fact_module,
            triplet_module=triplet_module,
            source='book.pdf',
        )
    )

    assert [call['target_node'].position for call in fact_module.calls] == [0, 0]
    assert [
        [node.content for node in call['context_after']]
        for call in fact_module.calls
    ] == [['Context B'], []]
    # evidence_positions are now positions
    assert [triplet.evidence_positions for triplet in result] == [[0], [1]]
    assert all(triplet.occurrence_uuids for triplet in result)


def test_extract_triplets_allows_an_image_anchor() -> None:
    nodes = [
        SimpleNamespace(
            type='image',
            content=None,
            assets=[models.VisualAsset(path='/tmp/figure.png')],
        )
    ]
    fact_module = _FactModule()
    triplet_module = _TripletModule()

    result = asyncio.run(
        triplet_extractor._extract_triplets(
            nodes,
            fact_module=fact_module,
            triplet_module=triplet_module,
            source='book.pdf',
        )
    )

    target = fact_module.calls[0]['target_node']
    assert target.type == 'image'
    assert target.content is None
    assert target.position == 0
    assert result[0].evidence_positions == [0]
    assert result[0].occurrence_uuids
