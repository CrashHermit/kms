import asyncio
from types import SimpleNamespace

from kms.construction import triplet_extractor


def _fact_module() -> triplet_extractor._FactExtractor:
    return triplet_extractor._FactExtractor.__new__(
        triplet_extractor._FactExtractor
    )


def test_fact_decode_returns_text_without_model_provenance() -> None:
    prediction = SimpleNamespace(
        facts=[SimpleNamespace(text='A is related to B')]
    )

    assert _fact_module().decode(prediction, current_nodes=[]) == [
        {'text': 'A is related to B'}
    ]


def test_fact_signature_has_no_provenance_output() -> None:
    assert 'node_ids' not in triplet_extractor._FactSignature.output_fields


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
        SimpleNamespace(
            type='paragraph', content='Anchor A', image_path=None
        ),
        SimpleNamespace(
            type='paragraph', content='Context B', image_path=None
        ),
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

    # With position-based references, the marked window anchor position
    # is 0 in each call (the only target in the window)
    assert [
        next(
            node.position for node in call['current_nodes'] if node.marker == 'anchor'
        )
        for call in fact_module.calls
    ] == [0, 1]
    # node_ids are now positions
    assert [triplet.node_ids for triplet in result] == [[0], [1]]
    assert all(triplet.occurrence_uuids for triplet in result)


def test_extract_triplets_allows_an_image_anchor() -> None:
    nodes = [
        SimpleNamespace(
            type='image',
            content=None,
            image_path='/tmp/figure.png',
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

    assert (
        fact_module.calls[0]['current_nodes'][0].image_path == '/tmp/figure.png'
    )
    assert result[0].node_ids == [0]
    assert result[0].occurrence_uuids
