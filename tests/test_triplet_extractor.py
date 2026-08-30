import asyncio
from types import SimpleNamespace

from kms.construction import triplet_extractor
from kms.core import models


def _fact_module() -> triplet_extractor._FactExtractor:
    return triplet_extractor._FactExtractor.__new__(
        triplet_extractor._FactExtractor
    )


def test_fact_decode_returns_text_without_model_provenance() -> None:
    prediction = SimpleNamespace(
        facts=[models.AtomicFact(text='A is related to B')]
    )

    assert _fact_module().decode(prediction, current_nodes=[]) == [
        {'text': 'A is related to B'}
    ]


def test_fact_signature_has_no_provenance_output() -> None:
    assert 'node_ids' not in triplet_extractor._FactSignature.output_fields


def test_fact_prompt_uses_structured_request_boundary():
    prompt = triplet_extractor._FactSignature.__doc__
    assert 'supplied request' in prompt
    assert 'neighboring' in prompt
    assert '<anchor>' not in prompt
    assert 'HARD EXCLUSIONS' not in prompt
    assert 'problem specifications' in prompt
    assert 'explicitly asserts' in prompt


def test_triplet_prompt_requires_concise_source_relations():
    prompt = triplet_extractor._TripletSignature.__doc__
    assert '1–5 words' in prompt
    assert 'Never output a sentence' in prompt
    assert 'Find the gradient' in prompt
    assert 'Return [] rather than guessing' in prompt


def test_triplet_decode_discards_sentence_predicates():
    module_instance = triplet_extractor._TripletDecomposer.__new__(
        triplet_extractor._TripletDecomposer
    )
    prediction = SimpleNamespace(
        triplets=[
            triplet_extractor._TripletInput(
                subject='f',
                predicate='The function is defined by the expression',
                object='x',
                subject_kind=models.NodeKind.ENTITY,
                object_kind=models.NodeKind.ENTITY,
            ),
            triplet_extractor._TripletInput(
                subject='f',
                predicate='is defined by',
                object='x',
                subject_kind=models.NodeKind.ENTITY,
                object_kind=models.NodeKind.ENTITY,
            ),
        ]
    )
    result = module_instance.decode(prediction, fact_text='f is defined by x')
    assert [(item.subject, item.predicate, item.object) for item in result] == [
        ('f', 'is defined by', 'x')
    ]
def test_fact_encode_passes_canonical_request():
    request = models.FactExtractionInput(
        context_before=[
            models.NodeInput(index=1, node_type='paragraph', text='Before')
        ],
        target_node=models.NodeInput(
            index=1, node_type='image', text='A diagram'
        ),
        context_after=[
            models.NodeInput(index=1, node_type='paragraph', text='After')
        ],
    )
    encoded = _fact_module().encode(request)

    assert encoded == {'request': request}


class _FactModule:
    async def aforward(self, *, request):
        self.calls.append(request)
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
    assert [call.target_node.index for call in fact_module.calls] == [1, 1]
    assert [
        [node.text for node in call.context_after]
        for call in fact_module.calls
    ] == [['Context B'], []]
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
    target = fact_module.calls[0].target_node
    assert target.node_type == 'image'
    assert target.text == ''
    assert result[0].evidence_positions == [0]
    assert result[0].occurrence_uuids
def test_triplet_decode_preserves_all_endpoint_kind_combinations():
    module_instance = triplet_extractor._TripletDecomposer.__new__(
        triplet_extractor._TripletDecomposer
    )
    combinations = [
        (models.NodeKind.ENTITY, models.NodeKind.ENTITY),
        (models.NodeKind.ENTITY, models.NodeKind.EVENT),
        (models.NodeKind.EVENT, models.NodeKind.ENTITY),
        (models.NodeKind.EVENT, models.NodeKind.EVENT),
    ]
    prediction = SimpleNamespace(
        triplets=[
            triplet_extractor._TripletInput(
                subject='left',
                predicate='relates to',
                object='right',
                subject_kind=subject_kind,
                object_kind=object_kind,
            )
            for subject_kind, object_kind in combinations
        ]
    )
    result = module_instance.decode(prediction, fact_text='left relates to right')
    assert [
        (item.subject_kind, item.object_kind) for item in result
    ] == combinations
