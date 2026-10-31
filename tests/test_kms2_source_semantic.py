import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kms2.config.inference import ContextWindowSettings
from kms2.core.model.block import SourceBlock
from kms2.core.model.context import PositionedSourceBlock
from kms2.core.model.source_extraction import (
    SourceEvidenceInput,
    SourceExtractionContext,
    SourceExtractionInput,
    SourceExtractionOccurrence,
    SourceExtractionRequest,
    SourceExtractionTarget,
)
from kms2.core.model.source_semantic.source_entity_extraction import (
    SourceEntityCandidate,
    SourceEntityExtractionResult,
)
from kms2.core.model.source_semantic.source_proposition_extraction import (
    SourcePropositionCandidate,
    SourcePropositionExtractionResult,
)
from kms2.node.source_extraction import (
    build_source_extraction_requests,
)
from kms2.node.source_semantic.source_entity_extraction import (
    SourceEntityExtractionNode,
)
from kms2.node.source_semantic.source_proposition_extraction import (
    SourcePropositionExtractionNode,
)


class _TokenCounter:
    def count_texts(self, texts):
        return [int(bool(text)) for text in texts]


_TOKEN_COUNTERS = (_TokenCounter(),)
_WINDOW = ContextWindowSettings(
    backward_budget=2, forward_budget=2, target_budget=1
)


def _block(uuid: str, content: str) -> SourceBlock:
    return SourceBlock(uuid=uuid, block_type='paragraph', content=content)


def test_extraction_requests_project_context_without_graph_identities():
    blocks = [_block('before-uuid', 'before'), _block('target-uuid', 'target')]

    request = build_source_extraction_requests(
        blocks,
        source_uuid='source-uuid',
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )[1]
    model_input = request.model_input()

    assert [item.content for item in model_input.context_before] == ['before']
    assert [
        (item.position, item.source_block.content)
        for item in model_input.target_blocks
    ] == [(0, 'target')]
    assert 'uuid' not in model_input.model_dump_json()
    assert 'source-uuid' not in model_input.model_dump_json()


def test_empty_source_builds_no_extraction_requests():
    assert (
        build_source_extraction_requests(
            [],
            source_uuid='source-uuid',
            context_window=_WINDOW,
            token_counters=_TOKEN_COUNTERS,
        )
        == []
    )


def test_entity_and_proposition_collectors_are_independent_and_ordered():
    blocks = [_block('b0', 'first'), _block('b1', 'second')]
    requests = build_source_extraction_requests(
        blocks,
        source_uuid='source',
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )
    proposition_node = SourcePropositionExtractionNode(
        extractor=None, context_window=_WINDOW, token_counters=_TOKEN_COUNTERS
    )
    proposition_state = SimpleNamespace(
        blocks=blocks,
        proposition_results=[
            SourcePropositionExtractionResult(
                request=requests[1],
                propositions=[
                    SourcePropositionCandidate(
                        text='second claim', source_position=0
                    )
                ],
            ),
            SourcePropositionExtractionResult(
                request=requests[0],
                propositions=[
                    SourcePropositionCandidate(
                        text='first claim', source_position=0
                    )
                ],
            ),
        ],
    )
    propositions = proposition_node.collect(proposition_state)[
        'source_propositions'
    ]
    entity_node = SourceEntityExtractionNode(
        extractor=None,
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )
    entity_state = SimpleNamespace(
        blocks=blocks,
        source_entity_extraction_results=[
            SourceEntityExtractionResult(
                request=requests[1],
                entities=[
                    SourceEntityCandidate(
                        name='Second concept', source_position=0
                    )
                ],
            ),
            SourceEntityExtractionResult(
                request=requests[0],
                entities=[
                    SourceEntityCandidate(
                        name='First concept', source_position=0
                    )
                ],
            ),
        ],
    )
    entities = entity_node.collect(entity_state)['source_entities']

    assert [item.text for item in propositions] == [
        'first claim',
        'second claim',
    ]
    assert [item.name for item in entities] == [
        'First concept',
        'Second concept',
    ]


def test_empty_output_on_one_stream_does_not_change_other_stream_results():
    block = _block('block', 'substantive heading')
    request = build_source_extraction_requests(
        [block],
        source_uuid='source',
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )[0]
    proposition_node = SourcePropositionExtractionNode(
        extractor=None, context_window=_WINDOW, token_counters=_TOKEN_COUNTERS
    )
    entity_node = SourceEntityExtractionNode(
        extractor=None,
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )
    proposition_state = SimpleNamespace(
        blocks=[block],
        proposition_results=[
            SourcePropositionExtractionResult(request=request, propositions=[])
        ],
    )
    entity_state = SimpleNamespace(
        blocks=[block],
        source_entity_extraction_results=[
            SourceEntityExtractionResult(
                request=request,
                entities=[
                    SourceEntityCandidate(
                        name='Named inequality', source_position=0
                    )
                ],
            )
        ],
    )

    assert (
        proposition_node.collect(proposition_state)['source_propositions'] == []
    )
    assert entity_node.collect(entity_state)['source_entities'][0].name == (
        'Named inequality'
    )


def test_each_extracted_occurrence_retains_its_owning_block():
    block = _block('block', 'one block supports two mentions')
    request = SourceExtractionRequest(
        source_uuid='source',
        target=SourceExtractionTarget(source_blocks=[block]),
        context_before=SourceExtractionContext(),
        context_after=SourceExtractionContext(),
    )
    result = SourceEntityExtractionResult(
        request=request,
        entities=[
            SourceEntityCandidate(name='First', source_position=0),
            SourceEntityCandidate(name='Second', source_position=0),
        ],
    )
    node = SourceEntityExtractionNode(
        extractor=None,
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )

    entities = node.collect(
        SimpleNamespace(
            blocks=[block],
            source_entity_extraction_results=[result],
        )
    )['source_entities']

    assert [entity.source_uuid for entity in entities] == ['source', 'source']
    assert entities[0].target.source_blocks[0].uuid == 'block'
    assert entities[1].target.source_blocks[0].uuid == 'block'


@pytest.mark.parametrize(
    ('node_type', 'candidate', 'request_key', 'result_key', 'output_field'),
    [
        (
            SourcePropositionExtractionNode,
            SourcePropositionCandidate(
                text='A complete claim', source_position=0
            ),
            'proposition_extraction_request',
            'proposition_results',
            'propositions',
        ),
        (
            SourceEntityExtractionNode,
            SourceEntityCandidate(name='Target concept', source_position=0),
            'entity_extraction_request',
            'source_entity_extraction_results',
            'entities',
        ),
    ],
)
def test_worker_uses_only_uuid_free_model_projection(
    node_type, candidate, request_key, result_key, output_field
):
    class Extractor:
        async def acall(self, *, request):
            assert (
                request.target_blocks[0].source_block.content == 'target text'
            )
            assert 'uuid' not in request.model_dump_json()
            return [candidate]

    block = _block('block', 'target text')
    request = build_source_extraction_requests(
        [block],
        source_uuid='source',
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )[0]
    node = node_type(Extractor(), _WINDOW, token_counters=_TOKEN_COUNTERS)
    output = asyncio.run(node.worker({request_key: request}))
    result = output[result_key][0]
    assert result.request.target.source_blocks == [block]
    assert getattr(result, output_field) == [candidate]


def test_each_extraction_stream_has_an_independent_empty_route():
    state = SimpleNamespace(source_uuid='source', blocks=[])
    proposition_node = SourcePropositionExtractionNode(
        extractor=None, context_window=_WINDOW, token_counters=_TOKEN_COUNTERS
    )
    entity_node = SourceEntityExtractionNode(
        extractor=None,
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )
    assert proposition_node.dispatch_propositions(state) == (
        'source_proposition_collect'
    )
    assert entity_node.dispatch_entities(state) == (
        'source_entity_extraction_collect'
    )


def test_chunk_requests_cover_source_once_and_reset_target_positions():
    class WeightedCounter:
        def count_texts(self, texts: list[str]) -> list[int]:
            costs = {
                'heading-a': 0,
                'heading-b': 0,
                'small-a': 1,
                'small-b': 1,
                'oversized': 5,
                'small-c': 1,
            }
            return [costs[text] for text in texts]

    blocks = [
        _block('a', 'heading-a'),
        _block('b', 'heading-b'),
        _block('c', 'small-a'),
        _block('d', 'small-b'),
        _block('e', 'oversized'),
        _block('f', 'small-c'),
    ]
    requests = build_source_extraction_requests(
        blocks,
        source_uuid='source',
        context_window=ContextWindowSettings(
            backward_budget=0, forward_budget=0, target_budget=2
        ),
        token_counters=(WeightedCounter(),),
    )

    assert [
        [block.uuid for block in request.target.source_blocks]
        for request in requests
    ] == [['a', 'b', 'c', 'd'], ['e'], ['f']]
    assert [
        block.uuid
        for request in requests
        for block in request.target.source_blocks
    ] == [block.uuid for block in blocks]
    for request in requests:
        model_input = request.model_input()
        assert [item.position for item in model_input.target_blocks] == list(
            range(len(request.target.source_blocks))
        )


def test_collectors_order_by_owner_without_persisting_sibling_context():
    before = _block('before', 'before')
    a = _block('a', 'a')
    b = _block('b', 'b')
    c = _block('c', 'c')
    d = _block('d', 'd')
    after = _block('after', 'after')
    blocks = [a, b, c, d]
    first_request = SourceExtractionRequest(
        source_uuid='source',
        target=SourceExtractionTarget(source_blocks=[a, b]),
        context_before=SourceExtractionContext(source_blocks=[before]),
        context_after=SourceExtractionContext(source_blocks=[after]),
    )
    second_request = SourceExtractionRequest(
        source_uuid='source',
        target=SourceExtractionTarget(source_blocks=[c, d]),
        context_before=SourceExtractionContext(source_blocks=[a, b]),
        context_after=SourceExtractionContext(source_blocks=[after]),
    )
    entity_node = SourceEntityExtractionNode(
        extractor=None,
        context_window=_WINDOW,
        token_counters=_TOKEN_COUNTERS,
    )
    entity_results = [
        SourceEntityExtractionResult(
            request=second_request,
            entities=[
                SourceEntityCandidate(name='D', source_position=1),
                SourceEntityCandidate(name='Same', source_position=0),
            ],
        ),
        SourceEntityExtractionResult(
            request=first_request,
            entities=[
                SourceEntityCandidate(name='Same', source_position=1),
                SourceEntityCandidate(name='A first', source_position=0),
                SourceEntityCandidate(name='A second', source_position=0),
            ],
        ),
    ]
    entities = entity_node.collect(
        SimpleNamespace(
            blocks=blocks,
            source_entity_extraction_results=entity_results,
        )
    )['source_entities']

    proposition_node = SourcePropositionExtractionNode(
        extractor=None, context_window=_WINDOW, token_counters=_TOKEN_COUNTERS
    )
    proposition_results = [
        SourcePropositionExtractionResult(
            request=second_request,
            propositions=[
                SourcePropositionCandidate(text='same', source_position=1),
                SourcePropositionCandidate(text='same', source_position=0),
            ],
        ),
        SourcePropositionExtractionResult(
            request=first_request,
            propositions=[
                SourcePropositionCandidate(text='same', source_position=1),
                SourcePropositionCandidate(text='same', source_position=0),
            ],
        ),
    ]
    propositions = proposition_node.collect(
        SimpleNamespace(blocks=blocks, proposition_results=proposition_results)
    )['source_propositions']

    assert [item.name for item in entities] == [
        'A first',
        'A second',
        'Same',
        'Same',
        'D',
    ]
    assert [item.text for item in propositions] == ['same'] * 4
    assert [item.target.source_blocks[0].uuid for item in entities] == [
        'a',
        'a',
        'b',
        'c',
        'd',
    ]
    assert [item.target.source_blocks[0].uuid for item in propositions] == [
        'a',
        'b',
        'c',
        'd',
    ]
    assert all(
        'context_before' not in item.model_dump()
        and 'context_after' not in item.model_dump()
        for item in [*entities, *propositions]
    )


@pytest.mark.parametrize(
    'positions',
    [
        [1],
        [0, 2],
    ],
)
def test_extraction_input_rejects_noncontiguous_local_positions(positions):
    source = _block('source-block', 'source text')
    base_input = SourceExtractionRequest(
        source_uuid='source',
        target=SourceExtractionTarget(source_blocks=[source]),
        context_before=SourceExtractionContext(),
        context_after=SourceExtractionContext(),
    ).model_input()
    target_blocks = [
        PositionedSourceBlock(
            position=position,
            source_block=base_input.target_blocks[0].source_block,
        )
        for position in positions
    ]

    with pytest.raises(ValidationError):
        SourceExtractionInput(target_blocks=target_blocks)


def test_candidates_require_nonnegative_source_positions_and_result_bounds():
    with pytest.raises(ValidationError):
        SourceEntityCandidate(name='Missing position')
    with pytest.raises(ValidationError):
        SourcePropositionCandidate(text='Missing position')
    with pytest.raises(ValidationError):
        SourceEntityCandidate(name='Negative position', source_position=-1)
    with pytest.raises(ValidationError):
        SourcePropositionCandidate(text='Negative position', source_position=-1)

    block = _block('only', 'one target')
    request = SourceExtractionRequest(
        source_uuid='source',
        target=SourceExtractionTarget(source_blocks=[block]),
        context_before=SourceExtractionContext(),
        context_after=SourceExtractionContext(),
    )
    with pytest.raises(ValidationError):
        SourceEntityExtractionResult(
            request=request,
            entities=[SourceEntityCandidate(name='Outside', source_position=1)],
        )
    with pytest.raises(ValidationError):
        SourcePropositionExtractionResult(
            request=request,
            propositions=[
                SourcePropositionCandidate(text='Outside', source_position=1)
            ],
        )


def test_occurrences_and_evidence_inputs_reject_multiple_owners():
    blocks = [_block('a', 'first'), _block('b', 'second')]
    with pytest.raises(ValidationError):
        SourceExtractionOccurrence(
            source_uuid='source',
            target=SourceExtractionTarget(source_blocks=blocks),
        )
    with pytest.raises(ValidationError):
        SourceEvidenceInput(
            target_blocks=[
                SourceExtractionRequest(
                    source_uuid='source',
                    target=SourceExtractionTarget(source_blocks=[block]),
                    context_before=SourceExtractionContext(),
                    context_after=SourceExtractionContext(),
                )
                .model_input()
                .target_blocks[0]
                .source_block
                for block in blocks
            ]
        )
