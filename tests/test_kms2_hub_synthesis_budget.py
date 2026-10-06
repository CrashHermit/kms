"""Bounded prompts, ordered transport, and typed hub reduction behavior."""

import asyncio
import json
from importlib import import_module

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from kms2_hub_cases import HUB_CASES, build_hub_fixture

from kms2.core.windowing import InputBudgetExceeded, TokenBudget, pack_items
from kms2.langgraph.global_semantic.state import GlobalSemanticState
from kms2.langgraph.source_semantic.state import SourceSemanticState


class _CaptureEmbedding:
    def __init__(self):
        self.inputs = []

    async def embed(self, texts):
        self.inputs = texts
        return [[0.25] for _ in texts]


def test_ordered_next_fit_packing_preserves_exact_boundary_items():
    assert pack_items([], token_counts=[], token_budget=5) == []
    assert pack_items(
        [3, 2, 4, 1], token_counts=[3, 2, 4, 1], token_budget=5
    ) == [[3, 2], [4, 1]]
    assert pack_items([5], token_counts=[5], token_budget=5) == [[5]]
    for items, costs in [([6], [6]), ([3, 6], [3, 6])]:
        with pytest.raises(InputBudgetExceeded, match='singleton'):
            pack_items(items, token_counts=costs, token_budget=5)


class _CostCounter:
    """Return deterministic serialized-record costs for reduction tests."""

    def __init__(self, kind):
        self.kind = kind

    def count_texts(self, texts):
        costs = []
        for text in texts:
            if self.kind == 'final':
                payload = json.loads(text)
                costs.append(
                    len(payload['summaries']) if 'summaries' in payload else 10
                )
            elif self.kind == 'summary':
                costs.append(11 if len(text) > 1000 else 10)
            else:
                costs.append(1)
        return costs


class _Predictor:
    """Typed async predictor boundary for deterministic multilevel coverage."""

    def __init__(self, field, output_type, kind):
        self.field = field
        self.output_type = output_type
        self.calls = []
        self.outputs = []
        self.kind = kind

    async def acall(self, *, request):
        self.calls.append(request)
        if self.field == 'summary':
            value = self.output_type(text=f'{self.kind}-{len(self.calls)}')
        else:
            value = self.output_type(
                **{
                    field: (
                        'Generated canonical definition'
                        if field in {'canonical_name', 'name'}
                        else 'Generated description'
                    )
                    for field in self.output_type.model_fields
                }
            )
        self.outputs.append(value)
        return dspy.Prediction(**{self.field: value})

    def __call__(self, *, request):
        raise AssertionError('Reduction workers must use async module calls')


def _module_parts(fixture, *, direct=False):
    """Create concrete modules around real typed signatures and test boundaries."""
    module_path = f'kms2.module.{fixture.scope}.{_snake_case(fixture.stem)}'
    module_api = import_module(module_path)
    model = fixture.model_module
    summary_type = getattr(model, f'{fixture.stem}Summary')
    definition_type = getattr(model, f'{fixture.stem}Definition')
    summary_module_type = getattr(module_api, f'{fixture.stem}SummaryModule')
    merge_module_type = getattr(module_api, f'{fixture.stem}SummaryMergeModule')
    final_module_type = getattr(module_api, f'{fixture.stem}Module')
    summary_predictor = _Predictor('summary', summary_type, 'summary')
    merge_predictor = _Predictor('summary', summary_type, 'merge')
    final_predictor = _Predictor('definition', definition_type, 'final')
    summary_module = summary_module_type(summary_predictor)
    merge_module = merge_module_type(merge_predictor)
    final_module = final_module_type(final_predictor)
    budget = {
        'final': TokenBudget(_CostCounter('final'), 100000 if direct else 2),
        'summary': TokenBudget(_CostCounter('summary'), 10),
        'merge': TokenBudget(_CostCounter('merge'), 2),
    }
    config_module = import_module(f'kms2.config.{fixture.scope}')
    settings_type = getattr(config_module, f'{fixture.stem}Settings')
    node = object.__new__(fixture.node_class)
    node._module = final_module
    node._settings = settings_type()
    node._summary_module = summary_module
    node._merge_module = merge_module
    node._final_budget = budget['final']
    node._summary_budget = budget['summary']
    node._merge_budget = budget['merge']
    return (
        node,
        (summary_predictor, merge_predictor, final_predictor),
        budget,
    )


def _snake_case(value):
    return ''.join(
        ('_' + character.lower()) if character.isupper() else character
        for character in value
    ).lstrip('_')


def _graph_state(stem, values):
    if stem.startswith('Source'):
        return SourceSemanticState(source_uuid='source-owner', **values)
    return GlobalSemanticState(**values)


@pytest.mark.parametrize(
    'stem', [case[0] for case in HUB_CASES], ids=[case[0] for case in HUB_CASES]
)
def test_typed_hub_direct_fit_uses_original_evidence_and_backend_metadata(stem):
    async def exercise():
        fixture = build_hub_fixture(stem, count=2)
        node, predictors, budgets = _module_parts(fixture, direct=True)
        module_api = import_module(
            f'kms2.module.{fixture.scope}.{_snake_case(stem)}'
        )
        definition = {
            field: (
                'Direct typed definition'
                if field in {'canonical_name', 'name'}
                else 'Original evidence synthesized once'
            )
            for field in getattr(
                fixture.model_module, f'{stem}Definition'
            ).model_fields
        }
        adapter = dspy.ChatAdapter(use_json_adapter_fallback=False)
        predictor = dspy.Predict(getattr(module_api, f'{stem}Signature'))
        lm = DummyLM([{'definition': definition}], adapter=adapter)
        predictor.set_lm(lm)
        node._module = getattr(module_api, f'{stem}Module')(predictor)
        with dspy.context(adapter=adapter):
            result_payload = await node.synthesis_worker(fixture.state)
        result_key = f'{_snake_case(stem)}_synthesis_results'
        [result] = result_payload[result_key]
        assert isinstance(
            result.definition,
            getattr(fixture.model_module, f'{stem}Definition'),
        )
        assert len(predictors[2].calls) == 0
        assert len(lm.history) == 1
        assert result.ordinal == 7
        if 'membership_uuids' in type(result).model_fields:
            assert result.membership_uuids == fixture.backend_uuids
        if 'aliases' in type(result).model_fields and fixture.expected_aliases:
            assert result.aliases == fixture.expected_aliases
        if stem.endswith('TripletHub'):
            group = fixture.backend_items[0]
            assert result.definition.model_dump()
            assert [
                group.subject_hub_uuid,
                group.predicate_hub_uuid,
                group.object_hub_uuid,
            ] == fixture.backend_uuids[:3]

        if stem == 'SourceEntityHub':
            embed_client = _CaptureEmbedding()
            node._embedding_client = embed_client
            embedded_entity = await node.embed(
                _graph_state(
                    stem,
                    {
                        'source_entity_hub_synthesis_results_ordered': [result],
                    },
                )
            )
            assert embed_client.inputs == [
                f'{result.definition.canonical_name}: '
                f'{result.definition.description}'
            ]
            assert embedded_entity['source_entity_hubs'][0].aliases == (
                fixture.expected_aliases
            )
            assert embedded_entity['source_entity_hub_memberships'] == [
                fixture.backend_uuids
            ]
        if stem.endswith('TripletHub'):
            group = fixture.backend_items[0]
            embed_client = _CaptureEmbedding()
            node._embedding_client = embed_client
            embedded = await node.embed(
                _graph_state(
                    stem,
                    {
                        f'{_snake_case(stem)}_synthesis_results_ordered': [
                            result
                        ],
                        f'{_snake_case(stem)}_groups': [group],
                    },
                )
            )
            assert embed_client.inputs == [
                f'{result.definition.canonical_name}: '
                f'{result.definition.description}'
            ]
            hub = embedded[f'{_snake_case(stem)}s'][0]
            assert hub.subject_hub_uuid == group.subject_hub_uuid
            assert hub.predicate_hub_uuid == group.predicate_hub_uuid
            assert hub.object_hub_uuid == group.object_hub_uuid
            membership_key = f'{_snake_case(stem)}_memberships'
            expected_memberships = (
                group.triplet_uuids
                if stem == 'SourceTripletHub'
                else group.global_triplet_uuids
            )
            assert embedded[membership_key] == [expected_memberships]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    'stem', [case[0] for case in HUB_CASES], ids=[case[0] for case in HUB_CASES]
)
def test_typed_hub_multilevel_reduction_carries_singletons_and_collects_in_order(
    stem,
):
    async def exercise():
        fixture = build_hub_fixture(stem, count=10)
        node, predictors, budgets = _module_parts(fixture)
        result_payload = await node.synthesis_worker(fixture.state)
        result_key = f'{_snake_case(stem)}_synthesis_results'
        [result] = result_payload[result_key]
        assert result.ordinal == 7
        assert isinstance(
            result.definition,
            getattr(fixture.model_module, f'{stem}Definition'),
        )
        summary_requests = predictors[0].calls
        leaf_batches = [
            request
            for request in summary_requests
            if type(request).__name__.endswith('SummaryInput')
        ]
        assert all(len(request.evidence) == 1 for request in leaf_batches)
        merge_requests = predictors[1].calls
        assert all(len(request.summaries) == 2 for request in merge_requests)
        descendants = {
            summary.text: [index]
            for index, summary in enumerate(predictors[0].outputs)
        }
        depths = {summary.text: 0 for summary in predictors[0].outputs}
        for request, summary in zip(
            merge_requests, predictors[1].outputs, strict=True
        ):
            descendants[summary.text] = [
                index
                for partial in request.summaries
                for index in descendants[partial.text]
            ]
            depths[summary.text] = 1 + max(
                depths[partial.text] for partial in request.summaries
            )
        final_request = predictors[2].calls[0]
        assert [
            index
            for partial in final_request.summaries
            for index in descendants[partial.text]
        ] == list(range(len(predictors[0].outputs)))
        assert (
            max(depths[partial.text] for partial in final_request.summaries)
            >= 3
        )
        assert len(predictors[2].calls) == 1

        member_ids = fixture.backend_uuids
        if 'membership_uuids' in type(result).model_fields:
            assert result.membership_uuids == member_ids
        if 'aliases' in type(result).model_fields and fixture.expected_aliases:
            assert result.aliases == fixture.expected_aliases
        for request in predictors[0].calls:
            assert len(request.evidence) == 1
        for request in predictors[1].calls:
            assert len(request.summaries) == 2
        assert len(final_request.summaries) <= 2
        expected_evidence = [request.evidence[0] for request in leaf_batches]
        assert [
            json.loads(request.evidence[0]) for request in leaf_batches
        ] == [json.loads(evidence) for evidence in expected_evidence]
        state_key = f'{_snake_case(stem)}_synthesis_results'
        later = result.model_copy(update={'ordinal': 9})
        earlier = result.model_copy(update={'ordinal': 1})
        collected = node.collect_synthesis(
            _graph_state(stem, {state_key: [later, earlier]})
        )
        ordered_key = f'{_snake_case(stem)}_synthesis_results_ordered'
        assert [item.ordinal for item in collected[ordered_key]] == [1, 9]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    'stem', [case[0] for case in HUB_CASES], ids=[case[0] for case in HUB_CASES]
)
def test_empty_hub_input_dispatches_to_collect_without_inference(stem):
    fixture = build_hub_fixture(stem)
    node, _, _ = _module_parts(fixture)
    slug = _snake_case(stem)
    input_key = (
        f'{slug}_groups'
        if stem.endswith('TripletHub')
        else f'{slug}_communities'
    )
    assert node.dispatch_synthesis(_graph_state(stem, {input_key: []})) == (
        f'{slug}_synthesis_collect'
    )


@pytest.mark.parametrize(
    ('stem', 'long_field'),
    [
        (
            stem,
            'role_description'
            if stem.endswith('TripletHub')
            else 'description',
        )
        for stem, _scope in HUB_CASES
    ]
    + [
        ('SourceTripletHub', 'fact_text'),
        ('GlobalTripletHub', 'source_triplet_description'),
    ],
)
def test_oversized_whole_record_is_rejected_before_inference(stem, long_field):
    async def exercise():
        large = ('A $x^2$ "quoted" \\\\ condition; ' * 180) + 'Ω'
        fixture = build_hub_fixture(
            stem, count=2, long_field=(long_field, large)
        )
        node, predictors, _ = _module_parts(fixture)

        with pytest.raises(InputBudgetExceeded, match='singleton'):
            await node.synthesis_worker(fixture.state)

        assert all(predictor.calls == [] for predictor in predictors)

    asyncio.run(exercise())
