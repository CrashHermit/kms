"""Behavioral checks for bounded final search relevance selection."""

import asyncio
import json
from io import BytesIO

import dspy
import pytest
from PIL import Image

from kms.config.inference import VisionInferenceSettings
from kms.config.settings import Settings
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import project_generation_inputs
from kms.inference.predictors import build_token_budget
from kms.media.budget import ImagePayloadBudget
from kms.media.metadata import ImageMetadataCache
from kms.media.models import ImageAsset, ImagePart, Material, TextPart
from kms.retrieval.selection.inference import SearchSelectionModule
from kms.retrieval.selection.models import (
    SearchMaterialKind,
    SearchSelectionCandidate,
    SearchSelectionInput,
)
from kms.retrieval.selection.selector import SearchSelector


class _Tokenizers:
    def for_inference(self, inference: VisionInferenceSettings):
        return _estimator()


class _CharacterCounter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _estimator() -> MaterialTokenEstimator:
    return MaterialTokenEstimator(
        text_counter=_CharacterCounter(),
        image_metadata=ImageMetadataCache(),
    )


class _SelectionModule:
    def __init__(self, selected: list[int] | Exception) -> None:
        self.selected = selected
        self.requests: list[SearchSelectionInput] = []

    async def acall(self, *, request: SearchSelectionInput) -> list[int]:
        self.requests.append(request)
        if isinstance(self.selected, Exception):
            raise self.selected
        return self.selected


def _candidates() -> list[SearchSelectionCandidate]:
    return [
        SearchSelectionCandidate(
            position=0, material=Material.from_text('first')
        ),
        SearchSelectionCandidate(
            position=1, material=Material.from_text('second')
        ),
        SearchSelectionCandidate(
            position=2, material=Material.from_text('third')
        ),
    ]


def _serialized_size(
    query: Material, candidates: list[SearchSelectionCandidate]
) -> int:
    request = SearchSelectionInput(
        query=query,
        material_kind=SearchMaterialKind.SOURCE_BLOCK,
        candidates=candidates,
    )
    return len(
        json.dumps(
            {'request': request.model_dump(mode='json')},
            separators=(',', ':'),
        )
    )


def _service(module: _SelectionModule, *, token_limit: int) -> SearchSelector:
    return SearchSelector(
        module,
        TokenBudget(counter=_estimator(), token_limit=token_limit),
        image_budget=ImagePayloadBudget(
            max_images=4,
            max_bytes=26214400,
            max_pixels=40000000,
        ),
    )


async def _select_with_budget(
    *,
    budget: int,
    candidates: list[SearchSelectionCandidate],
    query: Material | None = None,
) -> tuple[list[int], _SelectionModule]:
    module = _SelectionModule([1, 0])
    selected = await _service(module, token_limit=budget).select(
        query=query or Material.from_text('why?'),
        material_kind=SearchMaterialKind.SOURCE_BLOCK,
        candidates=candidates,
    )
    return selected, module


async def _test_select_uses_largest_whole_prefix_and_preserves_positions():
    candidates = _candidates()
    query = Material.from_text('why?')
    budget = _serialized_size(query, candidates[:2])
    selected, module = await _select_with_budget(
        budget=budget,
        candidates=candidates,
        query=query,
    )

    assert selected == [1, 0]
    assert module.requests[0].query == query
    assert [
        candidate.position for candidate in module.requests[0].candidates
    ] == [
        0,
        1,
    ]
    assert module.requests[0].candidates[1].material.text == 'second'


def test_select_uses_largest_whole_prefix_and_preserves_positions():
    asyncio.run(
        _test_select_uses_largest_whole_prefix_and_preserves_positions()
    )


async def _test_exact_budget_is_inclusive_and_singletons_are_judged():
    candidates = _candidates()[:1]
    query = Material.from_text('why?')
    selected, module = await _select_with_budget(
        budget=_serialized_size(query, candidates),
        candidates=candidates,
        query=query,
    )
    assert selected == [1, 0]
    assert len(module.requests) == 1
    assert module.requests[0].candidates == candidates


def test_exact_budget_is_inclusive_and_singletons_are_judged():
    asyncio.run(_test_exact_budget_is_inclusive_and_singletons_are_judged())


async def _test_oversized_query_or_first_candidate_fails_before_inference():
    module = _SelectionModule([])
    service = _service(module, token_limit=5)
    with pytest.raises(
        InputBudgetExceeded,
        match='configured input budget',
    ):
        await service.select(
            query=Material.from_text('query that cannot fit'),
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=_candidates(),
        )
    assert module.requests == []

    query = Material.from_text('q')
    first = SearchSelectionCandidate(
        position=0, material=Material.from_text('oversized')
    )
    service = _service(module, token_limit=_serialized_size(query, []) + 1)
    with pytest.raises(InputBudgetExceeded):
        await service.select(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=[first],
        )
    assert module.requests == []


def test_oversized_query_or_first_candidate_fails_before_inference():
    asyncio.run(
        _test_oversized_query_or_first_candidate_fails_before_inference()
    )


async def _test_empty_candidates_skip_inference_and_failures_propagate():
    module = _SelectionModule(RuntimeError('provider unavailable'))
    service = _service(module, token_limit=1000)
    assert (
        await service.select(
            query=Material.from_text('anything'),
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=[],
        )
        == []
    )
    assert module.requests == []

    with pytest.raises(RuntimeError, match='provider unavailable'):
        await service.select(
            query=Material.from_text('why?'),
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=_candidates()[:1],
        )


def test_empty_candidates_skip_inference_and_failures_propagate():
    asyncio.run(_test_empty_candidates_skip_inference_and_failures_propagate())


def test_selection_module_accepts_domain_request_inputs():
    class Predictor(dspy.Module):
        def forward(self, *, request: SearchSelectionInput):
            return dspy.Prediction(
                selected_positions=[request.candidates[0].position]
            )

        async def aforward(self, *, request: SearchSelectionInput):
            return dspy.Prediction(
                selected_positions=[request.candidates[0].position]
            )

    request = SearchSelectionInput(
        query=Material.from_text('query'),
        material_kind=SearchMaterialKind.SOURCE_UNIT,
        candidates=[
            SearchSelectionCandidate(
                position=0,
                material=Material.from_text('solve the exercise'),
                unit_kind='exercise',
            )
        ],
    )
    module = SearchSelectionModule(Predictor())

    assert module(request=request) == [0]
    assert asyncio.run(module.acall(request=request)) == [0]


def test_selection_module_preserves_image_only_and_mixed_query_material():
    output = BytesIO()
    Image.new('RGB', (2, 2), color='blue').save(output, format='PNG')
    asset = ImageAsset(('query',), output.getvalue())

    class Predictor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.requests: list[SearchSelectionInput] = []

        def forward(self, *, request: SearchSelectionInput):
            self.requests.append(request)
            return dspy.Prediction(selected_positions=[0])

        async def aforward(self, *, request: SearchSelectionInput):
            self.requests.append(request)
            return dspy.Prediction(selected_positions=[0])

    queries = [
        Material.from_image(asset),
        Material(
            parts=(TextPart(text='compare this with '), ImagePart(asset=asset))
        ),
    ]
    predictor = Predictor()
    module = SearchSelectionModule(predictor)
    candidate = SearchSelectionCandidate(
        position=0, material=Material.from_text('candidate')
    )

    for query in queries:
        request = SearchSelectionInput(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=[candidate],
        )
        assert module(request=request) == [0]
        assert asyncio.run(module.acall(request=request)) == [0]
        projected = project_generation_inputs({'request': request})
        assert projected.images == [asset]
        projected_parts = projected.inputs['request']['query']['parts']
        assert sum(part['type'] == 'image' for part in projected_parts) == 1
        assert projected_parts[-1]['image_position'] == 0

    assert [request.query for request in predictor.requests] == [
        query for query in queries for _ in range(2)
    ]


def test_default_search_selection_settings_are_constructible():
    selection = Settings(_env_file=None).search_selection
    assert selection.inference.max_tokens == 1024
    assert selection.inference.num_retries == 0
    assert selection.input_token_budget == 16384
    assert selection.safety_margin_tokens == 256


def test_selection_budget_respects_vision_context_and_output_reserve():
    settings = Settings(_env_file=None)
    inference = VisionInferenceSettings(
        profile='gemma-vision-8k',
        max_tokens=1024,
        num_retries=0,
    )
    budget = build_token_budget(
        settings,
        _Tokenizers(),
        inference,
        input_token_budget=16384,
        safety_margin_tokens=256,
    )
    profile = settings.local_models.router.model_server_profiles[
        'gemma-vision-8k'
    ]
    assert budget.token_limit == 16384
    assert budget.context_token_limit == (
        profile.context_size // profile.parallel - inference.max_tokens - 256
    )
    assert budget.context_token_limit > 0

    impossible = VisionInferenceSettings(
        profile='gemma-vision-8k',
        max_tokens=8192,
        num_retries=0,
    )
    impossible_budget = build_token_budget(
        settings,
        _Tokenizers(),
        impossible,
        input_token_budget=16384,
        safety_margin_tokens=256,
    )
    service = SearchSelector(
        _SelectionModule([0]),
        impossible_budget,
        image_budget=ImagePayloadBudget(
            max_images=4,
            max_bytes=26214400,
            max_pixels=40000000,
        ),
    )
    with pytest.raises(InputBudgetExceeded):
        asyncio.run(
            service.select(
                query=Material.from_text('too much output reserved'),
                material_kind=SearchMaterialKind.SOURCE_BLOCK,
                candidates=_candidates()[:1],
            )
        )


def test_selector_keeps_only_the_fitting_native_evidence_prefix():
    output = BytesIO()
    Image.new('RGB', (2, 2), color='red').save(output, format='PNG')
    payload = output.getvalue()
    assets = [ImageAsset((str(index),), payload) for index in range(6)]
    candidates = [
        SearchSelectionCandidate(
            position=0,
            material=Material(
                parts=tuple(ImagePart(asset=asset) for asset in assets[:3])
            ),
        ),
        SearchSelectionCandidate(
            position=1,
            material=Material(
                parts=tuple(ImagePart(asset=asset) for asset in assets[3:])
            ),
        ),
    ]
    query = Material.from_image(assets[5])
    module = _SelectionModule([0])
    selected = asyncio.run(
        _service(module, token_limit=10000).select(
            query=query,
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=candidates,
        )
    )

    assert selected == [0]
    assert module.requests[0].query == query
    assert module.requests[0].query.images[0].key == ('5',)
    assert [
        image.key for image in module.requests[0].candidates[0].material.images
    ] == [('0',), ('1',), ('2',)]
    assert len(module.requests[0].candidates) == 1
