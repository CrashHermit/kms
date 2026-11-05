"""Behavioral checks for bounded final search relevance selection."""

import asyncio
from io import BytesIO

import dspy
import pytest
from PIL import Image

from kms.composition.predictors import build_token_budget
from kms.config.inference import VisionInferenceSettings
from kms.config.settings import Settings
from kms.core.content import ContentImage
from kms.core.media_budget import ImagePayloadBudget
from kms.core.model.content import MultimodalContent
from kms.core.model.search import (
    SearchMaterialKind,
    SearchSelectionCandidate,
    SearchSelectionInput,
)
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.core.tokenization.images import ImageMetadataCache
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.module.search.selection import SearchSelectionModule
from kms.service.search_selection import SearchSelectionService


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
        self.images: list[list[dspy.Image]] = []

    async def acall(
        self,
        *,
        request: SearchSelectionInput,
        content_images: list[dspy.Image],
    ) -> list[int]:
        self.requests.append(request)
        self.images.append(content_images)
        if isinstance(self.selected, Exception):
            raise self.selected
        return self.selected


def _candidates() -> list[SearchSelectionCandidate]:
    return [
        SearchSelectionCandidate(
            position=0, material=MultimodalContent(text='first')
        ),
        SearchSelectionCandidate(
            position=1, material=MultimodalContent(text='second')
        ),
        SearchSelectionCandidate(
            position=2, material=MultimodalContent(text='third')
        ),
    ]


def _serialized_size(
    query: str, candidates: list[SearchSelectionCandidate]
) -> int:
    return len(
        SearchSelectionInput(
            query=query,
            material_kind=SearchMaterialKind.PROPOSITION,
            candidates=candidates,
        ).model_dump_json()
    )


def _service(
    module: _SelectionModule, *, token_limit: int
) -> SearchSelectionService:
    return SearchSelectionService(
        module,
        TokenBudget(counter=_estimator(), token_limit=token_limit),
        image_budget=ImagePayloadBudget(
            max_images=4,
            max_bytes=26214400,
            max_pixels=40000000,
        ),
    )


async def _select_with_budget(
    *, budget: int, candidates: list[SearchSelectionCandidate]
) -> tuple[list[int], _SelectionModule]:
    module = _SelectionModule([1, 0])
    selected = await _service(module, token_limit=budget).select(
        query='why?',
        material_kind=SearchMaterialKind.PROPOSITION,
        candidates=candidates,
        content_images=[],
    )
    return selected, module


async def _test_select_uses_largest_whole_prefix_and_preserves_positions():
    candidates = _candidates()
    budget = _serialized_size('why?', candidates[:2])
    selected, module = await _select_with_budget(
        budget=budget,
        candidates=candidates,
    )

    assert selected == [1, 0]
    assert [
        candidate.position for candidate in module.requests[0].candidates
    ] == [
        0,
        1,
    ]
    assert module.requests[0].candidates[1].material.text == 'second'
    assert module.images == [[]]


def test_select_uses_largest_whole_prefix_and_preserves_positions():
    asyncio.run(
        _test_select_uses_largest_whole_prefix_and_preserves_positions()
    )


async def _test_exact_budget_is_inclusive_and_singletons_are_judged():
    candidates = _candidates()[:1]
    selected, module = await _select_with_budget(
        budget=_serialized_size('why?', candidates),
        candidates=candidates,
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
        match='Search selection input exceeds the configured budget',
    ):
        await service.select(
            query='query that cannot fit',
            material_kind=SearchMaterialKind.PROPOSITION,
            candidates=_candidates(),
            content_images=[],
        )
    assert module.requests == []

    query = 'q'
    first = SearchSelectionCandidate(
        position=0, material=MultimodalContent(text='oversized')
    )
    service = _service(module, token_limit=_serialized_size(query, []) + 1)
    with pytest.raises(InputBudgetExceeded):
        await service.select(
            query=query,
            material_kind=SearchMaterialKind.PROPOSITION,
            candidates=[first],
            content_images=[],
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
            query='anything',
            material_kind=SearchMaterialKind.LEARNER_OBSERVATION,
            candidates=[],
            content_images=[],
        )
        == []
    )
    assert module.requests == []

    with pytest.raises(RuntimeError, match='provider unavailable'):
        await service.select(
            query='why?',
            material_kind=SearchMaterialKind.PROPOSITION,
            candidates=_candidates()[:1],
            content_images=[],
        )


def test_empty_candidates_skip_inference_and_failures_propagate():
    asyncio.run(_test_empty_candidates_skip_inference_and_failures_propagate())


def _png() -> bytes:
    buffer = BytesIO()
    Image.new('RGB', (2, 2), color='red').save(buffer, format='PNG')
    return buffer.getvalue()


def test_selector_projects_and_dispatches_only_images_in_fitting_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    payloads = [_png() for _ in range(4)]
    catalog = [
        ContentImage(key=(str(index),), payload=payload)
        for index, payload in enumerate(payloads)
    ]
    candidates = [
        SearchSelectionCandidate(
            position=0,
            material=MultimodalContent(text='kept', image_positions=[1]),
        ),
        SearchSelectionCandidate(
            position=1,
            material=MultimodalContent(text='too large', image_positions=[3]),
        ),
    ]
    module = _SelectionModule([0])
    service = _service(
        module,
        token_limit=_serialized_size('why?', candidates[:1]) + 10,
    )
    loaded_catalogs: list[list[ContentImage]] = []

    async def load(images, *, budget):
        loaded_catalogs.append(images)
        return [image.payload for image in images]

    monkeypatch.setattr(
        'kms.service.search_selection.load_content_images', load
    )
    asyncio.run(
        service.select(
            query='why?',
            material_kind=SearchMaterialKind.SOURCE_BLOCK,
            candidates=candidates,
            content_images=catalog,
        )
    )

    assert [image.key for image in loaded_catalogs[0]] == [('1',)]
    assert module.requests[0].candidates[0].material.image_positions == [0]
    assert module.images[0] and len(module.images[0]) == 1


def test_selection_module_passes_native_images_to_sync_and_async_predictors():
    class Predictor(dspy.Module):
        def forward(
            self,
            *,
            request: SearchSelectionInput,
            content_images: list[dspy.Image],
        ):
            assert content_images == []
            return dspy.Prediction(
                selected_positions=[request.candidates[0].position]
            )

        async def aforward(
            self,
            *,
            request: SearchSelectionInput,
            content_images: list[dspy.Image],
        ):
            assert content_images == []
            return dspy.Prediction(
                selected_positions=[request.candidates[0].position]
            )

    request = SearchSelectionInput(
        query='query',
        material_kind=SearchMaterialKind.STATEMENT,
        candidates=[
            SearchSelectionCandidate(
                position=0,
                material=MultimodalContent(text='solve the exercise'),
                is_exercise=True,
            )
        ],
    )
    module = SearchSelectionModule(Predictor())

    assert module(request=request, content_images=[]) == [0]
    assert asyncio.run(module.acall(request=request, content_images=[])) == [0]


def test_default_search_selection_settings_are_constructible():
    selection = Settings(_env_file=None).search_selection
    assert selection.inference.max_tokens == 1024
    assert selection.inference.num_retries == 0
    assert selection.input_token_budget == 16384
    assert selection.safety_margin_tokens == 256
    assert selection.image_payload.max_images == 4


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
    assert budget.token_limit == (
        profile.context_size // profile.parallel - inference.max_tokens - 256
    )
    assert budget.token_limit > 0

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
    service = SearchSelectionService(
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
                query='too much output reserved',
                material_kind=SearchMaterialKind.PROPOSITION,
                candidates=_candidates()[:1],
                content_images=[],
            )
        )
