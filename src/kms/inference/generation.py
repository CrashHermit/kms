"""Compile domain materials and enforce complete native-image request capacity."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Literal, get_origin

import dspy
from pydantic import BaseModel, TypeAdapter

from kms.concurrency import run_blocking_work
from kms.core.tokenization.budget import InputBudgetExceeded, TokenBudget
from kms.media.budget import ImagePayloadBudget
from kms.media.content import collect_image_assets
from kms.media.images import load_content_images, load_content_images_sync
from kms.media.metadata import ImageMetadataCache
from kms.media.models import ImageAsset, Material, TextPart

_JSON_INPUTS = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class ProjectedGenerationInput:
    """Backend-facing task values and their prediction-local native catalog."""

    inputs: dict[str, object]
    images: list[ImageAsset]


def project_generation_inputs(
    inputs: dict[str, object],
) -> ProjectedGenerationInput:
    """Replace asset references with local indexes without exposing file paths."""
    images = collect_image_assets(inputs)
    positions = {image.key: position for position, image in enumerate(images)}

    def project(value: object) -> object:
        if isinstance(value, Material):
            return {
                'parts': [
                    {'type': 'text', 'text': part.text}
                    if isinstance(part, TextPart)
                    else {
                        'type': 'image',
                        'image_position': positions[part.asset.key],
                    }
                    for part in value.parts
                ]
            }
        if isinstance(value, ImageAsset):
            return {'image_position': positions[value.key]}
        if isinstance(value, BaseModel):
            return {
                name: project(getattr(value, name))
                for name, field in type(value).model_fields.items()
                if not field.exclude
            }
        if isinstance(value, Mapping):
            return {key: project(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [project(item) for item in value]
        return value

    return ProjectedGenerationInput(
        {name: project(value) for name, value in inputs.items()}, images
    )


def _image_capacity_fits(
    images: Sequence[ImageAsset],
    budget: ImagePayloadBudget,
    metadata: ImageMetadataCache,
) -> bool:
    if len(images) > budget.max_images:
        return False
    byte_count = 0
    pixel_count = 0
    for image in images:
        measured = metadata.measure(image)
        byte_count += measured.byte_count
        pixel_count += measured.width * measured.height
    return byte_count <= budget.max_bytes and pixel_count <= budget.max_pixels


def _capacity_failure(
    projected: ProjectedGenerationInput,
    *,
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget | None,
) -> Literal['workload', 'context', 'images'] | None:
    prompt_images = budget.prompt_images if budget is not None else ()
    images = (*prompt_images, *projected.images)
    if image_budget is not None:
        metadata = (
            budget.counter.image_metadata
            if budget is not None
            else ImageMetadataCache()
        )
        if not _image_capacity_fits(images, image_budget, metadata):
            return 'images'
    if budget is None:
        return None
    input_text = _JSON_INPUTS.dump_json(projected.inputs).decode('utf-8')
    workload_tokens = budget.counter.estimate(
        text=input_text, images=projected.images
    )
    if workload_tokens > budget.token_limit:
        return 'workload'
    if budget.context_token_limit is not None:
        prompt_text = (
            budget.format_request(projected.inputs)
            if budget.format_request is not None
            else input_text
        )
        context_tokens = budget.counter.estimate(
            text=prompt_text, images=images
        )
        if context_tokens > budget.context_token_limit:
            return 'context'
    return None


def fits_generation_request(
    inputs: dict[str, object],
    *,
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget | None,
) -> bool:
    """Measure the live workload and full prompt against their independent caps."""
    return (
        _capacity_failure(
            project_generation_inputs(inputs),
            budget=budget,
            image_budget=image_budget,
        )
        is None
    )


def require_generation_capacity(
    inputs: dict[str, object],
    *,
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget | None,
) -> None:
    """Reject a whole request before image preparation or model dispatch."""
    _require_projected_capacity(
        project_generation_inputs(inputs),
        budget=budget,
        image_budget=image_budget,
    )


def _require_projected_capacity(
    projected: ProjectedGenerationInput,
    *,
    budget: TokenBudget | None,
    image_budget: ImagePayloadBudget | None,
) -> None:
    failure = _capacity_failure(
        projected, budget=budget, image_budget=image_budget
    )
    if failure == 'workload':
        raise InputBudgetExceeded('Content exceeds the configured input budget')
    if failure == 'context':
        raise InputBudgetExceeded(
            'Complete generation prompt exceeds the model input capacity'
        )
    if failure == 'images':
        raise InputBudgetExceeded(
            'Image content exceeds the configured payload budget'
        )


def _with_native_images(
    projected: ProjectedGenerationInput, payloads: list[bytes]
) -> dict[str, object]:
    return {
        **projected.inputs,
        'content_images': [dspy.Image(payload) for payload in payloads],
    }


def prepare_generation_examples(
    examples: Sequence[dspy.Example], signature: type[dspy.Signature]
) -> list[dspy.Example]:
    """Compile domain demonstrations at the same boundary as live requests."""
    prepared = []
    for example in examples:
        inputs = dict(example.inputs())
        projected = project_generation_inputs(inputs)
        native = _with_native_images(
            projected, load_content_images_sync(projected.images, budget=None)
        )
        if 'content_images' not in signature.input_fields:
            native.pop('content_images')
        outputs = {
            name: example[name]
            for name in signature.output_fields
            if name in example
        }
        prepared.append(
            dspy.Example(**native, **outputs).with_inputs(
                *signature.input_fields
            )
        )
    return prepared


@dataclass(frozen=True, slots=True)
class GenerationPrompt:
    """Format a complete textual prompt while measuring native images separately."""

    signature: type[dspy.Signature]
    examples: tuple[dspy.Example, ...]

    def __call__(self, inputs: dict[str, object]) -> str:
        current = dict(inputs)
        if 'content_images' in self.signature.input_fields:
            current['content_images'] = []
        adapter = dspy.settings.adapter or dspy.ChatAdapter()
        messages = adapter.format(self.signature, list(self.examples), current)
        return json.dumps(messages, ensure_ascii=False)


def generation_prompt(
    signature: type[dspy.Signature],
    examples: Sequence[dspy.Example],
) -> tuple[GenerationPrompt, tuple[ImageAsset, ...]]:
    """Build byte-free demonstration text and retain each demo's image costs."""
    projected_examples = []
    images = []
    for example in examples:
        projected = project_generation_inputs(dict(example.inputs()))
        images.extend(projected.images)
        inputs = projected.inputs
        if 'content_images' in signature.input_fields:
            inputs = {**inputs, 'content_images': []}
        outputs = {
            name: example[name]
            for name in signature.output_fields
            if name in example
        }
        projected_examples.append(
            dspy.Example(**inputs, **outputs).with_inputs(
                *signature.input_fields
            )
        )
    return GenerationPrompt(
        generation_signature(signature), tuple(projected_examples)
    ), tuple(images)


def generation_signature(
    signature: type[dspy.Signature],
) -> type[dspy.Signature]:
    """Declare the prepared JSON containers at the SDK input boundary.

    Domain requests retain their typed models. After native preparation their
    structured input values are JSON dictionaries/lists, not those domain
    instances; output schemas and field descriptions remain unchanged.
    """
    fields = deepcopy(signature.fields)
    for name in signature.input_fields:
        if name == 'content_images':
            continue
        annotation = fields[name].annotation
        origin = get_origin(annotation)
        if annotation is ImageAsset or (
            isinstance(annotation, type) and issubclass(annotation, BaseModel)
        ):
            fields[name].annotation = dict[str, object]
        elif origin in (list, tuple):
            fields[name].annotation = list[object]
    return dspy.Signature(fields, signature.instructions)


class NativePredictor(dspy.Module):
    """Prepare typed domain inputs once, then call the configured backend."""

    def __init__(
        self,
        predictor: dspy.Module,
        signature: type[dspy.Signature],
        *,
        budget: TokenBudget | None,
        image_budget: ImagePayloadBudget | None,
        supports_images: bool,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.signature = signature
        self._budget = budget
        self._image_budget = image_budget
        self._supports_images = (
            supports_images and 'content_images' in signature.input_fields
        )

    def forward(self, **inputs: object) -> object:
        """Run synchronous preparation outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acall(**inputs))
        raise RuntimeError('use acall() from an active event loop')

    async def aforward(self, **inputs: object) -> object:
        """Check capacity and preserve ordered native evidence at dispatch."""
        projected = project_generation_inputs(inputs)
        prompt_images = (
            self._budget.prompt_images if self._budget is not None else ()
        )
        if (projected.images or prompt_images) and not self._supports_images:
            raise ValueError(
                'Selected model profile does not support native images'
            )
        await run_blocking_work(
            partial(
                _require_projected_capacity,
                projected,
                budget=self._budget,
                image_budget=self._image_budget,
            )
        )
        payloads = await load_content_images(
            projected.images, budget=self._image_budget
        )
        prepared = _with_native_images(projected, payloads)
        if 'content_images' not in self.signature.input_fields:
            prepared.pop('content_images')
        return await self.predictor.acall(**prepared)
