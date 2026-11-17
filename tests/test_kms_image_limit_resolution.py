"""Model and operational image limits resolve for whole requests."""

from io import BytesIO

from PIL import Image

from kms.config.inference import VisionInferenceSettings
from kms.config.settings import Settings
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.inference.generation import fits_generation_request
from kms.inference.predictors import resolve_image_budget
from kms.media.models import ImageAsset, ImagePart, Material


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def test_profile_and_operational_caps_are_combined_for_multi_model_stages():
    settings = Settings(_env_file=None)
    settings.llm.image_payload.max_images = 7
    settings.llm.image_payload.max_bytes = 10000
    settings.llm.image_payload.max_pixels = 50000
    settings.llm.profiles['chatgpt-luna'].image_limits.max_images = 3
    settings.llm.profiles['gemma-vision-8k'].image_limits.max_bytes = 1000
    first = VisionInferenceSettings(profile='chatgpt-luna')
    second = VisionInferenceSettings(profile='gemma-vision-8k')
    budget = resolve_image_budget(settings, first, second)
    assert (budget.max_images, budget.max_bytes, budget.max_pixels) == (
        3,
        1000,
        50000,
    )
    assert resolve_image_budget(settings, second).max_images == 7


def test_profile_and_operational_limits_load_from_environment(monkeypatch):
    monkeypatch.setenv('KMS_LLM__IMAGE_PAYLOAD__MAX_IMAGES', '7')
    monkeypatch.setenv(
        'KMS_LLM__PROFILES__chatgpt-luna__IMAGE_LIMITS__MAX_IMAGES', '2'
    )
    settings = Settings(_env_file=None)
    assert settings.llm.image_payload.max_images == 7
    assert (
        resolve_image_budget(settings, VisionInferenceSettings()).max_images
        == 2
    )


def test_same_resolved_image_limit_applies_to_complete_request():
    settings = Settings(_env_file=None)
    settings.llm.image_payload.max_images = 5
    settings.llm.profiles['chatgpt-luna'].image_limits.max_images = 1
    image_budget = resolve_image_budget(settings, VisionInferenceSettings())
    data = BytesIO()
    Image.new('RGB', (4, 4)).save(data, format='PNG')
    first = ImageAsset(('first',), data.getvalue())
    second = ImageAsset(('second',), data.getvalue())
    budget = TokenBudget(MaterialTokenEstimator(_Counter()), 100000)
    assert fits_generation_request(
        {'evidence': Material(parts=(ImagePart(asset=first),))},
        budget=budget,
        image_budget=image_budget,
    )
    assert not fits_generation_request(
        {
            'evidence': Material(
                parts=(ImagePart(asset=first), ImagePart(asset=second))
            )
        },
        budget=budget,
        image_budget=image_budget,
    )
