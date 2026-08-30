"""Shared language-model factories and concurrency gates."""

import asyncio
from functools import cache

import dspy

from kms import config


def configured_serving_model(module_name: str) -> str:
    """Returns the configured serving preset for one logical module."""
    module_models = config.get_settings().serving.module_models
    try:
        return module_models[module_name]
    except KeyError as exc:
        raise RuntimeError(
            f'No serving model configured for module {module_name!r}.'
        ) from exc


def gate(limit: int | None = None) -> asyncio.Semaphore:
    """Returns a semaphore limiting concurrent LLM calls.

    Args:
        limit: The concurrency cap; defaults to the configured value.
    """
    if limit is None:
        limit = config.get_settings().concurrency.max_concurrent_calls
    return asyncio.Semaphore(limit)


def _require_key(value: str, name: str, example: str) -> str:
    """Returns a configured API key or raises."""
    if not value:
        raise RuntimeError(
            f'{name} is not set. Export your API key '
            f'(e.g. `export {name}={example}`) before running KMS.'
        )
    return value


def _provider_routing(provider: str | None) -> dict:
    """Builds the OpenRouter provider-pinning extra body."""
    if not provider:
        return {}
    return {
        'extra_body': {
            'provider': {'order': [provider], 'allow_fallbacks': False},
        }
    }


@cache
def module_lm(module_name: str) -> dspy.LM:
    """Returns the LM configured for one module.

    Module names are configuration keys, not aliases for a shared pipeline
    role. Keeping resolution here means every module has an explicit model
    boundary while callers remain independent of provider details.
    """
    settings = config.get_settings()
    try:
        module_config = settings.models.modules[module_name]
    except KeyError as exc:
        raise RuntimeError(
            f'No model configured for module {module_name!r}. '
            f'Add [models.modules.{module_name}] to config.toml.'
        ) from exc

    if module_config.base_url:
        model = _require_key(
            module_config.model,
            f'KMS_MODELS__MODULES__{module_name.upper()}__MODEL',
            'qwen3.5-9b-text',
        )
        if not model.startswith('openai/'):
            model = f'openai/{model}'
        language_model = dspy.LM(
            model,
            api_base=module_config.base_url,
            api_key=module_config.api_key or 'not-needed',
            temperature=module_config.temperature,
            max_tokens=module_config.max_tokens,
            cache=True,
        )
        language_model._kms_module_name = module_name
        return language_model

    language_model = dspy.LM(
        module_config.model,
        api_key=_require_key(
            settings.models.openrouter_api_key
            if module_config.api_key_source == 'openrouter'
            else settings.models.deepseek_api_key,
            'KMS_MODELS__OPENROUTER_API_KEY'
            if module_config.api_key_source == 'openrouter'
            else 'KMS_MODELS__DEEPSEEK_API_KEY',
            'sk-or-...'
            if module_config.api_key_source == 'openrouter'
            else 'sk-...',
        ),
        temperature=module_config.temperature,
        max_tokens=module_config.max_tokens,
        cache=True,
        **_provider_routing(module_config.provider or None),
    )
    return language_model
