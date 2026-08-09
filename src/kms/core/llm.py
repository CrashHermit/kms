import asyncio
import os
from functools import lru_cache

import dspy

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEEPSEEK_ENV_KEY = 'DEEPSEEK_API_KEY'
OPENROUTER_ENV_KEY = 'OPENROUTER_API_KEY'
MAX_CONCURRENT_CALLS = int(os.environ.get('KMS_MAX_CONCURRENT_CALLS', '16'))


def gate(limit: int | None = None) -> asyncio.Semaphore:
    return asyncio.Semaphore(limit or MAX_CONCURRENT_CALLS)


def _require_key(env_key: str, example: str) -> str:
    key = os.environ.get(env_key)
    if not key:
        raise RuntimeError(
            f'{env_key} is not set. Export your API key '
            f'(e.g. `export {env_key}={example}`) before running the pipeline.'
        )
    return key


def _provider_routing(provider: str | None) -> dict:
    if not provider:
        return {}
    return {
        'extra_body': {
            'provider': {'order': [provider], 'allow_fallbacks': False},
        }
    }


@lru_cache(maxsize=1)
def text_lm() -> dspy.LM:
    return dspy.LM(
        os.environ.get('TEXT_MODEL', 'deepseek/deepseek-v4-flash'),
        api_key=_require_key(DEEPSEEK_ENV_KEY, 'sk-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=True,
        extra_body={'thinking': {'type': 'disabled'}},
    )


@lru_cache(maxsize=1)
def corrector_lm() -> dspy.LM:
    return dspy.LM(
        os.environ.get(
            'CORRECTOR_MODEL', 'openrouter/qwen/qwen3-vl-235b-a22b-instruct'
        ),
        api_key=_require_key(OPENROUTER_ENV_KEY, 'sk-or-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=True,
        **_provider_routing(os.environ.get('CORRECTOR_PROVIDER', 'DeepInfra')),
    )

