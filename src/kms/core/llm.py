"""
Central LLM configuration for the DSPy modules.

Every DSPy module in this package delegates to a shared language model built
here, so the model choice, credentials, and routing live in one place instead of
being duplicated across nodes.

Three text backends, two vision — five total:

- Text reasoning nodes (extractor, seam merger, and the per-type entity finders) run
  on DeepSeek V4 Flash via DeepSeek's own API (litellm ``deepseek/`` provider, base
  https://api.deepseek.com). DeepSeek does automatic server-side context caching, so no
  provider pinning is needed. The key is read from DEEPSEEK_API_KEY.
- LLM-as-judge metrics for prompt optimisation use DeepSeek V4 Pro — same API key,
  just a different model name (``deepseek/deepseek-v4-pro``; override via METRIC_MODEL).
- The correction pass sends a page image, so it runs on Qwen3-VL-235B via OpenRouter.
- The corrector judge is MiMo-V2.5 via OpenRouter — a cheaper VLM from a different
  family, pinned to the same DeepInfra upstream for prompt-cache warmth.
  The key is read from OPENROUTER_API_KEY.

OpenRouter provider pinning (corrector)
---------------------------------------
OpenRouter can route the same model to different upstream providers between
requests, which defeats provider-side prompt caching. To keep cache hits warm we
pin the corrector to a single upstream provider (``allow_fallbacks: false``) via
OpenRouter's provider-routing preference, defaulting to DeepInfra (262k context +
prompt caching). Override CORRECTOR_PROVIDER to pin a different upstream, or set it
empty to unpin.

Keys are read from the environment — never hard-code them. LM objects are cached
so every module sharing a backend shares one instance (and therefore one
connection pool and prompt cache).
"""

import os
from functools import lru_cache

import dspy

# Load a local .env (if present) so the two API keys can live in a file instead of
# being exported by hand. Guarded: python-dotenv is a convenience, not a hard dep,
# and a missing .env is fine — keys still resolve from the real environment.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEEPSEEK_ENV_KEY = 'DEEPSEEK_API_KEY'
OPENROUTER_ENV_KEY = 'OPENROUTER_API_KEY'


def _require_key(env_key: str, example: str) -> str:
    """Return the named API key, raising a clear error if it is unset.

    We raise on use rather than at import time so the modules stay importable
    without credentials — the key is only required once a node actually runs.
    """
    key = os.environ.get(env_key)
    if not key:
        raise RuntimeError(
            f'{env_key} is not set. Export your API key '
            f'(e.g. `export {env_key}={example}`) before running the pipeline.'
        )
    return key


def _provider_routing(provider: str | None) -> dict:
    """OpenRouter provider-routing kwargs that pin a single upstream provider.

    Pinning keeps repeated calls on the same backend so its prompt cache stays
    warm. An empty/None provider means "let OpenRouter choose" (no pinning).
    """
    if not provider:
        return {}
    return {
        'extra_body': {
            'provider': {'order': [provider], 'allow_fallbacks': False},
        }
    }


@lru_cache(maxsize=1)
def text_lm() -> dspy.LM:
    """DeepSeek V4 Flash via DeepSeek's own API for the text reasoning nodes.

    Uses litellm's ``deepseek/`` provider (base https://api.deepseek.com), which
    reads the key we pass from DEEPSEEK_API_KEY. DeepSeek caches context
    server-side automatically, so there is no provider to pin.

    Thinking mode is disabled: v4-flash defaults to thinking and intermittently
    emits the whole answer into ``reasoning_content`` with an empty content
    channel, which makes dspy's adapter fail to parse. These nodes are extraction
    / classification and dspy's ChainOfThought already elicits its own reasoning
    field, so model-level thinking is redundant here — turning it off is both more
    reliable and cheaper.
    """
    return dspy.LM(
        os.environ.get('TEXT_MODEL', 'deepseek/deepseek-v4-flash'),
        api_key=_require_key(DEEPSEEK_ENV_KEY, 'sk-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=False,
        extra_body={'thinking': {'type': 'disabled'}},
    )


@lru_cache(maxsize=1)
def metric_lm() -> dspy.LM:
    """DeepSeek V4 Pro for LLM-as-judge metric scoring.

    Uses the same API key as ``text_lm`` (``DEEPSEEK_API_KEY``) — just a different
    model name. Thinking mode is disabled because the metric signature's own
    ``ChainOfThought`` already elicits reasoning.

    Override with the ``METRIC_MODEL`` env var.
    """
    return dspy.LM(
        os.environ.get('METRIC_MODEL', 'deepseek/deepseek-v4-pro'),
        api_key=_require_key(DEEPSEEK_ENV_KEY, 'sk-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=False,
        extra_body={'thinking': {'type': 'disabled'}},
    )


@lru_cache(maxsize=1)
def corrector_lm() -> dspy.LM:
    """Qwen3-VL-235B (via OpenRouter) for the correction pass on the Mistral front-end.

    The corrector reads a page image *and* Mistral's markdown and returns a corrected
    transcription — a verification task that a strong vision model does reliably (tested:
    Qwen3-VL-235B fixed subtle math errors, e.g. a misread root index, without disturbing
    correct content, where smaller models were less reliable). Set CORRECTOR_MODEL to swap
    the model; CORRECTOR_PROVIDER pins a single OpenRouter upstream so prompt caching stays
    warm (default DeepInfra; empty to unpin).
    """
    return dspy.LM(
        os.environ.get(
            'CORRECTOR_MODEL', 'openrouter/qwen/qwen3-vl-235b-a22b-instruct'
        ),
        api_key=_require_key(OPENROUTER_ENV_KEY, 'sk-or-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=False,
        **_provider_routing(os.environ.get('CORRECTOR_PROVIDER', 'DeepInfra')),
    )


@lru_cache(maxsize=1)
def prompt_optimizer_lm() -> dspy.LM:
    """DeepSeek V4 Pro for MIPROv2 prompt-model (instruction generation).

    Prompt optimisation is a meta-level creative task — writing better
    instructions — so it gets the same Pro model as the metrics judge.
    Override with ``PROMPT_OPTIMIZER_MODEL``.
    """
    return dspy.LM(
        os.environ.get(
            'PROMPT_OPTIMIZER_MODEL', 'deepseek/deepseek-v4-pro'
        ),
        api_key=_require_key(DEEPSEEK_ENV_KEY, 'sk-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=False,
        extra_body={'thinking': {'type': 'disabled'}},
    )


@lru_cache(maxsize=1)
def corrector_judge_lm() -> dspy.LM:
    """MiMo-V2.5 (via OpenRouter) for judging the corrector's output against the page image.

    A separate, stronger VLM from a different family than the corrector's Qwen —
    avoids self-grading bias. Provider-pinned to DeepInfra for prompt-cache warmth
    on repeated structured judge calls. The corrector judge is the only metric that
    needs a VLM (text stages use ``metric_lm``).

    Override with ``CORRECTOR_JUDGE_MODEL``; unpin with ``CORRECTOR_JUDGE_PROVIDER``.
    """
    return dspy.LM(
        os.environ.get(
            'CORRECTOR_JUDGE_MODEL', 'openrouter/xiaomi/mimo-v2.5'
        ),
        api_key=_require_key(OPENROUTER_ENV_KEY, 'sk-or-...'),
        temperature=0.0,
        max_tokens=128000,
        cache=False,
        **_provider_routing(
            os.environ.get('CORRECTOR_JUDGE_PROVIDER', 'DeepInfra')
        ),
    )
