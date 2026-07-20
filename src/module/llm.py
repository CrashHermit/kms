"""
Central LLM configuration for the DSPy modules.

Every DSPy module in this package delegates to a shared language model built
here, so the model choice, credentials, and routing live in one place instead of
being duplicated across nodes.

Two backends, each on its native/best gateway:

- Text reasoning nodes (extractor, seam merger, problem refiner, instruction
  governor) run on DeepSeek V4 Flash via DeepSeek's own API
  (litellm ``deepseek/`` provider, base https://api.deepseek.com). DeepSeek does
  automatic server-side context caching, so no provider pinning is needed. The
  key is read from DEEPSEEK_API_KEY.
- Vision nodes (OCR, image filter) send images, so they run on Qwen3-VL-235B via
  OpenRouter. The key is read from OPENROUTER_API_KEY.

OpenRouter provider pinning (vision)
------------------------------------
OpenRouter can route the same model to different upstream providers between
requests, which defeats provider-side prompt caching. To keep cache hits warm we
pin the vision model to a single upstream provider (``allow_fallbacks: false``)
via OpenRouter's provider-routing preference, defaulting to DeepInfra (262k
context + prompt caching). Override VISION_PROVIDER to pin a different upstream,
or set it empty to unpin.

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

DEEPSEEK_ENV_KEY = "DEEPSEEK_API_KEY"
OPENROUTER_ENV_KEY = "OPENROUTER_API_KEY"


def _require_key(env_key: str, example: str) -> str:
    """Return the named API key, raising a clear error if it is unset.

    We raise on use rather than at import time so the modules stay importable
    without credentials — the key is only required once a node actually runs.
    """
    key = os.environ.get(env_key)
    if not key:
        raise RuntimeError(
            f"{env_key} is not set. Export your API key "
            f"(e.g. `export {env_key}={example}`) before running the pipeline."
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
        "extra_body": {
            "provider": {"order": [provider], "allow_fallbacks": False},
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
        os.environ.get("TEXT_MODEL", "deepseek/deepseek-v4-flash"),
        api_key=_require_key(DEEPSEEK_ENV_KEY, "sk-..."),
        temperature=0.0,
        max_tokens=8000,
        extra_body={"thinking": {"type": "disabled"}},
    )


@lru_cache(maxsize=1)
def vision_lm() -> dspy.LM:
    """Qwen3-VL-235B (via OpenRouter) for the image-consuming nodes (OCR, image filter)."""
    return dspy.LM(
        os.environ.get("VISION_MODEL", "openrouter/qwen/qwen3-vl-235b-a22b-instruct"),
        api_key=_require_key(OPENROUTER_ENV_KEY, "sk-or-..."),
        temperature=0.0,
        max_tokens=8000,
        **_provider_routing(os.environ.get("VISION_PROVIDER", "DeepInfra")),
    )
