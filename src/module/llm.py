"""
Central LLM configuration for the DSPy modules.

Every DSPy module in this package delegates to a shared language model built
here, so the model choice, credentials, and routing live in one place instead of
being duplicated across nodes.

Everything is routed through OpenRouter (one OPENROUTER_API_KEY):

- Text reasoning nodes (extractor, seam merger, exercise/instruction refiners,
  instruction distributor) run on DeepSeek V4 Flash.
- Vision nodes (OCR, image filter) send images, so they run on Qwen3-VL-235B,
  a vision-language model.

Provider pinning
----------------
OpenRouter can route the same model to different upstream providers between
requests, which defeats provider-side prompt caching. To keep cache hits warm we
pin each model to a single upstream provider (``allow_fallbacks: false``) via
OpenRouter's provider-routing preference. The provider is configurable per
backend; the text path defaults to DeepSeek (a known-correct slug), the vision
path is left unpinned by default because the right Qwen provider slug depends on
availability — set VISION_PROVIDER to pin it once chosen.

The API key is read from the OPENROUTER_API_KEY environment variable — never
hard-code it. LM objects are cached so every module sharing a backend shares one
instance (and therefore one connection pool and prompt cache).
"""

import os
from functools import lru_cache

import dspy

OPENROUTER_ENV_KEY = "OPENROUTER_API_KEY"


def _require_key() -> str:
    """Return the OpenRouter API key, raising a clear error if it is unset.

    We raise on use rather than at import time so the modules stay importable
    without credentials — the key is only required once a node actually runs.
    """
    key = os.environ.get(OPENROUTER_ENV_KEY)
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_ENV_KEY} is not set. Export your OpenRouter API key "
            f"(e.g. `export {OPENROUTER_ENV_KEY}=sk-or-...`) before running the pipeline."
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
    """DeepSeek V4 Flash (via OpenRouter) for the text reasoning nodes."""
    return dspy.LM(
        os.environ.get("TEXT_MODEL", "openrouter/deepseek/deepseek-v4-flash"),
        api_key=_require_key(),
        temperature=0.0,
        max_tokens=8000,
        **_provider_routing(os.environ.get("TEXT_PROVIDER", "DeepSeek")),
    )


@lru_cache(maxsize=1)
def vision_lm() -> dspy.LM:
    """Qwen3-VL-235B (via OpenRouter) for the image-consuming nodes (OCR, image filter)."""
    return dspy.LM(
        os.environ.get("VISION_MODEL", "openrouter/qwen/qwen3-vl-235b-a22b-instruct"),
        api_key=_require_key(),
        temperature=0.0,
        max_tokens=8000,
        **_provider_routing(os.environ.get("VISION_PROVIDER", "")),
    )
