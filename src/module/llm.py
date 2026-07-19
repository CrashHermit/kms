"""
Central LLM configuration for the DSPy modules.

Every DSPy module in this package delegates to a shared language model built
here, so the model choice, credentials, and generation settings live in one place
instead of being duplicated across nodes.

Two backends are used:

- Text reasoning nodes (extractor, seam merger, exercise/instruction refiners,
  instruction distributor) run on DeepSeek's chat model over its
  OpenAI-compatible API. The API key is read from the DEEPSEEK_API_KEY
  environment variable — never hard-code it.
- Vision nodes (OCR, image filter) send images, which DeepSeek's chat model does
  not accept, so they run on the local Granite Vision server started by serve.sh
  (an OpenAI-compatible endpoint). Its location defaults to localhost:8080 and is
  overridable through the environment.

LM objects are cached so every module that asks for the same backend shares one
instance (and therefore one connection pool and prompt cache).
"""

import os
from functools import lru_cache

import dspy

DEEPSEEK_ENV_KEY = "DEEPSEEK_API_KEY"


@lru_cache(maxsize=1)
def text_lm() -> dspy.LM:
    """DeepSeek chat model for the text reasoning nodes.

    The API key is read from the ``DEEPSEEK_API_KEY`` environment variable. We
    raise here (rather than at import time) so the modules stay importable without
    credentials — the key is only required once a node actually runs.
    """
    api_key = os.environ.get(DEEPSEEK_ENV_KEY)
    if not api_key:
        raise RuntimeError(
            f"{DEEPSEEK_ENV_KEY} is not set. Export your DeepSeek API key "
            f"(e.g. `export {DEEPSEEK_ENV_KEY}=sk-...`) before running the pipeline."
        )
    return dspy.LM(
        os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-chat"),
        api_key=api_key,
        temperature=0.0,
        max_tokens=8000,
    )


@lru_cache(maxsize=1)
def vision_lm() -> dspy.LM:
    """Local Granite Vision server for the image-consuming nodes (OCR, image filter).

    DeepSeek's chat API is text-only, so these nodes target the OpenAI-compatible
    ``llama-server`` that serve.sh launches. Every setting is overridable via the
    environment; the defaults match serve.sh (localhost:8080).
    """
    return dspy.LM(
        os.environ.get("VISION_MODEL", "openai/granite-vision"),
        api_base=os.environ.get("VISION_API_BASE", "http://localhost:8080/v1"),
        api_key=os.environ.get("VISION_API_KEY", "not-needed"),
        temperature=0.0,
        max_tokens=8000,
    )
