"""Shared helpers for manually run live tests."""

from kms import config
from kms.core import llm


def configured_module_model(module_name: str) -> str:
    """Returns the configured model for a live-test module."""
    module = config.load_settings().models.modules[module_name]
    return module.model


def clear_cached_models() -> None:
    """Clears cached LMs after an external environment change."""
    llm.module_lm.cache_clear()
