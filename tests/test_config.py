import pytest
from pydantic import ValidationError

from kms import config


def test_nested_env_var_overrides_toml_default(monkeypatch):
    monkeypatch.setenv('KMS_CONCURRENCY__MAX_CONCURRENT_CALLS', '7')
    assert config.load_settings().concurrency.max_concurrent_calls == 7


def test_toml_provides_the_defaults():
    settings = config.load_settings()
    assert settings.embeddings.model == 'voyage-multimodal-3.5'
    assert settings.embeddings.dimension == 1024
    assert settings.concurrency.recursion_limit == 1000
    assert settings.serving.port == 8080
    assert settings.serving.max_loaded_models == 1
    assert settings.stages.search.rerank_top_n == 5
    assert settings.stages.procedure.entity_definition_top_k == 10
    assert settings.stages.component_enrichment.before_budget == 200
    assert (
        settings.serving.module_models['component_enrichment'] == 'qwen3.5-9b'
    )


def test_presets_are_loaded_from_toml():
    presets = config.load_settings().serving.presets
    assert 'qwen3.5-9b' in presets
    assert presets['qwen3.5-9b'].reasoning == 'off'


def test_unknown_toml_or_kwarg_is_rejected():
    with pytest.raises(ValidationError):
        config.Settings(bogus_key=1)


def test_invalid_operational_values_are_rejected():
    with pytest.raises(ValidationError):
        config.Settings(embeddings={'dimension': 0})
    with pytest.raises(ValidationError):
        config.Settings(database={'transport': 'socket'})
