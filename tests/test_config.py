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
    assert settings.stages.entity_enrichment.before_budget == 200
    assert settings.stages.statement_hubs.max_concurrent_calls == 16
    assert settings.stages.statement_hubs.recall_threshold == 0.55
    assert settings.serving.module_models['entity_enrichment'] == 'qwen3.5-9b'
    assert settings.models.modules['extractor'].max_tokens == 8192
    assert settings.models.modules['statement_enrichment'].model == (
        'openai/qwen3.5-9b'
    )
    assert settings.models.modules['procedure_enrichment'].model == (
        'openai/qwen3.5-9b'
    )
    assert settings.serving.module_models['statement_enrichment'] == (
        'qwen3.5-9b'
    )
    assert settings.serving.module_models['procedure_enrichment'] == (
        'qwen3.5-9b'
    )
    assert settings.serving.module_models['procedure_creator'] == ('qwen3.5-9b')


def test_statement_hub_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv('KMS_STAGES__STATEMENT_HUBS__MAX_CONCURRENT_CALLS', '5')
    assert (
        config.load_settings().stages.statement_hubs.max_concurrent_calls == 5
    )


def test_presets_are_loaded_from_toml():
    presets = config.load_settings().serving.presets
    assert 'qwen3.5-9b' in presets
    assert presets['qwen3.5-9b'].ctx_size == 32768
    assert presets['qwen3.5-9b'].reasoning == 'off'


def test_local_model_budget_must_fit_server_context():
    with pytest.raises(ValidationError, match='must be less than'):
        config.Settings(
            models={
                'modules': {
                    'extractor': {
                        'base_url': 'http://127.0.0.1:8080/v1',
                        'model': 'openai/qwen3.5-9b',
                        'max_tokens': 32768,
                    }
                }
            },
            serving={
                'module_models': {'extractor': 'qwen3.5-9b'},
                'presets': {'qwen3.5-9b': {'ctx_size': 32768}},
            },
        )


def test_unknown_toml_or_kwarg_is_rejected():
    with pytest.raises(ValidationError):
        config.Settings(bogus_key=1)


def test_invalid_operational_values_are_rejected():
    with pytest.raises(ValidationError):
        config.Settings(embeddings={'dimension': 0})
    with pytest.raises(ValidationError):
        config.Settings(database={'transport': 'socket'})
