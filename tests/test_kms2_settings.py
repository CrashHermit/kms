from kms2 import config

_SOURCE_MODEL_SERVER_PROFILE_ENVIRONMENT = {
    'KMS2_SOURCE__CONTENT_CORRECTION__MODEL_SERVER_PROFILE': (
        'custom-vision-profile'
    ),
    'KMS2_SOURCE__FORMATTING__MODEL_SERVER_PROFILE': 'custom-text-profile',
    'KMS2_SOURCE__TEXT_SEAM__JUDGE__MODEL_SERVER_PROFILE': (
        'custom-text-judge-profile'
    ),
    'KMS2_SOURCE__TEXT_SEAM__REWRITER__MODEL_SERVER_PROFILE': (
        'custom-text-rewriter-profile'
    ),
    'KMS2_SOURCE__IMAGE_SEAM__MODEL_SERVER_PROFILE': (
        'custom-vision-seam-profile'
    ),
    'KMS2_SOURCE__IMAGE_ENRICHMENT__INFERENCE__MODEL_SERVER_PROFILE': (
        'custom-vision-enrichment-profile'
    ),
    'KMS2_SOURCE__SPLITTER__ROUTER__MODEL_SERVER_PROFILE': (
        'custom-text-router-profile'
    ),
    'KMS2_SOURCE__SPLITTER__SPLITTER__MODEL_SERVER_PROFILE': (
        'custom-text-splitter-profile'
    ),
}
_SOURCE_PREDICT_ENVIRONMENT = {
    'KMS2_SOURCE__CONTENT_CORRECTION__STRATEGY': 'chain_of_thought',
    'KMS2_SOURCE__FORMATTING__STRATEGY': 'predict',
    'KMS2_SOURCE__TEXT_SEAM__JUDGE__STRATEGY': 'chain_of_thought',
    'KMS2_SOURCE__TEXT_SEAM__REWRITER__STRATEGY': 'predict',
    'KMS2_SOURCE__IMAGE_SEAM__STRATEGY': 'chain_of_thought',
    'KMS2_SOURCE__SPLITTER__ROUTER__STRATEGY': 'predict',
    'KMS2_SOURCE__IMAGE_ENRICHMENT__INFERENCE__STRATEGY': 'predict',
    'KMS2_SOURCE__SPLITTER__SPLITTER__STRATEGY': 'predict',
}
_SOURCE_BUDGET_ENVIRONMENT = {
    'KMS2_SOURCE__SPLITTER__CONTEXT_WINDOW__BACKWARD_BUDGET': '11',
    'KMS2_SOURCE__SPLITTER__CONTEXT_WINDOW__FORWARD_BUDGET': '33',
}


def _set_source_environment(monkeypatch) -> None:
    for name, value in (
        *_SOURCE_MODEL_SERVER_PROFILE_ENVIRONMENT.items(),
        *_SOURCE_PREDICT_ENVIRONMENT.items(),
        *_SOURCE_BUDGET_ENVIRONMENT.items(),
    ):
        monkeypatch.setenv(name, value)


def _clear_source_environment(monkeypatch) -> None:
    for name in (
        *_SOURCE_MODEL_SERVER_PROFILE_ENVIRONMENT,
        *_SOURCE_PREDICT_ENVIRONMENT,
        *_SOURCE_BUDGET_ENVIRONMENT,
    ):
        monkeypatch.delenv(name, raising=False)


def test_kms2_defaults_include_local_model_runtime(monkeypatch):
    _clear_source_environment(monkeypatch)
    runtime_settings = config.Settings()

    assert runtime_settings.source.content_correction.model_server_profile == (
        'gemma-vision-8k'
    )
    assert runtime_settings.source.formatting.model_server_profile == (
        'gemma-text-32k'
    )
    assert (
        runtime_settings.source.image_enrichment.inference.model_server_profile
        == 'gemma-vision-8k'
    )
    assert runtime_settings.source.splitter.router.model_server_profile == (
        'gemma-text-32k'
    )
    assert runtime_settings.local_models.router.port == 8080
    assert set(runtime_settings.local_models.router.model_server_profiles) == {
        'gemma-text-32k',
        'gemma-vision-8k',
    }
    assert (
        runtime_settings.local_models.router.model_server_profiles[
            'gemma-vision-8k'
        ].mmproj_path
        == '~/models/gemma-4-e4b-qat/gemma-4-E4B-it-mmproj.gguf'
    )
    assert runtime_settings.local_models.embedding.server.port == 8081
    assert runtime_settings.local_models.embedding.server.ubatch_size == 2048
    assert runtime_settings.local_models.reranker.server.port == 8082
    assert runtime_settings.local_models.reranker.server.ubatch_size == 2048
    assert runtime_settings.local_models.embedding.model.dimension == 4096


def test_kms2_source_uses_nested_environment(monkeypatch):
    _set_source_environment(monkeypatch)
    runtime_settings = config.Settings()

    assert (
        runtime_settings.source.content_correction.model_server_profile
        == 'custom-vision-profile'
    )
    assert (
        runtime_settings.source.content_correction.strategy
        is config.PredictorStrategy.CHAIN_OF_THOUGHT
    )
    assert (
        runtime_settings.source.formatting.model_server_profile
        == 'custom-text-profile'
    )
    assert (
        runtime_settings.source.text_seam.judge.model_server_profile
        == 'custom-text-judge-profile'
    )
    assert (
        runtime_settings.source.text_seam.rewriter.model_server_profile
        == 'custom-text-rewriter-profile'
    )
    assert runtime_settings.source.image_seam.model_server_profile == (
        'custom-vision-seam-profile'
    )
    assert (
        runtime_settings.source.image_enrichment.inference.model_server_profile
        == 'custom-vision-enrichment-profile'
    )
    assert runtime_settings.source.splitter.router.model_server_profile == (
        'custom-text-router-profile'
    )
    assert runtime_settings.source.splitter.splitter.model_server_profile == (
        'custom-text-splitter-profile'
    )
    assert runtime_settings.source.splitter.context_window.backward_budget == 11
    assert runtime_settings.source.splitter.context_window.forward_budget == 33
    assert runtime_settings.source.splitter.context_window.target_budget == 0
    assert runtime_settings.source.formatting.temperature == 0.0
    assert runtime_settings.source.formatting.max_tokens == 8192


def test_kms2_ocr_and_database_use_nested_environment(monkeypatch):
    _set_source_environment(monkeypatch)
    monkeypatch.setenv('KMS2_OCR__API_KEY', 'test-key')
    monkeypatch.setenv('KMS2_OCR__OUTPUT_DIR', 'output/kms2')
    monkeypatch.setenv('KMS2_DATABASE__URI', 'bolt://kms2')
    monkeypatch.setenv('KMS2_DATABASE__USERNAME', 'kms2-user')
    monkeypatch.setenv('KMS2_DATABASE__PASSWORD', 'kms2-password')
    monkeypatch.setenv('KMS2_DATABASE__DATABASE', 'kms2-db')
    monkeypatch.setenv('KMS2_LOCAL_MODELS__ROUTER__PORT', '8090')

    runtime_settings = config.Settings()

    assert runtime_settings.ocr.api_key == 'test-key'
    assert runtime_settings.ocr.output_dir == 'output/kms2'
    assert runtime_settings.database.uri == 'bolt://kms2'
    assert runtime_settings.database.username == 'kms2-user'
    assert runtime_settings.database.password == 'kms2-password'
    assert runtime_settings.database.database == 'kms2-db'
    assert runtime_settings.local_models.router.port == 8090


def test_kms2_settings_do_not_read_shared_kms_environment(monkeypatch):
    _set_source_environment(monkeypatch)
    monkeypatch.setenv('KMS_OCR__OUTPUT_DIR', 'output/shared-kms')
    monkeypatch.setenv('KMS_OCR__API_KEY', 'shared-kms-key')
    monkeypatch.setenv('KMS_DATABASE__URI', 'bolt://shared-kms')
    monkeypatch.setenv('KMS_DATABASE__USERNAME', 'shared-user')
    monkeypatch.setenv('KMS_DATABASE__PASSWORD', 'shared-password')
    monkeypatch.setenv('KMS_DATABASE__DATABASE', 'shared-db')

    runtime_settings = config.Settings()

    assert runtime_settings.ocr.output_dir == 'output'
    assert runtime_settings.ocr.api_key == ''
    assert runtime_settings.database.uri == ''
    assert runtime_settings.database.username == ''
    assert runtime_settings.database.password == ''
    assert runtime_settings.database.database == 'neo4j'


def test_kms2_semantic_settings_default_and_nested_environment(monkeypatch):
    monkeypatch.delenv(
        'KMS2_SEMANTIC__FACT_EXTRACTION__MODEL_SERVER_PROFILE', raising=False
    )
    monkeypatch.delenv(
        'KMS2_SEMANTIC__TRIPLET_DECOMPOSITION__MODEL_SERVER_PROFILE',
        raising=False,
    )
    settings = config.Settings()
    assert settings.semantic.fact_extraction.model_server_profile == (
        'gemma-text-32k'
    )
    assert settings.semantic.fact_extraction.num_retries == 0
    assert settings.semantic.context_window.backward_budget == 400
    assert settings.semantic.context_window.forward_budget == 400

    monkeypatch.setenv(
        'KMS2_SEMANTIC__FACT_EXTRACTION__MODEL_SERVER_PROFILE',
        'custom-fact-profile',
    )
    monkeypatch.setenv(
        'KMS2_SEMANTIC__TRIPLET_DECOMPOSITION__MODEL_SERVER_PROFILE',
        'custom-triplet-profile',
    )
    settings = config.Settings()
    assert settings.semantic.fact_extraction.model_server_profile == (
        'custom-fact-profile'
    )
    assert settings.semantic.triplet_decomposition.model_server_profile == (
        'custom-triplet-profile'
    )

    assert (
        settings.semantic.entity_enrichment.inference.model_server_profile
        == ('gemma-text-32k')
    )
    assert settings.semantic.event_enrichment.inference.num_retries == 0
    assert (
        settings.semantic.predicate_enrichment.context_window.forward_budget
        == 400
    )
    monkeypatch.setenv(
        'KMS2_SEMANTIC__ENTITY_ENRICHMENT__INFERENCE__MODEL_SERVER_PROFILE',
        'custom-entity-profile',
    )
    monkeypatch.setenv(
        'KMS2_SEMANTIC__EVENT_ENRICHMENT__CONTEXT_WINDOW__BACKWARD_BUDGET',
        '17',
    )
    monkeypatch.setenv(
        'KMS2_SEMANTIC__PREDICATE_ENRICHMENT__INFERENCE__MAX_TOKENS',
        '1234',
    )
    settings = config.Settings()
    assert (
        settings.semantic.entity_enrichment.inference.model_server_profile
        == 'custom-entity-profile'
    )
    assert (
        settings.semantic.event_enrichment.context_window.backward_budget == 17
    )
    assert settings.semantic.predicate_enrichment.inference.max_tokens == 1234
