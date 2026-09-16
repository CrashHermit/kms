from kms2 import config

_SOURCE_ENVIRONMENT = {
    'KMS2_SOURCE__INSTRUCTION_FINDER__START_ROUTER__MODEL_SERVER_PROFILE': (
        'custom-instruction-start-profile'
    ),
    'KMS2_SOURCE__PEDAGOGICAL_FINDER__START_ROUTER__MODEL_SERVER_PROFILE': (
        'custom-pedagogical-start-profile'
    ),
    'KMS2_SOURCE__PEDAGOGICAL_FINDER__BOUNDARY_ROUTER__MODEL_SERVER_PROFILE': (
        'custom-pedagogical-boundary-profile'
    ),
    'KMS2_SOURCE__STATEMENT_PROCEDURE__ROLE_TYPER__MODEL_SERVER_PROFILE': (
        'custom-role-profile'
    ),
    'KMS2_SOURCE__STATEMENT_PROCEDURE__STATEMENT_PARTITIONER__MODEL_SERVER_PROFILE': (
        'custom-statement-profile'
    ),
    'KMS2_SOURCE__STATEMENT_PROCEDURE__PROCEDURE_PARTITIONER__MODEL_SERVER_PROFILE': (
        'custom-procedure-profile'
    ),
    'KMS2_SOURCE__EXERCISE_FINDER__START_ROUTER__MODEL_SERVER_PROFILE': (
        'custom-exercise-start-profile'
    ),
    'KMS2_SOURCE__EXERCISE_FINDER__BOUNDARY_ROUTER__MODEL_SERVER_PROFILE': (
        'custom-exercise-boundary-profile'
    ),
}


def test_kms2_defaults_include_pointer_source_stages(monkeypatch):
    for name in _SOURCE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    settings = config.Settings()
    assert settings.local_models.reranker.model.model_id == 'Qwen3-Reranker-8B'
    assert settings.local_models.reranker.model.model_path == (
        '~/models/qwen3-reranker-8b-verified/Qwen3-Reranker-8B-Q4_K_M.gguf'
    )

    zero_context_windows = [
        settings.source.instruction_finder.start_context_window,
        settings.source.exercise_finder.start_context_window,
        settings.source.exercise_finder.boundary_context_window,
    ]
    assert all(
        window.backward_budget == 0 and window.forward_budget == 0
        for window in zero_context_windows
    )
    instruction_boundary_window = (
        settings.source.instruction_finder.boundary_context_window
    )
    assert instruction_boundary_window.backward_budget == 300
    assert instruction_boundary_window.forward_budget == 0
    context_windows = [
        settings.source.pedagogical_finder.start_context_window,
        settings.source.pedagogical_finder.boundary_context_window,
    ]
    assert all(
        window.backward_budget == 300 and window.forward_budget == 300
        for window in context_windows
    )
    assert (
        settings.source.statement_procedure.role_typer.model_server_profile
        == ('gemma-text-32k')
    )
    assert (
        settings.source.exercise_finder.start_router.model_server_profile
        == 'gemma-text-32k'
    )
    assert (
        settings.source.instruction_governance.context_window.forward_budget
        == 500
    )


def test_kms2_source_uses_nested_environment(monkeypatch):
    for name, value in _SOURCE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    settings = config.Settings()

    assert (
        settings.source.instruction_finder.start_router.model_server_profile
        == ('custom-instruction-start-profile')
    )
    assert (
        settings.source.pedagogical_finder.start_router.model_server_profile
        == ('custom-pedagogical-start-profile')
    )
    assert (
        settings.source.pedagogical_finder.boundary_router.model_server_profile
        == ('custom-pedagogical-boundary-profile')
    )
    assert (
        settings.source.statement_procedure.role_typer.model_server_profile
        == ('custom-role-profile')
    )
    assert (
        settings.source.statement_procedure.statement_partitioner.model_server_profile
        == 'custom-statement-profile'
    )
    assert (
        settings.source.statement_procedure.procedure_partitioner.model_server_profile
        == 'custom-procedure-profile'
    )
    assert (
        settings.source.exercise_finder.start_router.model_server_profile
        == 'custom-exercise-start-profile'
    )
    assert (
        settings.source.exercise_finder.boundary_router.model_server_profile
        == 'custom-exercise-boundary-profile'
    )


def test_kms2_source_hub_settings_are_independent(monkeypatch):
    names = (
        'KMS2_SEMANTIC__SOURCE_ENTITY_HUBS__INFERENCE__MODEL_SERVER_PROFILE',
        'KMS2_SEMANTIC__SOURCE_EVENT_HUBS__CANDIDATE_LIMIT',
        'KMS2_SEMANTIC__SOURCE_PREDICATE_HUBS__MINIMUM_SIMILARITY',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    settings = config.Settings()

    assert (
        settings.semantic.source_entity_hubs.inference.model_server_profile
        == ('gemma-text-32k')
    )
    assert settings.semantic.source_event_hubs.candidate_limit == 50
    assert settings.semantic.source_predicate_hubs.minimum_similarity == 0.82

    monkeypatch.setenv(names[0], 'entity-hub-profile')
    monkeypatch.setenv(names[1], '17')
    monkeypatch.setenv(names[2], '0.91')
    settings = config.Settings()

    assert (
        settings.semantic.source_entity_hubs.inference.model_server_profile
        == 'entity-hub-profile'
    )
    assert settings.semantic.source_event_hubs.candidate_limit == 17
    assert settings.semantic.source_predicate_hubs.minimum_similarity == 0.91


def test_kms2_hub_filtering_settings_are_independent(monkeypatch):
    monkeypatch.setenv(
        'KMS2_SEMANTIC__SOURCE_ENTITY_HUBS__JUDGE__MODEL_SERVER_PROFILE',
        'entity-judge',
    )
    monkeypatch.setenv(
        'KMS2_SEMANTIC__SOURCE_EVENT_HUBS__RERANKER_TOKEN_BUDGET',
        '2048',
    )
    monkeypatch.setenv(
        'KMS2_SEMANTIC__SOURCE_PREDICATE_HUBS__JUDGE_TOKEN_BUDGET',
        '16384',
    )

    settings = config.Settings()

    assert settings.semantic.source_entity_hubs.judge.model_server_profile == (
        'entity-judge'
    )
    assert settings.semantic.source_entity_hubs.judge.num_retries == 0
    assert settings.semantic.source_event_hubs.reranker_token_budget == 2048
    assert settings.semantic.source_predicate_hubs.judge_token_budget == 16384
    assert settings.semantic.source_entity_hubs.reranker_token_budget == 4096
    assert settings.semantic.source_entity_hubs.judge_batch_size == 16
    assert settings.semantic.source_event_hubs.judge.model_server_profile == (
        'gemma-text-32k'
    )
