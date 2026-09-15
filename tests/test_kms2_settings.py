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
