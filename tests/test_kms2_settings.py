import pytest
from pydantic import ValidationError

from kms2.config.inference import TextInferenceSettings, VisionInferenceSettings
from kms2.config.llm import ChatGPTLLMProfile, LLMSettings, LocalLLMProfile
from kms2.config.settings import Settings

_SOURCE_PEDAGOGICAL_ENVIRONMENT = {
    'KMS2_SOURCE_PEDAGOGICAL__INSTRUCTION_FINDER__START_ROUTER__PROFILE': (
        'custom-instruction-start-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__PEDAGOGICAL_FINDER__START_ROUTER__PROFILE': (
        'custom-pedagogical-start-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__PEDAGOGICAL_FINDER__BOUNDARY_ROUTER__PROFILE': (
        'custom-pedagogical-boundary-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__STATEMENT_PROCEDURE__ROLE_TYPER__PROFILE': (
        'custom-role-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__STATEMENT_PROCEDURE__STATEMENT_PARTITIONER__PROFILE': (
        'custom-statement-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__STATEMENT_PROCEDURE__PROCEDURE_PARTITIONER__PROFILE': (
        'custom-procedure-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__EXERCISE_FINDER__START_ROUTER__PROFILE': (
        'custom-exercise-start-profile'
    ),
    'KMS2_SOURCE_PEDAGOGICAL__EXERCISE_FINDER__BOUNDARY_ROUTER__PROFILE': (
        'custom-exercise-boundary-profile'
    ),
}


def test_kms2_defaults_include_source_pedagogical_stages(monkeypatch):
    for name in _SOURCE_PEDAGOGICAL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)

    settings = Settings()
    assert settings.local_models.reranker.model.model_id == 'Qwen3-Reranker-8B'
    assert settings.local_models.reranker.model.model_path == (
        '~/models/qwen3-reranker-8b-verified/Qwen3-Reranker-8B-Q4_K_M.gguf'
    )

    zero_context_windows = [
        settings.source_pedagogical.instruction_finder.start_context_window,
        settings.source_pedagogical.exercise_finder.start_context_window,
        settings.source_pedagogical.exercise_finder.boundary_context_window,
    ]
    assert all(
        window.backward_budget == 0 and window.forward_budget == 0
        for window in zero_context_windows
    )
    instruction_boundary_window = (
        settings.source_pedagogical.instruction_finder.boundary_context_window
    )
    assert instruction_boundary_window.backward_budget == 300
    assert instruction_boundary_window.forward_budget == 0
    context_windows = [
        settings.source_pedagogical.pedagogical_finder.start_context_window,
        settings.source_pedagogical.pedagogical_finder.boundary_context_window,
    ]
    assert all(
        window.backward_budget == 300 and window.forward_budget == 300
        for window in context_windows
    )
    assert (
        settings.source_pedagogical.instruction_governance.context_window.forward_budget
        == 500
    )
    assert settings.source_learning.context_window.backward_budget == 400
    assert settings.source_learning.context_window.forward_budget == 400


def test_content_correction_settings_use_nested_inference_and_budgets(
    monkeypatch,
):
    names = (
        'KMS2_SOURCE_PROCESSING__CONTENT_CORRECTION__INFERENCE__PROFILE',
        'KMS2_SOURCE_PROCESSING__CONTENT_CORRECTION__BATCH_SIZE',
        'KMS2_SOURCE_PROCESSING__CONTENT_CORRECTION__INPUT_TOKEN_BUDGET',
        'KMS2_SOURCE_PROCESSING__CONTENT_CORRECTION__SAFETY_MARGIN_TOKENS',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)
    correction = settings.source_processing.content_correction
    assert isinstance(correction.inference, VisionInferenceSettings)

    for name, value in zip(
        names,
        (
            'local-correction',
            '1',
            '720',
            '32',
        ),
        strict=True,
    ):
        monkeypatch.setenv(name, value)

    correction = Settings(_env_file=None).source_processing.content_correction
    assert correction.inference.profile == 'local-correction'
    assert correction.batch_size == 1
    assert correction.input_token_budget == 720
    assert correction.safety_margin_tokens == 32


def test_image_seam_uses_nested_inference_and_batch_environment(monkeypatch):
    names = (
        'KMS2_SOURCE_PROCESSING__IMAGE_SEAM__INFERENCE__PROFILE',
        'KMS2_SOURCE_PROCESSING__IMAGE_SEAM__BATCH_SIZE',
        'KMS2_SOURCE_PROCESSING__IMAGE_SEAM__INPUT_TOKEN_BUDGET',
        'KMS2_SOURCE_PROCESSING__IMAGE_SEAM__SAFETY_MARGIN_TOKENS',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    defaults = Settings(_env_file=None).source_processing.image_seam
    assert isinstance(defaults.inference, VisionInferenceSettings)
    assert defaults.batch_size == 4
    assert defaults.input_token_budget == 2048
    assert defaults.safety_margin_tokens == 256

    for name, value in zip(
        names,
        ('custom-image-profile', '1', '720', '32'),
        strict=True,
    ):
        monkeypatch.setenv(name, value)

    image_seam = Settings(_env_file=None).source_processing.image_seam
    assert image_seam.inference.profile == 'custom-image-profile'
    assert image_seam.batch_size == 1
    assert image_seam.input_token_budget == 720
    assert image_seam.safety_margin_tokens == 32


def test_kms2_source_pedagogical_uses_nested_environment(monkeypatch):
    for name, value in _SOURCE_PEDAGOGICAL_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    settings = Settings()

    assert (
        settings.source_pedagogical.instruction_finder.start_router.profile
        == ('custom-instruction-start-profile')
    )
    assert (
        settings.source_pedagogical.pedagogical_finder.start_router.profile
        == ('custom-pedagogical-start-profile')
    )
    assert (
        settings.source_pedagogical.pedagogical_finder.boundary_router.profile
        == ('custom-pedagogical-boundary-profile')
    )
    assert (
        settings.source_pedagogical.statement_procedure.role_typer.profile
        == ('custom-role-profile')
    )
    assert (
        settings.source_pedagogical.statement_procedure.statement_partitioner.profile
        == 'custom-statement-profile'
    )
    assert (
        settings.source_pedagogical.statement_procedure.procedure_partitioner.profile
        == 'custom-procedure-profile'
    )
    assert (
        settings.source_pedagogical.exercise_finder.start_router.profile
        == 'custom-exercise-start-profile'
    )
    assert (
        settings.source_pedagogical.exercise_finder.boundary_router.profile
        == 'custom-exercise-boundary-profile'
    )


def test_kms2_source_hub_settings_are_independent(monkeypatch):
    names = (
        'KMS2_SOURCE_SEMANTIC__SOURCE_ENTITY_HUBS__JUDGE__PROFILE',
        'KMS2_SOURCE_SEMANTIC__SOURCE_STATEMENT_HUBS__CANDIDATE_LIMIT',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv(names[0], 'entity-hub-profile')
    monkeypatch.setenv(names[1], '17')
    settings = Settings()

    assert (
        settings.source_semantic.source_entity_hubs.judge.profile
        == 'entity-hub-profile'
    )
    assert settings.source_semantic.source_statement_hubs.candidate_limit == 17


def test_kms2_source_hub_filtering_settings_are_independent(monkeypatch):
    monkeypatch.setenv(
        'KMS2_SOURCE_SEMANTIC__SOURCE_ENTITY_HUBS__JUDGE__PROFILE',
        'entity-judge',
    )
    monkeypatch.setenv(
        'KMS2_RERANKER__BATCH_TOKEN_BUDGET',
        '2048',
    )
    monkeypatch.setenv(
        'KMS2_SOURCE_SEMANTIC__SOURCE_STATEMENT_HUBS__JUDGE_TOKEN_BUDGET',
        '16384',
    )

    settings = Settings()

    assert (
        settings.source_semantic.source_entity_hubs.judge.profile
        == 'entity-judge'
    )
    assert settings.source_semantic.source_entity_hubs.judge.num_retries == 0
    assert settings.reranker.batch_token_budget == 2048
    assert (
        settings.source_semantic.source_statement_hubs.judge_token_budget
        == 16384
    )


def test_kms2_training_examples_are_opt_in(monkeypatch):
    monkeypatch.delenv(
        'KMS2_TRAINING__EXAMPLES_DIRECTORY',
        raising=False,
    )

    settings = Settings()

    assert settings.training.examples_directory is None


def test_kms2_training_examples_use_nested_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(
        'KMS2_TRAINING__EXAMPLES_DIRECTORY',
        str(tmp_path),
    )

    settings = Settings()

    assert settings.training.examples_directory == tmp_path


def test_kms2_entity_processing_defaults_survive_nested_environment(
    monkeypatch,
):
    monkeypatch.setenv(
        'KMS2_SOURCE_SEMANTIC__SOURCE_ENTITY_EXTRACTION__MAX_TOKENS',
        '2048',
    )

    settings = Settings()

    assert settings.source_semantic.source_entity_extraction.max_tokens == 2048
    assert settings.source_semantic.source_entity_extraction.num_retries == 0


def test_explicit_stage_selection_overrides_subscription_default(monkeypatch):
    monkeypatch.setenv('KMS2_LLM__DEFAULT_TEXT_PROFILE', 'chatgpt-luna')
    monkeypatch.setenv(
        'KMS2_SOURCE_SEMANTIC__SOURCE_ENTITY_EXTRACTION__PROFILE',
        'gemma-text-32k',
    )
    settings = Settings(_env_file=None)
    entity = settings.llm.resolve(
        settings.source_semantic.source_entity_extraction
    )
    assert isinstance(entity, LocalLLMProfile)
    assert entity.model_server_profile == 'gemma-text-32k'


def test_subscription_profile_uses_typed_nested_environment(monkeypatch):
    monkeypatch.setenv('KMS2_LLM__DEFAULT_TEXT_PROFILE', 'chatgpt-luna')
    monkeypatch.setenv(
        'KMS2_LLM__PROFILES__CHATGPT-LUNA__REASONING_EFFORT', 'medium'
    )
    settings = Settings(_env_file=None)
    profile = settings.llm.resolve(TextInferenceSettings())
    assert isinstance(profile, ChatGPTLLMProfile)
    assert profile.reasoning_effort == 'medium'


def test_unknown_profile_is_not_a_local_fallback():
    with pytest.raises(ValidationError, match='configured LLM profile'):
        LLMSettings(default_text_profile='missing')
    settings = LLMSettings()
    with pytest.raises(ValueError, match='Unknown LLM profile: missing'):
        settings.resolve(TextInferenceSettings(profile='missing'))
    with pytest.raises(ValidationError):
        TextInferenceSettings(profile='')


def test_old_stage_selection_and_mixed_provider_options_are_rejected():
    with pytest.raises(ValidationError, match='extra_forbidden'):
        TextInferenceSettings.model_validate({'model_server_profile': 'old'})
    with pytest.raises(ValidationError, match='extra_forbidden'):
        ChatGPTLLMProfile.model_validate({'top_k': 64})
    with pytest.raises(ValidationError, match='union_tag_invalid'):
        LLMSettings(
            profiles={'unsupported': {'backend': 'unknown'}},
            default_text_profile='unsupported',
        )


def test_kms2_source_search_limits_use_nested_environment(monkeypatch):
    monkeypatch.setenv('KMS2_SOURCE_SEARCH__CANDIDATE_LIMIT', '41')
    monkeypatch.setenv('KMS2_SOURCE_SEARCH__RESULT_LIMIT', '6')

    settings = Settings(_env_file=None)

    assert settings.source_search.candidate_limit == 41
    assert settings.source_search.result_limit == 6


def test_kms2_source_search_limits_have_defaults(monkeypatch):
    monkeypatch.delenv('KMS2_SOURCE_SEARCH__CANDIDATE_LIMIT', raising=False)
    monkeypatch.delenv('KMS2_SOURCE_SEARCH__RESULT_LIMIT', raising=False)

    settings = Settings(_env_file=None)

    assert settings.source_search.candidate_limit == 32
    assert settings.source_search.result_limit == 8


def test_teaching_settings_use_nested_environment_and_safe_defaults(
    monkeypatch,
):
    for name in (
        'KMS2_TEACHING__INFERENCE__PROFILE',
        'KMS2_TEACHING__INPUT_TOKEN_BUDGET',
        'KMS2_TEACHING__SAFETY_MARGIN_TOKENS',
        'KMS2_TEACHING__MAX_TOOL_CALLS',
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = Settings(_env_file=None).teaching
    assert defaults.inference.profile == 'chatgpt-luna'
    assert defaults.inference.num_retries == 0
    assert defaults.input_token_budget == 16384
    assert defaults.safety_margin_tokens == 256
    assert defaults.max_tool_calls == 8

    monkeypatch.setenv('KMS2_TEACHING__INFERENCE__PROFILE', 'custom-teacher')
    monkeypatch.setenv('KMS2_TEACHING__INPUT_TOKEN_BUDGET', '12000')
    monkeypatch.setenv('KMS2_TEACHING__SAFETY_MARGIN_TOKENS', '128')
    monkeypatch.setenv('KMS2_TEACHING__MAX_TOOL_CALLS', '3')

    configured = Settings(_env_file=None).teaching
    assert configured.inference.profile == 'custom-teacher'
    assert configured.input_token_budget == 12000
    assert configured.safety_margin_tokens == 128
    assert configured.max_tool_calls == 3


def test_research_settings_are_opt_in_and_environment_configurable(monkeypatch):
    for name in (
        'KMS2_RESEARCH__ENABLED',
        'KMS2_RESEARCH__TAVILY__API_KEY',
        'KMS2_RESEARCH__INFERENCE__PROFILE',
        'KMS2_RESEARCH__MAX_CONCURRENCY',
        'KMS2_RESEARCH__SOURCE_DIRECTORY',
    ):
        monkeypatch.delenv(name, raising=False)

    defaults = Settings(_env_file=None).research
    assert not defaults.enabled
    assert defaults.tavily.api_key is None
    assert defaults.inference.num_retries == 0
    assert defaults.formatting.inference.num_retries == 0
    assert defaults.max_concurrency == 4
    assert defaults.max_tool_calls == 12
    assert defaults.source_directory.as_posix() == 'output/research/sources'

    monkeypatch.setenv('KMS2_RESEARCH__ENABLED', 'true')
    monkeypatch.setenv('KMS2_RESEARCH__TAVILY__API_KEY', 'test-secret')
    monkeypatch.setenv(
        'KMS2_RESEARCH__INFERENCE__PROFILE',
        'vision-research',
    )
    monkeypatch.setenv('KMS2_RESEARCH__MAX_CONCURRENCY', '2')
    monkeypatch.setenv(
        'KMS2_RESEARCH__SOURCE_DIRECTORY',
        '/tmp/kms2-research-sources',
    )

    configured = Settings(_env_file=None).research
    assert configured.enabled
    assert configured.tavily.api_key.get_secret_value() == 'test-secret'
    assert configured.inference.profile == 'vision-research'
    assert configured.max_concurrency == 2
    assert (
        configured.source_directory.as_posix() == '/tmp/kms2-research-sources'
    )


def test_research_settings_reject_unknown_fields():
    with pytest.raises(ValidationError, match='extra_forbidden'):
        Settings.model_validate({'research': {'unknown': True}})
