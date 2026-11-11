import pytest
from pydantic import ValidationError

from kms.config.inference import TextInferenceSettings, VisionInferenceSettings
from kms.config.llm import ChatGPTLLMProfile, LLMSettings, LocalLLMProfile
from kms.config.settings import Settings

_SOURCE_PROCESSING_ENVIRONMENT = {
    'KMS_SOURCE_PROCESSING__INSTRUCTION_FINDER__START_ROUTER__PROFILE': (
        'custom-instruction-start-profile'
    ),
    'KMS_SOURCE_PROCESSING__PEDAGOGICAL_FINDER__START_ROUTER__PROFILE': (
        'custom-pedagogical-start-profile'
    ),
    'KMS_SOURCE_PROCESSING__PEDAGOGICAL_FINDER__BOUNDARY_ROUTER__PROFILE': (
        'custom-pedagogical-boundary-profile'
    ),
    'KMS_SOURCE_PROCESSING__STATEMENT_PROCEDURE__ROLE_TYPER__PROFILE': (
        'custom-role-profile'
    ),
    'KMS_SOURCE_PROCESSING__STATEMENT_PROCEDURE__STATEMENT_PARTITIONER__PROFILE': (
        'custom-statement-profile'
    ),
    'KMS_SOURCE_PROCESSING__STATEMENT_PROCEDURE__PROCEDURE_PARTITIONER__PROFILE': (
        'custom-procedure-profile'
    ),
    'KMS_SOURCE_PROCESSING__EXERCISE_FINDER__START_ROUTER__PROFILE': (
        'custom-exercise-start-profile'
    ),
    'KMS_SOURCE_PROCESSING__EXERCISE_FINDER__BOUNDARY_ROUTER__PROFILE': (
        'custom-exercise-boundary-profile'
    ),
}


def test_kms_defaults_include_source_processing_stages(monkeypatch):
    for name in (
        *_SOURCE_PROCESSING_ENVIRONMENT,
        'KMS_EMBEDDING__API_KEY',
        'KMS_EMBEDDING__MODEL',
        'KMS_EMBEDDING__BATCH_TARGET_TOKENS',
    ):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    zero_context_windows = [
        settings.source_processing.instruction_finder.start_context_window,
        settings.source_processing.exercise_finder.start_context_window,
        settings.source_processing.exercise_finder.boundary_context_window,
    ]
    assert all(
        window.backward_budget == 0 and window.forward_budget == 0
        for window in zero_context_windows
    )
    instruction_boundary_window = (
        settings.source_processing.instruction_finder.boundary_context_window
    )
    assert instruction_boundary_window.backward_budget == 300
    assert instruction_boundary_window.forward_budget == 0
    context_windows = [
        settings.source_processing.pedagogical_finder.start_context_window,
        settings.source_processing.pedagogical_finder.boundary_context_window,
    ]
    assert all(
        window.backward_budget == 300 and window.forward_budget == 300
        for window in context_windows
    )
    assert (
        settings.source_processing.instruction_governance.context_window.forward_budget
        == 500
    )


def test_content_correction_settings_use_nested_inference_and_budgets(
    monkeypatch,
):
    names = (
        'KMS_SOURCE_PROCESSING__CONTENT_CORRECTION__INFERENCE__PROFILE',
        'KMS_SOURCE_PROCESSING__CONTENT_CORRECTION__BATCH_SIZE',
        'KMS_SOURCE_PROCESSING__CONTENT_CORRECTION__INPUT_TOKEN_BUDGET',
        'KMS_SOURCE_PROCESSING__CONTENT_CORRECTION__SAFETY_MARGIN_TOKENS',
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
        'KMS_SOURCE_PROCESSING__IMAGE_SEAM__INFERENCE__PROFILE',
        'KMS_SOURCE_PROCESSING__IMAGE_SEAM__BATCH_SIZE',
        'KMS_SOURCE_PROCESSING__IMAGE_SEAM__INPUT_TOKEN_BUDGET',
        'KMS_SOURCE_PROCESSING__IMAGE_SEAM__SAFETY_MARGIN_TOKENS',
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


def test_kms_source_processing_includes_pedagogical_nested_environment(
    monkeypatch,
):
    for name, value in _SOURCE_PROCESSING_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    settings = Settings()

    assert (
        settings.source_processing.instruction_finder.start_router.profile
        == ('custom-instruction-start-profile')
    )
    assert (
        settings.source_processing.pedagogical_finder.start_router.profile
        == ('custom-pedagogical-start-profile')
    )
    assert (
        settings.source_processing.pedagogical_finder.boundary_router.profile
        == ('custom-pedagogical-boundary-profile')
    )
    assert (
        settings.source_processing.statement_procedure.role_typer.profile
        == ('custom-role-profile')
    )
    assert (
        settings.source_processing.statement_procedure.statement_partitioner.profile
        == 'custom-statement-profile'
    )
    assert (
        settings.source_processing.statement_procedure.procedure_partitioner.profile
        == 'custom-procedure-profile'
    )
    assert (
        settings.source_processing.exercise_finder.start_router.profile
        == 'custom-exercise-start-profile'
    )
    assert (
        settings.source_processing.exercise_finder.boundary_router.profile
        == 'custom-exercise-boundary-profile'
    )


def test_kms_training_examples_are_opt_in(monkeypatch):
    monkeypatch.delenv(
        'KMS_TRAINING__EXAMPLES_DIRECTORY',
        raising=False,
    )

    settings = Settings()

    assert settings.training.examples_directory is None


def test_kms_training_examples_use_nested_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(
        'KMS_TRAINING__EXAMPLES_DIRECTORY',
        str(tmp_path),
    )

    settings = Settings()

    assert settings.training.examples_directory == tmp_path


def test_explicit_stage_selection_overrides_subscription_default(monkeypatch):
    monkeypatch.setenv('KMS_LLM__DEFAULT_TEXT_PROFILE', 'chatgpt-luna')
    monkeypatch.setenv(
        'KMS_SOURCE_PROCESSING__FORMATTING__INFERENCE__PROFILE',
        'gemma-text-32k',
    )
    settings = Settings(_env_file=None)
    entity = settings.llm.resolve(
        settings.source_processing.formatting.inference
    )
    assert isinstance(entity, LocalLLMProfile)
    assert entity.model_server_profile == 'gemma-text-32k'


def test_subscription_profile_uses_typed_nested_environment(monkeypatch):
    monkeypatch.setenv('KMS_LLM__DEFAULT_TEXT_PROFILE', 'chatgpt-luna')
    monkeypatch.setenv(
        'KMS_LLM__PROFILES__CHATGPT-LUNA__REASONING_EFFORT', 'medium'
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


def test_kms_source_search_limits_use_nested_environment(monkeypatch):
    monkeypatch.setenv('KMS_SOURCE_SEARCH__CANDIDATE_LIMIT', '41')
    monkeypatch.setenv('KMS_SOURCE_SEARCH__RESULT_LIMIT', '6')

    settings = Settings(_env_file=None)

    assert settings.source_search.candidate_limit == 41
    assert settings.source_search.result_limit == 6


def test_kms_source_search_limits_have_defaults(monkeypatch):
    monkeypatch.delenv('KMS_SOURCE_SEARCH__CANDIDATE_LIMIT', raising=False)
    monkeypatch.delenv('KMS_SOURCE_SEARCH__RESULT_LIMIT', raising=False)

    settings = Settings(_env_file=None)

    assert settings.source_search.candidate_limit == 32
    assert settings.source_search.result_limit == 8


def test_discovery_and_capture_settings_use_nested_environment(monkeypatch):
    monkeypatch.delenv('KMS_SOURCE_DISCOVERY__TAVILY__API_KEY', raising=False)
    defaults = Settings(_env_file=None)
    assert defaults.source_discovery.tavily.api_key.get_secret_value() == ''
    assert defaults.source_discovery.inference.num_retries == 0
    assert isinstance(
        defaults.source_discovery.inference, TextInferenceSettings
    )

    monkeypatch.setenv('KMS_SOURCE_DISCOVERY__TAVILY__API_KEY', 'test-secret')
    monkeypatch.setenv(
        'KMS_SOURCE_DISCOVERY__INFERENCE__PROFILE', 'text-research'
    )
    monkeypatch.setenv('KMS_SOURCE_DISCOVERY__TAVILY__RESULT_LIMIT', '7')
    monkeypatch.setenv('KMS_SOURCE_CAPTURE__TIMEOUT_SECONDS', '23')
    configured = Settings(_env_file=None)
    assert (
        configured.source_discovery.tavily.api_key.get_secret_value()
        == 'test-secret'
    )
    assert configured.source_discovery.inference.profile == 'text-research'
    assert configured.source_discovery.tavily.result_limit == 7
    assert configured.source_capture.timeout_seconds == 23


@pytest.mark.parametrize('root', ['source_discovery', 'source_capture'])
def test_source_capability_settings_reject_unknown_fields(root):
    with pytest.raises(ValidationError, match='extra_forbidden'):
        Settings.model_validate({root: {'unknown': True}})


def test_embedding_batching_uses_nested_environment(monkeypatch):
    monkeypatch.setenv('KMS_EMBEDDING__BATCH_TARGET_ITEMS', '2000')
    monkeypatch.setenv('KMS_EMBEDDING__BATCH_TARGET_TOKENS', '400000')
    settings = Settings(_env_file=None)
    assert settings.embedding.batch_target_items == 2000
    assert settings.embedding.batch_target_tokens == 400000
