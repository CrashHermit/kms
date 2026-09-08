from kms2 import config
from kms2.setting.model.language.base import LanguageModelBase


def test_language_model_provider_options_are_optional_and_preserved():
    language_model = LanguageModelBase(
        model='openai/qwen3.5-9b-text',
        provider_options={
            'timeout': 30,
            'response_format': {'type': 'json_object'},
        },
    )

    assert language_model.provider_options == {
        'timeout': 30,
        'response_format': {'type': 'json_object'},
    }


def test_kms2_ocr_and_mistral_defaults_are_separate():
    settings = config.load_settings()

    assert settings.ocr.output_dir == 'output'
    assert settings.ocr.render_scale == 1.0
    assert settings.ocr.block_crop_scale == 1.0
    assert settings.mistral_ocr.model == 'mistral-ocr-latest'
    assert settings.mistral_ocr.url == 'https://api.mistral.ai/v1/ocr'
    assert not hasattr(settings.ocr, 'api_key')


def test_kms2_mistral_credentials_use_mistral_environment(monkeypatch):
    monkeypatch.setenv('KMS2_MISTRAL_OCR__API_KEY', 'test-key')
    monkeypatch.setenv('KMS2_OCR__OUTPUT_DIR', 'output/kms2')

    settings = config.load_settings()

    assert settings.mistral_ocr.api_key == 'test-key'
    assert settings.ocr.output_dir == 'output/kms2'


def test_kms2_settings_do_not_read_shared_kms_environment(monkeypatch):
    monkeypatch.setenv('KMS_OCR__OUTPUT_DIR', 'output/shared-kms')
    monkeypatch.setenv('KMS_OCR__API_KEY', 'shared-kms-key')

    settings = config.load_settings()

    assert settings.ocr.output_dir == 'output'
    assert settings.mistral_ocr.api_key == ''


def test_kms2_ocr_output_dir_uses_kms2_environment(monkeypatch):
    monkeypatch.setenv('KMS2_OCR__OUTPUT_DIR', 'output/kms2')

    settings = config.load_settings()

    assert settings.ocr.output_dir == 'output/kms2'
