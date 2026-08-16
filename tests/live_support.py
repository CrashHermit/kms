"""Shared configuration for manually run live tests."""

import os


def configure_local_formatter() -> None:
    """Configures the local formatter model used by live tests."""
    os.environ.update(
        {
            'KMS_MODELS__MODULES__FORMATTER__BASE_URL': (
                'http://localhost:8080/v1'
            ),
            'KMS_MODELS__MODULES__FORMATTER__MODEL': (
                'openai/unsloth/gemma-4-e4b-it-GGUF'
            ),
            'KMS_MODELS__MODULES__FORMATTER__API_KEY': 'not-needed',
        }
    )
