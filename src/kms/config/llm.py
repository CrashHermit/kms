"""Named local and ChatGPT subscription inference profiles."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kms.config.inference import StageInferenceSettings


class LocalLLMProfile(BaseModel):
    """Select a separately configured llama.cpp server profile."""

    model_config = ConfigDict(extra='forbid')

    backend: Literal['local'] = 'local'
    model_server_profile: str = Field(min_length=1)


class ChatGPTLLMProfile(BaseModel):
    """Configure subscription inference without local sampling parameters.

    Tokenizer counts are estimates using the configured encoding, not native
    server counts. Completion reserves are budgeting inputs: the subscription
    backend does not enforce client-supplied output-token limits.
    """

    model_config = ConfigDict(extra='forbid')

    backend: Literal['chatgpt'] = 'chatgpt'
    model: str = Field(default='gpt-6-luna', min_length=1)
    reasoning_effort: Literal[
        'none', 'low', 'medium', 'high', 'xhigh', 'max'
    ] = 'low'
    context_size: int = Field(default=1_050_000, gt=0)
    max_input_tokens: int = Field(default=922_000, gt=0)
    tokenizer_encoding: str = Field(default='o200k_base', min_length=1)


LLMProfile = Annotated[
    LocalLLMProfile | ChatGPTLLMProfile, Field(discriminator='backend')
]


class LLMSettings(BaseModel):
    """Resolve stage overrides against one application text default."""

    model_config = ConfigDict(extra='forbid')
    max_concurrency: int = Field(default=4, gt=0)

    default_text_profile: str = Field(default='chatgpt-luna', min_length=1)
    profiles: dict[str, LLMProfile] = Field(
        default_factory=lambda: {
            **{
                name: LocalLLMProfile(model_server_profile=name)
                for name in (
                    'gemma-text-32k',
                    'gemma-4-e2b-text',
                    'gemma-vision-8k',
                    'qwen3.8-9b-distill-text',
                )
            },
            'chatgpt-luna': ChatGPTLLMProfile(),
        }
    )

    @model_validator(mode='after')
    def validate_default_profile(self) -> LLMSettings:
        """Reject an unknown application default before starting inference."""
        if self.default_text_profile not in self.profiles:
            raise ValueError(
                'default_text_profile must name a configured LLM profile'
            )
        return self

    def profile_name(self, inference: StageInferenceSettings) -> str:
        """Return an explicit stage selection or the application text default."""
        return inference.profile or self.default_text_profile

    def resolve(self, inference: StageInferenceSettings) -> LLMProfile:
        """Resolve a profile; unknown explicit selections are configuration errors."""
        name = self.profile_name(inference)
        if name not in self.profiles:
            raise ValueError(f'Unknown LLM profile: {name}')
        return self.profiles[name]
