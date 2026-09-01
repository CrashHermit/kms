"""Central KMS configuration.

Settings load once from ``config.toml`` (repo root) with
environment-variable overrides. Precedence, highest first:

1. Environment variables (``KMS_MODELS__MODULES__FORMATTER__MODEL``, ...).
2. The repo-root ``.env`` file.
3. ``config.toml`` (repo root, or the path in ``KMS_CONFIG``).
4. Code defaults on each field.

Secrets (API keys, the database password) live only in the environment
(``.env`` or exported), never in the committed ``config.toml``.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / 'config.toml'
_DEFAULT_ENV = _DEFAULT_CONFIG.parent / '.env'
load_dotenv(_DEFAULT_ENV)


class _ConfigModel(BaseModel):
    """Base for config models: reject unknown keys in the TOML."""

    model_config = ConfigDict(extra='forbid')


class LMModel(_ConfigModel):
    """One module's endpoint, model id, gateway, and sampling knobs."""

    base_url: str = ''
    api_key: str = ''
    api_key_source: str = 'deepseek'
    model: str = ''
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=8192, gt=0)


class ProviderLMModel(LMModel):
    """An LLM role that may pin an OpenRouter provider."""

    provider: str = ''


class ModelsConfig(_ConfigModel):
    """Per-module LLM settings plus provider API keys."""

    modules: dict[str, ProviderLMModel] = Field(default_factory=dict)
    openrouter_api_key: str = ''
    deepseek_api_key: str = ''


class ModelPreset(_ConfigModel):
    """One entry in the llama-server models-preset INI."""

    model: str = ''
    mmproj: str = ''
    ctx_size: int = Field(default=32768, gt=0)
    cache_type_k: str = ''
    cache_type_v: str = ''
    n_gpu_layers: int = 999
    no_warmup: bool = True
    parallel: int = 1
    flash_attn: Literal['on', 'off', 'auto'] = 'auto'
    reasoning: str = ''
    temperature: float | None = None
    threads: int = 0


class DedicatedRetrievalServerConfig(_ConfigModel):
    """Configuration for one dedicated local retrieval server."""

    manage: bool = True
    host: str = '127.0.0.1'
    port: int = Field(..., ge=1, le=65535)
    model: str
    threads: int = Field(default=12, gt=0)
    threads_batch: int = Field(default=12, gt=0)
    ubatch_size: int = Field(default=512, gt=0)
    ctx_size: int = Field(..., gt=0)
    parallel: int = Field(default=1, gt=0)
    n_gpu_layers: int = 0
    device: str = 'none'
    cache_ram: int = Field(default=0, ge=0)
    ready_timeout: float = Field(default=300.0, gt=0.0)
    poll_interval: float = Field(default=1.0, gt=0.0)
    request_timeout: float = Field(default=30.0, gt=0.0)
    terminate_timeout: float = Field(default=15.0, gt=0.0)


class RetrievalServingConfig(_ConfigModel):
    """Dedicated embedding and reranking server settings."""

    embedding: DedicatedRetrievalServerConfig = Field(
        default_factory=lambda: DedicatedRetrievalServerConfig(
            port=8081,
            model='~/models/qwen3-embedding-8b/Qwen3-Embedding-8B-Q4_K_M.gguf',
            ctx_size=32768,
        )
    )
    reranker: DedicatedRetrievalServerConfig = Field(
        default_factory=lambda: DedicatedRetrievalServerConfig(
            port=8082,
            model='~/models/qwen3-reranker-4b/Qwen3-Reranker-4B-Q4_K_M.gguf',
            ctx_size=8192,
        )
    )


class ServingConfig(_ConfigModel):
    """The router-mode llama-server and per-module model IDs."""

    manage: bool = False
    host: str = '127.0.0.1'
    port: int = Field(default=8080, ge=1, le=65535)
    module_models: dict[str, str] = Field(default_factory=dict)
    max_loaded_models: int = Field(default=1, gt=0)
    ready_timeout: float = Field(default=300.0, gt=0.0)
    poll_interval: float = Field(default=1.0, gt=0.0)
    request_timeout: float = Field(default=30.0, gt=0.0)
    terminate_timeout: float = Field(default=15.0, gt=0.0)
    presets: dict[str, ModelPreset] = Field(default_factory=dict)
    retrieval: RetrievalServingConfig = Field(
        default_factory=RetrievalServingConfig
    )


class EmbeddingsConfig(_ConfigModel):
    """OpenAI-compatible local embedding client settings."""

    model: str = 'Qwen3-Embedding-8B'
    dimension: int = Field(default=4096, gt=0)
    batch_size: int = Field(default=32, gt=0)
    base_url: str = 'http://127.0.0.1:8081/v1'
    timeout_seconds: float = Field(default=120.0, gt=0.0)


class RerankerConfig(_ConfigModel):
    """OpenAI-compatible local reranking client settings."""

    model: str = 'Qwen3-Reranker-4B'
    batch_size: int = Field(default=8, gt=0)
    base_url: str = 'http://127.0.0.1:8082/v1'
    timeout_seconds: float = Field(default=120.0, gt=0.0)


class OCRConfig(_ConfigModel):
    """Mistral OCR service settings."""

    api_key: str = ''
    model: str = 'mistral-ocr-latest'
    url: str = 'https://api.mistral.ai/v1/ocr'
    render_scale: float = Field(default=1.0, gt=0.0)
    block_crop_scale: float = Field(default=1.0, gt=0.0)
    reference_annotation_prompt: str = Field(
        default='Scan the entire document for explicit references to figures, equations, '
        'tables, sections, theorems, definitions, examples, or other numbered '
        'items. Return each reference exactly as printed, with its surrounding '
        'sentence and a short description of the referenced target when visible. '
        'Do not infer references that are not explicitly present.'
    )


class DatabaseConfig(_ConfigModel):
    """Neo4j connection settings."""

    uri: str = ''
    username: str = ''
    password: str = ''
    database: str = 'neo4j'
    transport: Literal['auto', 'bolt', 'http'] = 'auto'
    http_url: str = ''
    http_timeout: float = Field(default=60.0, gt=0.0)
    bolt_probe_timeout: float = Field(default=5.0, gt=0.0)
    max_connection_lifetime: float = Field(default=300.0, gt=0.0)
    http_port: int = Field(default=7474, ge=1, le=65535)
    https_port: int = Field(default=7473, ge=1, le=65535)

    @field_validator('transport', mode='before')
    @classmethod
    def _normalize_transport(cls, value: str) -> str:
        return value.lower() if isinstance(value, str) else value


class ImageConfig(_ConfigModel):
    """Image preprocessing settings."""

    max_dim: int = Field(default=768, gt=0)


class ConcurrencyConfig(_ConfigModel):
    """Concurrency limits for independent module calls."""

    max_concurrent_calls: int = Field(default=16, gt=0)
    recursion_limit: int = Field(default=1000, gt=0)


class RecordingConfig(_ConfigModel):
    """LLM example recording settings."""

    enabled: bool = False


class SplitterConfig(_ConfigModel):
    """Splitter context-window budgets."""

    backward_context_budget: int = Field(default=200, ge=0)
    lookahead_budget: int = Field(default=2000, gt=0)


class InstructionFinderConfig(_ConfigModel):
    """Instruction finder context-window budgets."""

    context_budget: int = Field(default=300, gt=0)
    max_span_budget: int = Field(default=8000, gt=0)


class PedagogicalFinderConfig(_ConfigModel):
    """Pedagogical finder router context-window budgets."""

    start_before_budget: int = Field(default=300, ge=0)
    start_after_budget: int = Field(default=300, ge=0)
    end_before_budget: int = Field(default=300, ge=0)
    end_after_budget: int = Field(default=300, ge=0)


class FindersConfig(_ConfigModel):
    """Instruction and pedagogical finder budgets."""

    instruction_finder: InstructionFinderConfig = Field(
        default_factory=InstructionFinderConfig
    )
    pedagogical: PedagogicalFinderConfig = Field(
        default_factory=PedagogicalFinderConfig
    )


class EnrichmentConfig(_ConfigModel):
    """Semantic enrichment window and concurrency settings."""

    before_budget: int = Field(default=200, ge=0)
    after_budget: int = Field(default=200, ge=0)
    max_concurrent_calls: int = Field(default=16, gt=0)


class TripletConfig(_ConfigModel):
    """Triplet extraction window budgets."""

    window_budget: int = Field(default=600, gt=0)
    backward_context_budget: int = Field(default=400, ge=0)
    forward_context_budget: int = Field(default=400, ge=0)


class GovernanceConfig(_ConfigModel):
    """Governance static-context settings."""

    backward_context_budget: int = Field(default=200, ge=0)
    forward_context_budget: int = Field(default=500, ge=0)
    max_concurrent_calls: int = Field(default=8, gt=0)


class HubConfig(_ConfigModel):
    """Semantic hub clustering thresholds."""

    recall_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    merge_above: float = Field(default=0.85, ge=0.0, le=1.0)
    separate_below: float = Field(default=0.35, ge=0.0, le=1.0)
    max_concurrent_calls: int = Field(default=16, gt=0)
    comparison_token_budget: int = Field(default=4096, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)
    rerank_candidate_limit: int = Field(default=32, gt=0)


class TripletHubConfig(_ConfigModel):
    """Triplet hub synthesis settings."""

    max_concurrent_calls: int = Field(default=16, gt=0)


class SearchConfig(_ConfigModel):
    """Search retrieval settings."""

    top_k: int = Field(default=20, gt=0)
    rerank_top_n: int = Field(default=5, gt=0)


class ProcedureConfig(_ConfigModel):
    """Procedure retrieval settings."""

    entity_definition_top_k: int = Field(default=10, gt=0)


class CardCreationConfig(_ConfigModel):
    """Hub-scoped component card creation settings."""

    max_concurrent_calls: int = Field(default=16, gt=0)


class StagesConfig(_ConfigModel):
    """Settings for module-level processing stages."""

    splitter: SplitterConfig = Field(default_factory=SplitterConfig)
    finders: FindersConfig = Field(default_factory=FindersConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    image_enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    entity_enrichment: EnrichmentConfig = Field(
        default_factory=EnrichmentConfig
    )
    event_enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    predicate_enrichment: EnrichmentConfig = Field(
        default_factory=EnrichmentConfig
    )
    triplet: TripletConfig = Field(default_factory=TripletConfig)
    entity_hubs: HubConfig = Field(default_factory=HubConfig)
    predicate_hubs: HubConfig = Field(default_factory=HubConfig)
    triplet_hubs: TripletHubConfig = Field(default_factory=TripletHubConfig)
    statement_enrichment: EnrichmentConfig = Field(
        default_factory=EnrichmentConfig
    )
    event_hubs: HubConfig = Field(default_factory=HubConfig)
    procedure_enrichment: EnrichmentConfig = Field(
        default_factory=EnrichmentConfig
    )
    search: SearchConfig = Field(default_factory=SearchConfig)
    procedure: ProcedureConfig = Field(default_factory=ProcedureConfig)
    statement_hubs: HubConfig = Field(default_factory=HubConfig)
    procedure_hubs: HubConfig = Field(default_factory=HubConfig)
    card_creation: CardCreationConfig = Field(
        default_factory=CardCreationConfig
    )


class Settings(BaseSettings):
    """Root settings object, loaded from TOML plus environment."""

    model_config = SettingsConfigDict(
        env_prefix='KMS_',
        env_nested_delimiter='__',
        extra='forbid',
    )

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    concurrency: ConcurrencyConfig = Field(default_factory=ConcurrencyConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)

    @model_validator(mode='after')
    def validate_local_model_configuration(self):
        """Validates local module routing and context budgets."""
        for module_name, module in self.models.modules.items():
            if not module.base_url:
                continue
            preset_name = self.serving.module_models.get(module_name)
            if self.serving.manage and not preset_name:
                raise ValueError(
                    f'local module {module_name!r} has no serving model mapping'
                )
            if not preset_name:
                continue
            preset = self.serving.presets.get(preset_name)
            if preset is None:
                raise ValueError(
                    f'local module {module_name!r} references missing '
                    f'serving preset {preset_name!r}'
                )
            model_name = module.model.removeprefix('openai/')
            if model_name != preset_name:
                raise ValueError(
                    f'local module {module_name!r} model {model_name!r} '
                    f'does not match serving preset {preset_name!r}'
                )
            if module.max_tokens >= preset.ctx_size:
                raise ValueError(
                    f'local module {module_name!r} max_tokens '
                    f'({module.max_tokens}) must be less than '
                    f'{preset_name!r} ctx_size ({preset.ctx_size})'
                )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Orders settings sources: init, env, TOML, then file secrets."""
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=_config_path()),
            file_secret_settings,
        )


def _config_path() -> str:
    """Returns the config.toml path, overridable via ``KMS_CONFIG``."""
    return os.environ.get('KMS_CONFIG') or str(_DEFAULT_CONFIG)


def load_settings() -> Settings:
    """Builds a fresh Settings from env, .env, and config.toml."""
    return Settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the shared Settings, building it once."""
    return load_settings()
