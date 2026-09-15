"""Typed runtime configuration for KMS2."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelServerProfileSettings(BaseModel):
    """Load-time settings for one llama.cpp model-server profile."""

    model_config = ConfigDict(extra='forbid')

    model_path: str = Field(min_length=1)
    mmproj_path: str | None = None
    context_size: int = Field(default=32768, gt=0)
    cache_type_k: str = ''
    cache_type_v: str = 'q4_0'
    n_gpu_layers: int = 999
    no_warmup: bool = True
    parallel: int = Field(default=1, gt=0)
    flash_attention: str = 'on'
    reasoning: str = 'off'
    threads: int = Field(default=0, ge=0)


class PredictorStrategy(StrEnum):
    """Available DSPy predictor strategies."""

    PREDICT = 'predict'
    CHAIN_OF_THOUGHT = 'chain_of_thought'


class StageInferenceSettings(BaseModel):
    """Request-time inference settings for one source stage."""

    model_config = ConfigDict(extra='forbid')

    model_server_profile: str = Field(min_length=1)
    strategy: PredictorStrategy = PredictorStrategy.PREDICT
    temperature: float = 0.0
    max_tokens: int = Field(default=8192, gt=0)
    num_retries: int = Field(default=3, ge=0)


class ContextWindowSettings(BaseModel):
    """Estimated token budgets for surrounding source context."""

    model_config = ConfigDict(extra='forbid')

    backward_budget: int | None = None
    forward_budget: int | None = None
    target_budget: int = 0


class ImageDescriptionSettings(BaseModel):
    """Language model and context-window settings for image descriptions."""

    model_config = ConfigDict(extra='forbid')

    inference: StageInferenceSettings
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200,
            forward_budget=200,
        )
    )


class TermDescriptionSettings(BaseModel):
    """Language model and context-window settings for typed descriptions."""

    model_config = ConfigDict(extra='forbid')

    inference: StageInferenceSettings = Field(
        default_factory=lambda: StageInferenceSettings(
            model_server_profile='gemma-text-32k',
            num_retries=0,
        )
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )

    @model_validator(mode='before')
    @classmethod
    def merge_inference_defaults(cls, value: Any) -> Any:
        """Preserve typed defaults when one nested inference field is overridden."""
        if not isinstance(value, dict) or 'inference' not in value:
            return value
        inference = value['inference']
        if not isinstance(inference, dict):
            return value
        defaults = StageInferenceSettings(
            model_server_profile='gemma-text-32k',
            num_retries=0,
        ).model_dump()
        defaults.update(inference)
        return {**value, 'inference': defaults}


class SplitterSettings(BaseModel):
    """Language model and context-window settings for source splitting."""

    model_config = ConfigDict(extra='forbid')

    router: StageInferenceSettings
    splitter: StageInferenceSettings
    context_window: ContextWindowSettings = Field(
        default_factory=ContextWindowSettings
    )


class InstructionFinderSettings(BaseModel):
    """Language model and context-window settings for instruction discovery."""

    model_config = ConfigDict(extra='forbid')

    start_router: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    boundary_router: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300,
            forward_budget=0,
        )
    )


class PedagogicalFinderSettings(BaseModel):
    """Language model and context-window settings for pedagogical discovery."""

    model_config = ConfigDict(extra='forbid')

    start_router: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    boundary_router: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300,
            forward_budget=300,
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=300,
            forward_budget=300,
        )
    )


class StatementProcedureSettings(BaseModel):
    """Language model settings for statement and procedure construction."""

    model_config = ConfigDict(extra='forbid')

    role_typer: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    statement_partitioner: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    procedure_partitioner: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )


class ExerciseFinderSettings(BaseModel):
    """Language model and context-window settings for exercise discovery."""

    model_config = ConfigDict(extra='forbid')

    start_router: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    boundary_router: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    start_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
        )
    )
    boundary_context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=0,
            forward_budget=0,
        )
    )


class InstructionGovernanceSettings(BaseModel):
    """Language model and context-window settings for instruction governance."""

    model_config = ConfigDict(extra='forbid')

    inference: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=200,
            forward_budget=500,
        )
    )


class TextSeamSettings(BaseModel):
    """Language model settings for text seam judging and rewriting."""

    model_config = ConfigDict(extra='forbid')

    judge: StageInferenceSettings
    rewriter: StageInferenceSettings


class SemanticSettings(BaseModel):
    """Language model and context-window settings for semantic extraction."""

    model_config = ConfigDict(extra='forbid')

    fact_extraction: StageInferenceSettings = Field(
        default_factory=lambda: StageInferenceSettings(
            model_server_profile='gemma-text-32k',
            num_retries=0,
        )
    )
    triplet_decomposition: StageInferenceSettings = Field(
        default_factory=lambda: StageInferenceSettings(
            model_server_profile='gemma-text-32k',
            num_retries=0,
        )
    )
    context_window: ContextWindowSettings = Field(
        default_factory=lambda: ContextWindowSettings(
            backward_budget=400,
            forward_budget=400,
        )
    )
    source_entity_description: TermDescriptionSettings = Field(
        default_factory=lambda: TermDescriptionSettings(
            inference=StageInferenceSettings(
                model_server_profile='gemma-text-32k',
                num_retries=0,
            )
        )
    )
    source_event_description: TermDescriptionSettings = Field(
        default_factory=lambda: TermDescriptionSettings(
            inference=StageInferenceSettings(
                model_server_profile='gemma-text-32k',
                num_retries=0,
            )
        )
    )
    source_predicate_description: TermDescriptionSettings = Field(
        default_factory=lambda: TermDescriptionSettings(
            inference=StageInferenceSettings(
                model_server_profile='gemma-text-32k',
                num_retries=0,
            )
        )
    )


class SourceSettings(BaseModel):
    """Settings for the KMS2 source-processing pipeline."""

    model_config = ConfigDict(extra='forbid')

    content_correction: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-vision-8k')
    )
    formatting: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-text-32k')
    )
    text_seam: TextSeamSettings = Field(
        default_factory=lambda: TextSeamSettings(
            judge=_stage_inference('gemma-text-32k'),
            rewriter=_stage_inference('gemma-text-32k'),
        )
    )
    image_seam: StageInferenceSettings = Field(
        default_factory=lambda: _stage_inference('gemma-vision-8k')
    )
    image_description: ImageDescriptionSettings = Field(
        default_factory=lambda: ImageDescriptionSettings(
            inference=_stage_inference('gemma-vision-8k')
        )
    )
    splitter: SplitterSettings = Field(
        default_factory=lambda: SplitterSettings(
            router=_stage_inference('gemma-text-32k'),
            splitter=_stage_inference('gemma-text-32k'),
        )
    )
    instruction_finder: InstructionFinderSettings = Field(
        default_factory=InstructionFinderSettings
    )
    pedagogical_finder: PedagogicalFinderSettings = Field(
        default_factory=PedagogicalFinderSettings
    )
    statement_procedure: StatementProcedureSettings = Field(
        default_factory=StatementProcedureSettings
    )
    exercise_finder: ExerciseFinderSettings = Field(
        default_factory=ExerciseFinderSettings
    )
    instruction_governance: InstructionGovernanceSettings = Field(
        default_factory=InstructionGovernanceSettings
    )


class OCRSettings(BaseModel):
    """Mistral OCR service and artifact materialization settings."""

    model_config = ConfigDict(extra='forbid')

    api_key: str = ''
    model: str = 'mistral-ocr-4-1'
    url: str = 'https://api.mistral.ai/v1/ocr'
    output_dir: str = 'output'
    render_scale: float = Field(default=1.0, gt=0.0)
    block_crop_scale: float = Field(default=1.0, gt=0.0)


class DatabaseSettings(BaseModel):
    """Neo4j connection settings owned by KMS2."""

    model_config = ConfigDict(extra='forbid')

    uri: str = ''
    username: str = ''
    password: str = ''
    database: str = 'neo4j'


class LlamaServerSettings(BaseModel):
    """Shared process and readiness settings for llama.cpp servers."""

    model_config = ConfigDict(extra='forbid')

    executable: str = 'llama-server'
    host: str = '127.0.0.1'
    port: int = Field(gt=0, le=65535)
    ready_timeout: float = Field(default=300.0, gt=0.0)
    poll_interval: float = Field(default=1.0, gt=0.0)
    health_timeout: float = Field(default=30.0, gt=0.0)
    terminate_timeout: float = Field(default=15.0, gt=0.0)


class RouterServerSettings(LlamaServerSettings):
    """llama.cpp router settings and its model-server profiles."""

    port: int = 8080
    model_server_profiles: dict[str, ModelServerProfileSettings] = Field(
        default_factory=lambda: {
            'gemma-text-32k': ModelServerProfileSettings(
                model_path='~/models/gemma-4-e4b-qat/gemma-4-E4B_q4_0-it.gguf',
            ),
            'gemma-vision-8k': ModelServerProfileSettings(
                model_path='~/models/gemma-4-e4b-qat/gemma-4-E4B_q4_0-it.gguf',
                mmproj_path=(
                    '~/models/gemma-4-e4b-qat/gemma-4-E4B-it-mmproj.gguf'
                ),
                context_size=8192,
                cache_type_k='q4_0',
            ),
        }
    )


class DedicatedLlamaServerSettings(LlamaServerSettings):
    """Dedicated llama.cpp server execution settings."""

    device: str = 'CUDA0'
    n_gpu_layers: int = 999
    threads: int = Field(gt=0)
    threads_batch: int = Field(gt=0)
    ubatch_size: int = Field(gt=0)
    context_size: int = Field(gt=0)
    parallel: int = Field(default=1, gt=0)
    cache_ram: int = Field(default=0, ge=-1)
    no_warmup: bool = True


class EmbeddingModelSettings(BaseModel):
    """Embedding model identity and vector dimensions."""

    model_config = ConfigDict(extra='forbid')

    model_id: str = 'Qwen3-Embedding-8B'
    model_path: str = (
        '~/models/qwen3-embedding-8b/Qwen3-Embedding-8B-Q4_K_M.gguf'
    )
    dimension: int = Field(default=4096, gt=0)


class RerankerModelSettings(BaseModel):
    """Reranker model identity and model path."""

    model_config = ConfigDict(extra='forbid')

    model_id: str = 'Qwen3-Reranker-8B'
    model_path: str = '~/models/qwen3-reranker-8b/Qwen3-Reranker-8B.Q4_K_M.gguf'


class EmbeddingSettings(BaseModel):
    """Embedding model, server, batching, and request settings."""

    model_config = ConfigDict(extra='forbid')

    model: EmbeddingModelSettings = Field(
        default_factory=EmbeddingModelSettings
    )
    server: DedicatedLlamaServerSettings = Field(
        default_factory=lambda: DedicatedLlamaServerSettings(
            port=8081,
            threads=12,
            threads_batch=12,
            ubatch_size=2048,
            context_size=4096,
        )
    )
    batch_size: int = Field(default=32, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)


class RerankerSettings(BaseModel):
    """Reranker model, server, batching, and request settings."""

    model_config = ConfigDict(extra='forbid')

    model: RerankerModelSettings = Field(default_factory=RerankerModelSettings)
    server: DedicatedLlamaServerSettings = Field(
        default_factory=lambda: DedicatedLlamaServerSettings(
            port=8082,
            threads=4,
            threads_batch=4,
            ubatch_size=2048,
            context_size=8192,
        )
    )
    batch_size: int = Field(default=8, gt=0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)


class LocalModelRuntimeSettings(BaseModel):
    """Settings for all KMS2-owned local model servers."""

    model_config = ConfigDict(extra='forbid')

    router: RouterServerSettings = Field(
        default_factory=lambda: RouterServerSettings(port=8080)
    )
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


class Settings(BaseSettings):
    """KMS2 runtime settings with environment-variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix='KMS2_',
        env_nested_delimiter='__',
    )

    source: SourceSettings = Field(default_factory=SourceSettings)
    semantic: SemanticSettings = Field(default_factory=SemanticSettings)
    local_models: LocalModelRuntimeSettings = Field(
        default_factory=LocalModelRuntimeSettings
    )
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


def _stage_inference(model_server_profile: str) -> StageInferenceSettings:
    """Build default request-time settings for one model-server profile."""
    return StageInferenceSettings(model_server_profile=model_server_profile)


__all__ = [
    'ContextWindowSettings',
    'DatabaseSettings',
    'DedicatedLlamaServerSettings',
    'EmbeddingModelSettings',
    'EmbeddingSettings',
    'ExerciseFinderSettings',
    'ImageDescriptionSettings',
    'InstructionFinderSettings',
    'InstructionGovernanceSettings',
    'LocalModelRuntimeSettings',
    'LlamaServerSettings',
    'ModelServerProfileSettings',
    'OCRSettings',
    'PedagogicalFinderSettings',
    'PredictorStrategy',
    'RerankerModelSettings',
    'RerankerSettings',
    'RouterServerSettings',
    'SemanticSettings',
    'Settings',
    'SourceSettings',
    'StageInferenceSettings',
    'StatementProcedureSettings',
    'TermDescriptionSettings',
]
