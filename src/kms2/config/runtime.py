"""Local model-server and retrieval configuration models."""

from pydantic import BaseModel, Field


class ModelServerProfileSettings(BaseModel):
    """Load-time settings for one llama.cpp model-server profile."""

    model_path: str
    mmproj_path: str | None = None
    context_size: int = 32768
    cache_type_k: str = ''
    cache_type_v: str = 'q4_0'
    n_gpu_layers: int = 999
    no_warmup: bool = True
    parallel: int = 1
    flash_attention: str = 'on'
    reasoning: str = 'off'
    threads: int = 0


class LlamaServerSettings(BaseModel):
    """Shared process and readiness settings for llama.cpp servers."""

    executable: str = 'llama-server'
    host: str = '127.0.0.1'
    port: int
    ready_timeout: float = 300.0
    poll_interval: float = 1.0
    health_timeout: float = 30.0
    terminate_timeout: float = 15.0


class RouterServerSettings(LlamaServerSettings):
    """llama.cpp router settings and per-profile model definitions."""

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
    threads: int
    threads_batch: int
    ubatch_size: int
    context_size: int
    parallel: int = 1
    cache_ram: int = 0
    no_warmup: bool = True


class EmbeddingServerSettings(DedicatedLlamaServerSettings):
    """Dedicated llama.cpp settings for embedding inference."""

    port: int = 8081
    threads: int = 12
    threads_batch: int = 12
    ubatch_size: int = 2048
    context_size: int = 4096


class RerankerServerSettings(DedicatedLlamaServerSettings):
    """Dedicated llama.cpp settings for reranking inference."""

    port: int = 8082
    threads: int = 4
    threads_batch: int = 4
    ubatch_size: int = 2048
    context_size: int = 8192


class EmbeddingModelSettings(BaseModel):
    """Embedding model identity and vector dimensions."""

    model_id: str = 'Qwen3-Embedding-8B'
    model_path: str = (
        '~/models/qwen3-embedding-8b/Qwen3-Embedding-8B-Q4_K_M.gguf'
    )
    dimension: int = 4096


class RerankerModelSettings(BaseModel):
    """Reranker model identity and model path."""

    model_id: str = 'Qwen3-Reranker-8B'
    model_path: str = (
        '~/models/qwen3-reranker-8b-verified/Qwen3-Reranker-8B-Q4_K_M.gguf'
    )


class EmbeddingSettings(BaseModel):
    """Embedding model, server, batching, and request settings."""

    model: EmbeddingModelSettings = Field(
        default_factory=EmbeddingModelSettings
    )
    server: EmbeddingServerSettings = Field(
        default_factory=EmbeddingServerSettings
    )
    batch_size: int = 32
    timeout_seconds: float = 120.0


class RerankerSettings(BaseModel):
    """Reranker model, server, and request settings."""

    model: RerankerModelSettings = Field(default_factory=RerankerModelSettings)
    server: RerankerServerSettings = Field(
        default_factory=RerankerServerSettings
    )
    timeout_seconds: float = 120.0


class LocalModelRuntimeSettings(BaseModel):
    """Settings for all KMS2-owned local model servers."""

    router: RouterServerSettings = Field(default_factory=RouterServerSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
