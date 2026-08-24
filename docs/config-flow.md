# KMS Configuration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFIG LOADING (settings_customise_sources)         │
│                         Priority: HIGH → LOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  1. INIT SETTINGS (programmatic Settings(...) args)             │
        │     Highest priority - used for testing/overrides               │
        └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  2. ENVIRONMENT VARIABLES (including .env via load_dotenv)      │
        │     Prefix: KMS_  |  Delimiter: __                              │
        │                                                                 │
        │     Examples:                                                   │
        │     KMS_MODELS__MODULES__FORMATTER__MODEL=openai/qwen3.5-9b    │
        │     KMS_STAGES__ENTITY_ENRICHMENT__BEFORE_BUDGET=300           │
        │     KMS_DATABASE__PASSWORD=secret                              │
        └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  3. TOML FILE (config.toml or KMS_CONFIG path)                  │
        │     Project defaults committed to repo                          │
        └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │  4. FILE SECRETS (Docker secrets, /run/secrets/*)               │
        │     Lowest priority                                             │
        └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PYDANTIC SETTINGS VALIDATION                             │
│  • extra='forbid' rejects unknown keys                                      │
│  • model_validator(mode='after') runs cross-field validation               │
│    - Local modules must have serving.module_models mapping                 │
│    - Module model must match preset name (after openai/ prefix)            │
│    - max_tokens < preset ctx_size                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SETTINGS OBJECT (cached singleton)                   │
│  get_settings() → Settings with all nested configs:                        │
│                                                                             │
│  Settings {                                                                 │
│    models: ModelsConfig {                                                   │
│      modules: dict[str, ProviderLMModel]  # per-module LLM config          │
│      openrouter_api_key: str                                              │
│      deepseek_api_key: str                                                │
│    }                                                                        │
│    serving: ServingConfig {                                                │
│      manage: bool, host, port, max_loaded_models                           │
│      module_models: dict[str, str]      # module → preset                  │
│      presets: dict[str, ModelPreset>   # llama-server presets             │
│    }                                                                        │
│    embeddings: EmbeddingsConfig { model, api_key, base_url, ... }          │
│    reranker: RerankerConfig { model, api_key, base_url, ... }              │
│    ocr: OCRConfig { api_key, model, url, ... }                             │
│    database: DatabaseConfig { uri, username, password, ... }               │
│    image: ImageConfig { max_dim }                                          │
│    concurrency: ConcurrencyConfig { max_concurrent_calls, recursion_limit }│
│    recording: RecordingConfig { enabled }                                  │
│    stages: StagesConfig {                                                  │
│      splitter, finders, governance,                                        │
│      entity_enrichment, predicate_enrichment, triplet,                     │
│      entity_hubs, predicate_hubs, triplet_hubs,                            │
│      statement_enrichment, procedure_enrichment,                           │
│      search, procedure, statement_hubs, procedure_hubs                     │
│    }                                                                        │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │   LLAMA-SERVER        │         │   MODULE LMs          │
        │   (RouterManager)     │         │   (module_lm)         │
        └───────────────────────┘         └───────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐         ┌───────────────────────┐
        │ default_router()      │         │ module_lm(name)       │
        │ 1. Get serving config │         │ 1. Get module config  │
        │ 2. Generate preset INI│         │ 2. If base_url:       │
        │    from presets       │         │    - Use local llama- │
        │ 3. Start llama-server │         │      server endpoint  │
        │    in router mode     │         │    - Model = preset   │
        │ 4. Manage load/unload │         │ 3. Else:              │
        │    via /models API    │         │    - Use OpenRouter/  │
        │                       │         │      DeepSeek         │
        └───────────────────────┘         └───────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │                    PIPELINE EXECUTION                           │
        │                                                                 │
        │  RouterManager.aexecute(model_id, operation)                   │
        │  • Concurrent ops on same model run together                   │
        │  • Different model → wait, switch, then run                    │
        │  • ContextVar binds manager to call context                    │
        │                                                                 │
        │  Each stage calls:                                             │
        │    lm = module_lm("entity_enrichment")                         │
        │    await lm.acall(...)                                         │
        └─────────────────────────────────────────────────────────────────┘
```

## Key Code Paths

| File | Responsibility |
|------|----------------|
| `src/kms/config.py` | Settings model, source precedence, validation |
| `src/kms/core/llm.py` | `module_lm()`, `gate()`, provider routing |
| `src/kms/core/serve.py` | `RouterManager`, `default_router()`, preset INI generation |
| `config.toml` | Committed defaults |
| `.env` | Local overrides (not committed) |

## Environment Variable Mapping

```python
# TOML path                    →  Environment Variable
models.modules.formatter.model →  KMS_MODELS__MODULES__FORMATTER__MODEL
serving.module_models.fmt     →  KMS_SERVING__MODULE_MODELS__FMT
serving.presets.qwen.ctx_size →  KMS_SERVING__PRESETS__QWEN__CTX_SIZE
stages.entity_enrichment.before_budget → KMS_STAGES__ENTITY_ENRICHMENT__BEFORE_BUDGET
database.password             →  KMS_DATABASE__PASSWORD
```

## Validation Rules (in config.py model_validator)

1. **Local module → preset mapping required** if `serving.manage = true`
2. **Model name must match preset** (after stripping `openai/` prefix)
3. **max_tokens < preset.ctx_size** (prevents context overflow)
4. **Unknown keys rejected** at all levels (`extra='forbid'`)