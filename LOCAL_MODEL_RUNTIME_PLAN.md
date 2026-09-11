# KMS2 local model runtime plan

## Context
KMS2 needs one explicit llama.cpp lifecycle that starts and stops the local LLM router, embedding server, and reranker server together. The LLM configuration must distinguish model artifacts from per-role prediction behavior and from process settings. KMS2 remains self-owned: no `kms` imports, global settings/runtime caches, TOML, compatibility APIs, or retrieval stages added to the current source graph.

## Approach

### 1. Replace generic remote inference configuration with local profiles, roles, and server settings
Update `src/kms2/config.py` so `Settings` gains `local_models: LocalModelRuntimeSettings` and source inference roles select a local LLM profile rather than carrying endpoint and provider configuration.

- Remove `LanguageModelSettings` and replace it with `LLMProfileSettings`, whose fields are `model_path: str`, `capabilities: set[LanguageModelCapability]`, `mmproj_path: str | None`, `context_size: int`, `cache_type_k: str`, `cache_type_v: str`, `n_gpu_layers: int`, `no_warmup: bool`, `parallel: int`, `flash_attention: str`, `reasoning: str`, `temperature: float | None`, and `threads: int`. The map key is the llama.cpp preset and OpenAI model ID; do not duplicate it as another field.
- Change `InferenceSettings` to `profile: str`, `strategy: PredictorStrategy = PredictorStrategy.PREDICT`, `temperature: float = 0.0`, `max_tokens: int = 8192`, `num_retries: int = 3`, and `cache: bool = True`. Remove `api_base`, `api_key`, and `provider_options`; all current KMS2 source roles use local llama.cpp.
- Change `_inference()` to accept a profile ID. Source defaults must use the text profile for formatting, text-seam judge/rewriter, splitter router, and splitter; use the vision profile for content correction, image-seam judge, and image enrichment.
- Add `LlamaServerSettings` for process fields shared by every role: `executable = 'llama-server'`, `host = '127.0.0.1'`, `port`, `ready_timeout = 300.0`, `poll_interval = 1.0`, `health_timeout = 30.0`, and `terminate_timeout = 15.0`.
- Add `LLMRouterSettings(LlamaServerSettings)` with `port = 8080`, `max_loaded_models = 1`, and `profiles: dict[str, LLMProfileSettings]`. Direct defaults must be `gemma-4-e4b-qat-text` at `~/models/gemma-4-e4b-qat/gemma-4-E4B_q4_0-it.gguf` (text, context 32768, flash attention on, empty K cache, V cache `q4_0`, 999 GPU layers, no warmup, one parallel request, reasoning off, zero threads) and `gemma-4-e4b-qat-vision` using the same GGUF plus `~/models/gemma-4-e4b-qat/gemma-4-E4B-it-mmproj.gguf` (text+vision, context 8192, K/V caches `q4_0`, temperature 0.0, otherwise the same runtime values).
- Add dedicated retrieval model and server models: `EmbeddingModelSettings(model_id, model_path, dimension)`, `RerankerModelSettings(model_id, model_path)`, and `DedicatedLlamaServerSettings(LlamaServerSettings)` with `device`, `n_gpu_layers`, `threads`, `threads_batch`, `ubatch_size`, `context_size`, `parallel`, `cache_ram`, and `no_warmup`. `EmbeddingSettings` and `RerankerSettings` each own a model, a dedicated server, and their request settings (`batch_size`, `timeout_seconds`).
- Direct defaults must reproduce the studied local deployment: embedding model `Qwen3-Embedding-0.6B` at `~/models/qwen3-embedding-0.6b/Qwen3-Embedding-0.6B-Q4_K_M.gguf`, dimension 1024, port 8081, CUDA0, 999 layers, 12/12 threads, batch 32, ubatch 2048, context 4096; reranker model `Qwen3-Reranker-0.6B` at `~/models/qwen3-reranker-0.6b/Qwen3-Reranker-0.6B-Q4_K_M.gguf`, port 8082, CUDA0, 999 layers, 4/4 threads, batch 8, ubatch 2048, context 8192. Both use host `127.0.0.1`, one parallel request, zero cache RAM, no warmup, 300-second readiness, one-second polls, 30-second health probes, 15-second termination, and 120-second client requests.
- Retain `KMS2_` and `__` as the sole configuration override convention. Role selection therefore uses values such as `KMS2_SOURCE__FORMATTING__PROFILE=gemma-4-e4b-qat-text`; server tuning uses values such as `KMS2_LOCAL_MODELS__ROUTER__PORT=8080`. Do not add TOML or runtime environment discovery.

### 2. Add a single KMS2-owned llama.cpp runtime
Create `src/kms2/local_models.py`; no equivalent KMS2 lifecycle module exists.

- Implement private `_ManagedLlamaServer(command: list[str], endpoint: str, readiness_path: str, settings: LlamaServerSettings)`. `start()` starts only when its endpoint is not ready, uses `subprocess.Popen(..., start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)`, waits for the configured readiness endpoint, and retains only processes it spawned. `astart()` calls `start()` through `asyncio.to_thread()`. `close()` terminates an owned process, waits `terminate_timeout`, then kills it if needed; externally running endpoints are never terminated.
- Implement `LLMRouter` on the same managed-process primitive. Its start command is exactly `llama-server --host <host> --port <port> --models-preset <runtime-owned-preset-path> --models-max <max_loaded_models> --no-models-autoload`; readiness is `/models`. Generate the profile INI in a `tempfile.TemporaryDirectory` owned by `LocalModelRuntime`, including each `LLMProfileSettings` field as its llama.cpp preset equivalent (`model`, optional `mmproj`, `ctx-size`, caches, GPU layers, warmup, parallel, flash attention, reasoning, optional temperature, threads). Delete that directory in `close()`.
- `LLMRouter.ensure_model(profile_id)` must query `/models`, unload every different `loaded` or `loading` profile through `POST /models/unload`, load the requested profile with `POST /models/load`, and wait until its status is `loaded`. `LLMRouter.aexecute(profile_id, operation)` must serialize switches but allow concurrent operations for the resident profile; do not switch until all active operations for the prior profile complete.
- Implement `RoutedPredictor` with `__call__(*args, **kwargs)` and `async acall(*args, **kwargs)`. `LocalModelRuntime.predictor()` resolves `inference.profile` from `router.profiles`, builds `dspy.LM(f'openai/{profile_id}', api_base=f'{router.endpoint}/v1', api_key='not-needed', temperature=inference.temperature, max_tokens=inference.max_tokens, num_retries=inference.num_retries, cache=inference.cache)`, then builds the requested `dspy.Predict` or `dspy.ChainOfThought` and wraps it. The synchronous path calls `LLMRouter.ensure_model(profile_id)` before delegation and the asynchronous path delegates under `LLMRouter.aexecute(profile_id, ...)`.
- Construct dedicated embedding and reranking commands from their typed settings, expanding `~` in every model path. Embedding uses `--model <path> --host <host> --port <port> --embedding --device <device> --pooling last` plus the shared dedicated flags. Reranking uses the same command plus `--reranking --pooling rank`. Both use `/health` readiness.
- Implement `EmbeddingClient(settings: EmbeddingSettings, endpoint: str, client: httpx.AsyncClient)` with `embed(texts: Sequence[str]) -> list[list[float]]` and `RerankerClient(settings: RerankerSettings, endpoint: str, client: httpx.AsyncClient)` with `rerank(query: str, documents: Sequence[str], top_n: int | None = None) -> list[dict[str, object]]`. `LocalModelRuntime` creates and owns both clients. The embedding client issues `POST /v1/embeddings` with `{"model": model_id, "input": [...]}` in configured batches; the reranker issues `POST /v1/rerank` with `{"model": model_id, "query": query, "documents": [...], "top_n": top_n}` when supplied. Inputs and response payloads are assumed valid, so preserve server ordering and do not add legacy response validation, retries, client caches, or global singletons.
- Implement `LocalModelRuntime(settings: LocalModelRuntimeSettings)` with `async start()`, `async close()`, `async __aenter__()`, `async __aexit__()`, `predictor(inference: InferenceSettings, signature: type[dspy.Signature]) -> RoutedPredictor`, and exposed `embedding` / `reranker` clients. `start()` eagerly starts router, embedding, and reranker processes concurrently but leaves the router model unloaded. `close()` closes the two async HTTP clients, closes the three managed servers, and cleans its preset directory. A caller owns one runtime explicitly; no package-level factory or cache is allowed.

### 3. Inject the local runtime into source composition
Update `src/kms2/composition.py` so `build_source_graph` has the exact signature `build_source_graph(settings: Settings, local_models: LocalModelRuntime) -> SourceGraph`.

- Remove `build_language_model()` and `build_predictor()` and remove both from `__all__`; their only production caller is the existing source graph composition and their direct tests are migrated in the next step.
- Replace each of the eight existing predictor constructions with `local_models.predictor(<role inference settings>, <existing DSPy signature>)`, preserving the existing source module/node graph topology and role order.
- Construct the runtime before graph construction and own it around graph execution: `async with LocalModelRuntime(Settings().local_models) as local_models: graph = build_source_graph(settings, local_models).build_graph(); await graph.ainvoke(...)`. Do not add the runtime, embedding, or reranking data to `SourceState`; all source modules already receive their predictor dependencies explicitly.
- Do not change `DatabaseClient` ownership or add a KMS2 CLI/runtime entrypoint in this change. The hidden database client lifecycle is independent of local model process startup.

### 4. Migrate tests to the clean cutover and prove lifecycle behavior
Update `tests/test_kms2_settings.py`, `tests/test_kms2_inference.py`, and `tests/test_kms2_source_application.py` to construct `InferenceSettings(profile=...)`, assert the two default profile IDs and paths, assert nested `KMS2_` overrides for role profiles and server ports, and assert composition receives routed predictors instead of direct `dspy.Predict` instances.

Create `tests/test_kms2_local_models.py` using fake `Popen`, fake router status responses, and `httpx.MockTransport` clients. Cover all observable new contracts:

- Given the default local-model settings, runtime command construction produces the exact router preset/flags, embedding `--embedding --pooling last` flags, and reranker `--embedding --reranking --pooling rank` flags with reranker ubatch size `2048`.
- Given unavailable endpoints followed by ready fake health responses, `await runtime.start()` launches all three server roles, does not load an LLM model, and `await runtime.close()` closes only its spawned processes and both HTTP clients. Given already-ready external endpoints, startup launches nothing and close terminates nothing.
- Given one routed async prediction for the text profile followed by one for the vision profile, the router sends a text load, waits for the first operation lease before changing profiles, unloads text, loads vision, and delegates both DSPy calls to the matching profile endpoint.
- Given embedding inputs and a rerank query/documents through mock HTTP transports, clients send the exact `/v1/embeddings` and `/v1/rerank` payloads and return the server payload ordering unchanged.

## Critical files & anchors

- `src/kms2/config.py` — replace `LanguageModelSettings` / `InferenceSettings` and add all typed local profile, role, and process settings.
- `src/kms2/local_models.py` — new sole owner of shared llama.cpp process startup, router leasing, generated presets, and retrieval clients.
- `src/kms2/composition.py` — `build_source_graph` is the only production predictor construction boundary and must receive `LocalModelRuntime` explicitly.
- `tests/test_kms2_local_models.py` — new behavioral proof for all three managed server roles, router leases, and retrieval wire protocols.
- `tests/test_kms2_source_application.py` — proves source composition uses profile-routed predictors without changing node/graph boundaries.

## Verification

From the repository root, run:

```bash
uv run ruff format --check src/kms2 tests/test_kms2*.py
uv run ruff check src/kms2 tests/test_kms2*.py
uv run pytest -q tests/test_kms2_local_models.py tests/test_kms2_settings.py tests/test_kms2_inference.py tests/test_kms2_source_application.py tests/test_kms2_source_graph.py
uv run pytest -q tests/test_kms2*.py
```

The new runtime test is the end-to-end behavioral proof without requiring installed model files: fake ready endpoints and fake subprocesses must observe startup of all three role commands; mock router state must observe load/unload/lease order; mock HTTP endpoints must observe exact embedding and reranking payloads.

## Assumptions & contingencies

- llama.cpp remains the local backend and `llama-server` remains on `PATH`; the currently installed binary is `/home/joshua/.local/bin/llama-server`, version `1 (86a9c79)`.
- Model locations and CUDA0 settings use the existing local deployment paths and values above. If a model path differs on the target machine, override only its typed `KMS2_LOCAL_MODELS__...` value; do not add a second configuration source.
- KMS2 has no retrieval consumer today. The embedding and reranker clients are delivered as runtime-owned injectable services, but no source graph, DTO, persistence schema, or retrieval behavior is added until a caller exists.
- Input and output contracts are assumed perfect. Startup retains process-exit and timeout errors because those describe machine lifecycle failure, but it does not add legacy payload validation, fallback endpoints, retry policies, or compatibility behavior.
