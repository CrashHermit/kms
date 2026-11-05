# Multimodal retrieval components and tokenizer ownership

## Context

Replace KMS's text-only embedding/reranking boundaries with reusable multimodal Protocol interfaces. Reorganize provider adapters and token accounting by capability; composition factories construct matching clients, limits, tokenizer resources and embedding identity/dimension, while the application owns their lifetime. Assume internal inputs and model outputs are valid and complete; do not add repairs, fake fallbacks or compatibility aliases.

## Confirmed findings

- `core/embedding.py::EmbeddingClient.embed(texts: list[str])` and `core/reranking.py::RerankerClient.rerank(query: str, documents: list[str], top_n: int | None)` are currently text-only Protocols.
- `core/content.py::ContentImage` already represents an immutable keyed `Path | bytes` native image. `MultimodalContentBuilder` already projects text and references into a request-local image catalog.
- `inference/` currently contains flat `embedding.py`, `reranking.py`, `token_counting.py`, and `chatgpt.py`; provider-specific and capability-specific accounting are not grouped by capability.
- User selected hosted multimodal providers and explicitly authorized a fresh configured graph: existing source, session, and learner graph data may be deleted at cutover. Do not delete media files or modify credential files.
- Pin direct Voyage `voyage-multimodal-3.5` embedding API (`https://api.voyageai.com/v1/multimodalembeddings`) and IONOS `Qwen/Qwen3-VL-Reranker-8B` reranking API (`https://openai.inference.de-txl.ionos.com/v1/rerank`). Primary API contracts support mixed text/image inputs; IONOS supports content-part arrays for image-bearing queries and candidate objects.
- Voyage documents 1024 default output dimensions, query/document modes, `truncation=false`, maximum 1000 inputs, 32000 tokens per input, 320000 total tokens, and per-image limits of 16000000 pixels and 20 MB. Each 560 image pixels counts as one input token.
- IONOS documents candidate count 1–1000 and 32768 tokens per query/document pair. Image accounting uses 32-pixel patches with provider-side downscale above 1310720 pixels; local accounting must not reuse the LLM's OpenAI-style image formula.
- Primary sources: https://docs.voyageai.com/reference/multimodal-embeddings-api ; https://docs.voyageai.com/docs/multimodal-embeddings ; https://docs.ionos.com/cloud/ai/ai-model-hub/how-tos/reranking ; https://docs.ionos.com/cloud/ai/ai-model-hub/models/reranking-models/qwen3-vl-reranker-8b .
- Reuse installed LiteLLM for embedding transport: `inference/embedding.py` already calls `litellm.aembedding`, and installed `litellm/llms/voyage/embedding/transformation_multimodal.py::VoyageMultimodalEmbeddingConfig` routes `voyage/*multimodal*` models to `/multimodalembeddings`, accepts dictionaries with `content`, passes explicit `image_base64` parts through unchanged, and maps `dimensions` to `output_dimension`. Use explicit Voyage `image_base64` data URI content parts; do not send `image_url` data URIs through the transform that strips their URI header. Retain a thin application Embedder Protocol/adapter rather than a second provider-routing framework. Docs: https://docs.litellm.ai/docs/providers/voyage .

## Approach

Execute steps 1–5 as one integrated interface cutover with callers/tests migrated together; do not keep compatibility layers to make intermediate signatures pass. Step 6 is a separate one-time runtime action after provider smoke proof. Reuse `docs/STYLE.md` and neighboring composition/node conventions.

### 1. Define the multimodal capability contracts and final module ownership

Use existing `core/model/content.py::MultimodalContent` and `core/content.py::ContentImage` / `MultimodalContentBuilder`; do not invent a second text/image packet. Text-only inputs have `image_positions=[]`; image-only inputs have `text=None`; valid material contains text or images. Images are owned `Path | bytes` artifacts, never provider-fetched web URLs. Keep image order, source ownership and source provenance separate from request-local catalog ordinals.

Replace `core/embedding.py::EmbeddingClient` with the following `Protocol`, named `Embedder`, with no runtime-check decorator:

```python
async def embed(
    self,
    materials: list[MultimodalContent],
    *,
    content_images: list[ContentImage],
    input_type: Literal['query', 'document'],
) -> list[list[float]]: ...
```

Add frozen/slotted `EmbeddingModel(model: str, dimension: int)` in that same core module. The pinned production identity is `voyage/voyage-multimodal-3.5`, dimension `1024`. All stored vectors use document mode; search queries use query mode. Hub reconciliation may compare two stored document vectors exactly as before; do not change its algorithms.

Replace `core/reranking.py::RerankerClient` with a `Reranker` Protocol:

```python
async def rerank(
    self,
    query: MultimodalContent,
    documents: list[MultimodalContent],
    *,
    content_images: list[ContentImage],
    top_n: int | None = None,
) -> list[RerankResult]: ...
```

Retain `RerankResult`'s exact `index: int` and `relevance_score: float` fields. Each result refers to the original candidate position; no provider types escape the adapter. Empty input batches return `[]` without loading tokenizers, reading images, or making a network call. Inputs and model outputs are assumed valid: remove `decode_embedding` and the old malformed-output/coercion branches and their exclusively malformed-output tests during cutover; keep real HTTP/file errors, budgets, authorization and persistence integrity checks. Valid response indexes still determine output order; never assume provider response order is input order.

Reorganize into these concrete modules, using minimal `__init__.py` files without re-exports:

```text
core/
  embedding.py                     # Embedder, EmbeddingModel
  reranking.py                     # Reranker, RerankResult
  tokenization/
    text.py                        # TextTokenCounter, count_text_tokens
    images.py                      # ImageMetadata, ImageMetadataCache
    material.py                    # MaterialTokenEstimator (LLM material heuristic)
    budget.py                      # TokenBudget, InputBudgetExceeded, fits_token_budget
inference/
  tokenization/
    text.py                        # HuggingFaceTextTokenCounter, TiktokenTextTokenCounter
  embedding/
    litellm.py                     # LiteLLMEmbedder, Voyage multimodal wire conversion
    budget.py                      # VoyageEmbeddingBudget
  reranking/
    ionos.py                       # IonosReranker
    budget.py                      # IonosRerankingBudget
  chatgpt.py                       # Leave authentication here
composition/
  tokenization.py                  # LanguageModelTokenizers
  embedding.py                     # EmbeddingResources, embedding_resources
  reranking.py                     # RerankingResources, reranking_resources
```

No equivalent cohesive retrieval resource or multimodal provider adapter exists; reuse existing batching, catalog construction, artifact reading and HTTP ownership code rather than introducing a plugin registry, universal model base class, or provider discovery service. Remove the old flat `inference/embedding.py`, `inference/reranking.py`, `inference/token_counting.py`, `local_models/token_counting.py`, `core/token_counting.py` and `core/image_token_counting.py` after moving their retained responsibilities. Migrate tests and all callers in the same cutover; do not leave aliases or dual APIs.

### 2. Separate generative token estimation from provider retrieval budgeting

Move `TextTokenCounter` and vectorized `count_text_tokens` into `core/tokenization/text.py`. Keep caller-specific token semantics in each concrete counter. Move the existing LLM `TokenCounter` to `core/tokenization/material.py` and rename it `MaterialTokenEstimator`, with frozen/slotted fields `text_counter: TextTokenCounter` and `image_metadata: ImageMetadataCache = field(default_factory=ImageMetadataCache)`. Retain `count_texts(...)` delegation and `estimate(*, text=None, images=())` behavior, including zero cost for absent/empty material. Its fixed LLM image heuristic remains `ceil(ceil(width/32) * ceil(height/32) * 1.2)` and remains explicitly an estimate, not complete DSPy prompt accounting. Do not change existing source-window selection, budgets, repeated-path overcount shortcut or LLM instructions in this retrieval refactor.

Extract the existing image header/stat cache from `ImageTokenCounter.measure` into `ImageMetadataCache.measure(ContentImage) -> ImageMetadata`; keep positive dimensions and immutable image keys as the existing valid-input invariant. Keep the cached width/height/encoded byte count; read headers only, close handles, and propagate missing/corrupt/read errors. `MaterialTokenEstimator` owns a reference to this cache; expose it as `image_metadata` so `service/content.py::_image_metadata_fits` reuses it. Move LLM image summation into `MaterialTokenEstimator.estimate_images(images: Sequence[ContentImage]) -> int`; change `core/source_context.py` to call this method instead of reaching into a fixed image-counter implementation. `service/content.py` creates a per-pack `ImageMetadataCache` when no token budget is provided. Retrieval budgets also receive this same kind of cache but implement their own model rules.

Move `TokenBudget`, `InputBudgetExceeded` and `fits_token_budget` from `core/windowing.py` to `core/tokenization/budget.py`; `TokenBudget.counter` is `MaterialTokenEstimator`. `windowing.py` retains only ordered selection and packing, importing the exception where needed, not compatibility re-exports. Migrate imports of all moved symbols under `src/kms`, `tests/test_kms_*.py`, and `tests/kms_token_helpers.py`.

Implement `inference/tokenization/text.py::HuggingFaceTextTokenCounter(tokenizer_path: str, *, add_special_tokens: bool)`, using installed `tokenizers.Tokenizer`, `no_padding()`, `no_truncation()`, and ordered `encode_batch_fast`. Load lazily on the first nonempty counting call, not in factory construction. Absolute paths, paths starting `~`/`.` and existing file paths use `Tokenizer.from_file` after expansion; all other values are explicit repository IDs for `Tokenizer.from_pretrained`. A missing local path must raise as a missing file, not be retried as a repository ID. Reuse the existing local/Voyage counter implementation here rather than retaining three copies. `TiktokenTextTokenCounter(encoding_name: str)` retains the existing hosted generative encoding semantics. Compute the normalizer's `text_token_limit` lazily when requested, not at retrieval resource construction.

Replace `InferenceTokenizers` with `composition/tokenization.py::LanguageModelTokenizers(settings: Settings)`, retaining `for_inference(stage) -> MaterialTokenEstimator` and `counters(*stages) -> tuple[MaterialTokenEstimator, ...]`, per-profile deduplication, lazy local tokenizer loading and one metadata cache per owner lifetime. Remove `embedding`, `selected_embedding`, `reranker` and `_LocalEmbeddingCounter`. It selects only generative text counters: local router profile tokenizer with `add_special_tokens=False`, or configured tiktoken encoding. Retrieval factories construct their own matching counters; no other code branches on retrieval backend to pick a tokenizer.

`VoyageEmbeddingBudget` exposes `text_counter: TextTokenCounter`, `text_token_limit: int`, and `count_inputs(materials, *, content_images, input_type) -> list[int]`. Use tokenizer repository `MongoDB/voyage-multimodal-3.5` (its tokenizer files were confirmed) with `add_special_tokens=True`. Count the documented query/document prefix plus text; image references are charged per material occurrence, even when the native image catalog is deduplicated. Image upper-bound cost is `(max(50_000, width * height) + 559) // 560`, based on Voyage's published ratio and minimum-pixel tokenizer metadata; do not reuse the LLM patch heuristic or silently resize. Add a fixed 256-token per-input reserve for provider framing and label the total a conservative budget estimate, not billing truth. `text_token_limit` is `32_000 - 256 - count(document prefix)` and is for the normalizer's text-only output capacity only; actual provider dispatch always uses `count_inputs`, including images. Hard per-image checks: at most 16,000,000 pixels and 20,000,000 encoded bytes.

`IonosRerankingBudget.count_pairs(query, documents, *, content_images) -> list[int]` uses `Qwen/Qwen3-VL-Reranker-8B`'s confirmed tokenizer repository, with no truncation/padding and `add_special_tokens=False`. Count query text + each candidate text + 256 per-pair framing reserve + `ceil(width/32)*ceil(height/32)` per image occurrence in that pair. Intentionally do not assume the server's documented downscale savings; this is a conservative upper bound, not a claim of exact server tokenization. Enforce each pair <=32768 estimated tokens before sending. No provider output repair or image replacement is permitted.

### 3. Build the two hosted components with their matching resources

Replace `EmbeddingProviderSettings` with `EmbeddingSettings` in `config/embedding.py`, fields:

- `model: Literal['voyage/voyage-multimodal-3.5'] = 'voyage/voyage-multimodal-3.5'`
- `api_key: SecretStr | None = None`
- `max_concurrency: int = Field(default=4, gt=0)`
- `batch_size: int = Field(default=32, gt=0, le=1000)`
- `batch_token_budget: int = Field(default=8192, gt=0, le=320000)`
- `timeout_seconds: float = Field(default=120.0, gt=0)`
- `tokenizer_path: str = 'MongoDB/voyage-multimodal-3.5'`

Replace `RerankerProviderSettings` with `RerankingSettings` in `config/reranking.py`, fields:

- `model: Literal['Qwen/Qwen3-VL-Reranker-8B'] = 'Qwen/Qwen3-VL-Reranker-8B'`
- `api_key: SecretStr | None = None`
- `max_concurrency: int = Field(default=4, gt=0)`
- `timeout_seconds: float = Field(default=120.0, gt=0)`
- `batch_token_budget: int = Field(default=32768, gt=0)`
- `tokenizer_path: str = 'Qwen/Qwen3-VL-Reranker-8B'`

Keep root names `settings.embedding` and `settings.reranker` and corresponding environment prefixes. Remove backend flags and old model literals; no backend options for implementations that no longer exist. API keys are explicit factory dependencies resolved from these settings, not global environment mutation. Missing/empty keys raise `ValueError('Embedding API key is required')` or `ValueError('Reranking API key is required')` on acquisition; operations requiring neither capability must not demand their credentials. Do not repurpose an OpenRouter key as a Voyage key or a Voyage key as an IONOS key.

`composition/embedding.py` defines frozen/slotted `EmbeddingResources(embedder: Embedder, model: EmbeddingModel, budget: VoyageEmbeddingBudget)` and `@asynccontextmanager async def embedding_resources(settings: Settings) -> AsyncIterator[EmbeddingResources]`. Factory selects the matching tokenizer/budget and returns one `LiteLLMEmbedder` for the scope. Pinned model metadata comes from this factory only; remove separate `embedding_model(settings)` and `embedding_dimension(settings)` functions and `build_embedding_client`. Use the context manager even though LiteLLM owns its transport pools; do not close global LiteLLM pools when an application scope ends.

`LiteLLMEmbedder` constructs content parts for each material: a nonempty text part when present, followed by images in `image_positions` order. Read only referenced catalog images off-thread through the existing image-loading path; determine the MIME type from image headers, not filename suffixes. Reuse an encoded data URI once per catalog image within that call; each material still carries its referenced images. Call:

Add `service/content.py::encode_content_images(images: list[ContentImage]) -> list[str]`, an async utility returning same-order image data URIs for the supplied native catalog. Reuse `load_content_images(images, budget=None)` for existing artifact reads, detect format/MIME with Pillow headers, and Base64 encode off-thread. Both retrieval adapters use this helper rather than duplicating media reading and encoding. It does not fetch URLs, alter images, impose a second model-specific budget, or stringify file paths; provider budgets run before encoding. Missing/corrupt/read errors propagate. Never encode or load unused catalog entries.

```python
await litellm.aembedding(
    model=settings.model,
    input=[{'content': parts} for parts in batch_parts],
    api_key=settings.api_key.get_secret_value(),
    input_type=input_type,
    dimensions=1024,
    truncation=False,
    num_retries=0,
    timeout=settings.timeout_seconds,
)
```

Pass `input_type` and `truncation` as the provider-specific kwargs shown, not by mutating `litellm.drop_params` or patching the adapter. Installed `utils.py::PreProcessNonDefaultParams.base_pre_process_non_default_params` copies these kwargs into `passed_params`, and `add_provider_specific_params_to_optional_params` restores provider-only fields. Test actual outgoing JSON for `input_type`, `truncation=false` and `output_dimension=1024`; merely asserting `aembedding` received kwargs is insufficient.

For images use explicit `{'type': 'image_base64', 'image_base64': 'data:<mime>;base64,...'}` parts to preserve headers through the installed Voyage transform. Do not pass public image URLs, substitute descriptions, stringify paths or count Base64 characters. Enforce per-input 32000 tokens, request total <=320000, <=1000 inputs, and per-image limits. Reuse `pack_items` for ordered whole-item batches: an item exceeding the configured soft batch target may be sent alone only when it fits the hard per-input/request limits. Bound async work using the existing worker-pool/cancellation pattern, not one task per input; concurrent batches restore vectors by original indexes. Read valid `response.data`, order by `index`, return vectors directly; do not add malformed-result repairs.

`composition/reranking.py` defines frozen/slotted `RerankingResources(reranker: Reranker, budget: IonosRerankingBudget)` and `@asynccontextmanager async def reranking_resources(settings: Settings) -> AsyncIterator[RerankingResources]`. It opens one `httpx.AsyncClient(timeout=...)`, injects it into `IonosReranker`, and closes it on success, exception or cancellation. Use direct HTTP here because the installed LiteLLM shared rerank request declares a text-only query and does not preserve the full multimodal query contract. No Hugging Face serving/Transformers reranker, custom LiteLLM provider, proxy or vendor patch is part of this cutover.

`IonosReranker` sends `POST https://openai.inference.de-txl.ionos.com/v1/rerank`, Bearer key, model above, query as a content-part array, documents as `{'content': parts}` objects, and `return_documents=false`. Omit `top_n` from every HTTP batch request so all batch scores return; apply the caller's top_n once after global index remapping/merging. Text parts use `{'type':'text','text':...}`; images use `{'type':'image_url','image_url':{'url':data_uri}}`. Do not send LiteLLM's Voyage content-part keys to this provider.

Keep whole-document ordered batching, bounded concurrency, original-index restoration, and final global top_n selection from the existing batching wrapper, but move that behavior into the single capability implementation instead of stacking two provider-specific semaphore/budget layers. Count the repeated query for every candidate pair. Batch by configured soft token budget, <=1000 candidates, <=16 MiB exact UTF-8 JSON body and <=12 MiB decoded bytes per inline image, <=2048 content parts per query/document. Serialize JSON once for the exact body-size check and send those bytes with `Content-Type: application/json`; a one-candidate body or pair exceeding hard limits raises `InputBudgetExceeded` before HTTP. Retain one valid provider request's ordering; for multiple requests retain existing descending-score/original-index tie-break behavior, documenting that this is not a newly calibrated cross-request ranking algorithm. Apply top_n only after all candidate scores have been remapped; do not drop each batch's candidates prematurely.

### 4. Remove obsolete local retrieval ownership and compose resources once

Hosted multimodal retrieval replaces the current local text-only backends. Remove `LocalModelRuntime.embed`, `.rerank`, their HTTP clients/response TypedDicts, dedicated-server commands/settings and `ResidentRole.EMBEDDING/RERANKER` branches. Remove obsolete `DedicatedLlamaServerSettings`, `EmbeddingServerSettings`, `RerankerServerSettings`, local `EmbeddingModelSettings`, `RerankerModelSettings`, `EmbeddingSettings`, `RerankerSettings` and `LocalModelRuntimeSettings.embedding/reranker` from `config/runtime.py` once their references are migrated. Keep `ManagedLlamaServer` and all router/model profile support used by generative inference.

`LocalModelRuntime(settings: LocalModelRuntimeSettings)` no longer takes `embedding_token_counter`. Collapse `_execute` to `(model_server_profile: str, operation: Operation) -> object`, store only active LLM profile, and preserve its existing lock, lifecycle state, activation-failure rollback, cancellation cleanup, lazy router startup and profile release semantics. Update `RuntimePredictor.aforward` accordingly. Remove the one-member `ResidentRole` enum rather than retaining generic role machinery with no remaining consumers.

Change `_ApplicationResources` to hold `embeddings: EmbeddingResources | None`, `reranking: RerankingResources | None`, and `tokenizers: LanguageModelTokenizers`, plus existing local runtime/database/recorder fields. Preserve existing `embeddings_required` / `reranking_required` acquisition boundaries because source-learning-only and database-only operations use them. Acquire factories once with `AsyncExitStack`; use `embeddings.model.dimension` for `schema.ensure_schema`, before inference/persistence. Generative-only paths create only local LLM runtime/tokenizers; do not load retrieval tokenizer downloads or demand keys.

Propagate the embedding resource to `build_source_processing_graph`, `build_source_admission_graph`, and `build_teaching_graph` as required keyword `embeddings: EmbeddingResources`; inject `.embedder` into consumers and `.model` into provenance owners. Use the same selected model descriptor for `EmbeddingNode`'s `embedding_profile`, `SourceAdmissionService` expectations and vector schema. Other graph builders/services receive the `Embedder` or `Reranker` Protocol, not the full settings object or a provider client. Migrate all `EmbeddingClient`, `RerankerClient`, old factory functions, `InferenceTokenizers`, `LocalTokenizers`, `LocalTokenCounter`, runtime constructor and obsolete local retrieval configuration references under `src/kms` and `tests/test_kms_*.py`. Use repository-aware rename when supported; otherwise verified AST-aware renames and explicit imports. The repository now has one Python package namespace: `kms`.

The normalizer still exists in the current repository and must not be left broken: in `composition/teaching.py`, remove backend-specific `embedding_capacity` selection and `tokenizers.selected_embedding`; derive its text-only output `TokenBudget` from `embeddings.budget.text_counter` and `.text_token_limit`, wrapping that text counter with `MaterialTokenEstimator`. Do not remove the HTML/normalizer route or redesign corrector/filter stages in this retrieval task.

### 5. Migrate every embedding and reranking consumer to one material contract

Find the exact callsites with `grep` pattern `\\.embed\\(|\\.rerank\\(` under `src/kms` and all constructors with `EmbeddingClient|RerankerClient`. There are currently 15 direct embedding call expressions, 10 application rerank call expressions and the old batching wrapper's delegate. Update each production call plus its deterministic fake in the same change.

For text-only consumers, construct `MultimodalContent(text=...)`, use `content_images=[]`, and pass explicit embedding mode. Do not create images for propositions, arguments/reasoning, observations, statements, procedures, entity descriptions, or synthesized hubs. Source learning's atomic/coherent LLM inputs are not retrieval calls and keep their current multimodal context behavior.

Document embedding sites:

- `node/source_embedding.py::embed_source_pages`: eligible missing vector blocks have nonblank text **or assets**. Project each whole block with `project_source_block` and one builder, sending text plus all ordered assets, not `crop_path`. Image-only blocks must be embedded. Empty text/no assets remains skipped; existing vectors are preserved. Reuse the resulting index-to-block mapping to rebuild pages without changing UUIDs, crops, assets or ordering.
- `node/knowledge/identity.py::run`: proposition and reasoning material; preserve existing reuse of stored/exact/earlier-draft vectors and reasoning text construction.
- `node/source_pedagogical/source_statement_embedding.py::run`, `source_procedure_embedding.py::run`.
- `node/source_semantic/source_entity_embedding.py::run`, `source_statement_hub.py::embed`, `source_procedure_hub.py::embed`.
- `node/teaching/observation.py::run`.

Query embedding sites all use `input_type='query'`: `service/knowledge_search.py::search_propositions/search_reasoning_steps`, `service/learner_observation_search.py::search`, and `service/source_evidence.py::SourceSearchService.search_sources/search_statements/search_procedures`. Public search APIs keep string queries; project strings at the capability boundary. The core protocols and provider tests support multimodal query material without adding unrelated new TUI/tool parameters.

Rerank sites: `node/knowledge/identity.py::_candidate_pool`, `node/source_semantic/source_entity_hub.py::rerank_worker`, `source_statement_hub.py::rerank_worker`, `source_procedure_hub.py::rerank_worker`, and the six search service methods listed above. Text-only domain entities use native text materials, not an old string overload. `SourceSearchService.search_sources` projects each candidate's `SourceBlock` through `project_source_block`, including real native assets, with a fresh shared catalog; query text uses that same builder. Preserve current result-index lookup, similarity, relevance score, authorization, shortlist and exercise distinctions.

Source search currently excludes `block.content IS NULL`; remove this predicate in both `SEARCH_OWNED_SOURCE_BLOCKS` and `SEARCH_OWNED_SOURCE_BLOCKS_VECTOR` in `database/source/queries/source_blocks.py`. Eligibility becomes an available vector with valid text or assets; embedding production already ensures empty/no-asset blocks have no vector. Keep owner/source restrictions, ordered assets and queryNodes behavior intact.

Native-image candidates must not then be dropped or converted to a description-only packet by final selection. Make this narrow supporting cutover:

- `core/model/search.py::SearchSelectionCandidate.content` becomes `material: MultimodalContent`; keep `position` and `is_exercise`.
- `SearchSelectionInput.query` stays a text string because public search query parameters remain text.
- `SearchSelectionModule.forward/aforward` accept required `content_images: list[dspy.Image]`; its signature judges candidates using their native material, with identical relevance/position semantics.
- `SearchSelectionService.select` adds required `content_images: list[ContentImage]`. Every whole-prefix fit counts the request JSON plus only the catalog images actually referenced by that prefix; use a fresh compact builder/projection per prefix so rejected tail images are neither loaded nor charged. Final dispatch loads just the selected prefix's images under the existing `ImagePayloadBudget` convention and sends native `dspy.Image` values. Retain existing fail-on-oversized-first-candidate and contiguous-prefix behavior.
- `SearchSelectionSettings.inference` becomes `VisionInferenceSettings(max_tokens=1024, num_retries=0)` and adds existing `ImagePayloadSettings`; retain 16384 input limit and 256 margin. `build_search_selection_service` supplies token and image limits. Migrate all text candidate constructors/examples to `material=MultimodalContent(text=...)` and no-image calls to `content_images=[]`.
- Source selection receives the same source assets as reranking, remapped for its final shortlist. Candidate image loading is internal retrieval, not teacher inspection; do not issue teacher citation receipts or mark workspace image inspections.

### 6. Perform the explicitly approved fresh-graph cutover once

First finish deterministic verification and confirm both new provider credentials work with real image inputs. The authorized reset boundary is the **entire single database named by `Settings().database.database` at its configured URI**, including any legacy data in that same database; it is not the `system` database, other databases on the server, another URI or the filesystem. Print the nonsecret URI/database and node count before the operation; refuse a `system` target. Execute the one-time Cypher `MATCH (n) DETACH DELETE n` through that explicitly selected database and drop only vector indexes named in `database/schema.py::VECTOR_INDEX_NAMES`, then let ordinary schema creation rebuild them at dimension 1024. User explicitly selected clearing the configured graph, including source, session and learner data, rather than a preservation migration. Do not put reset behavior in startup, add a reset flag or ship a permanent destructive migration command. Do not delete artifact files, edit `.env`, print secrets, launch a replacement database, or reset any other instance.

If the configured database cannot be reached or either new credential is absent/rejected, do not clear data; finish all code/deterministic work and report the exact live prerequisite. After successful reset, ingest fresh fixture sources; never reuse old vectors or source model identities even if their dimension is 1024. Keep existing source snapshot ownership and embedding-profile checks. No new per-node model registry or background re-embedding workflow is needed for this authorized fresh graph.

## Verification

### Deterministic behavior and regression coverage

Use the existing uv environment from repository root, with network/provider calls intercepted by `httpx.MockTransport` for IONOS and the installed LiteLLM HTTP transport for embedding wire integration. No live keys or model downloads in permanent unit tests: construct tiny deterministic tokenizer JSON files using the existing `tests/test_kms_token_counting.py::_write_tokenizer` pattern and real small Pillow images.

1. **Shared material + provider wire:** text-only, image-only, text+two-images and image-bearing query examples use valid catalog ordinals. Through the public Embedder/Reranker Protocols, inspect actual outgoing JSON after library/provider transformation, decode inline bytes, and verify correct text/asset ownership and image order. A repeated catalog image referenced by two materials is encoded once internally but appears in both providers' per-item contents and is charged twice across those inputs. No path, backend UUID or Base64 text enters text-token counting. Valid provider responses can be in non-input order; vectors return in input order, rerank indexes still refer to original candidates.
2. **Limits + batching:** exact inclusive boundaries fit; one extra input token/image byte/pixel fails before network; eligible whole items split by soft targets and hard request counts/body bytes. An oversized soft singleton that fits hard limits is sent alone; hard singleton failures do not truncate or fall back to descriptions. Test that query image costs repeat per rerank pair and that top_n applies after index remapping across batches. Exercise empty batches with no tokenizer/image/network access; real HTTP/file errors propagate and cancellation closes the IONOS pool.
3. **Token semantics + ownership:** migrate `tests/test_kms_token_counting.py`, `test_kms_context_window.py`, `test_kms_multimodal_content.py` and `test_kms_source_context.py`. Preserve local generative raw token counts, tiktoken special-token behavior, lazy loading, conservative per-block image estimates and LLM budget limits. Test Voyage's 560-pixel rule/minimum estimate separately from IONOS 32-pixel estimates; never assert these are exact billed-token totals. Factories always pair the correct tokenizer/model with the implementation; generative-only operations must not acquire either retrieval tokenizer.
4. **Source ingest:** extend `tests/test_kms_source_embedding.py` and `test_kms_source_processing_graph.py` with text, caption+figure, image-only assets and truly empty blocks. Only the empty/no-assets block is skipped; source block/asset UUIDs, page order and existing vectors remain intact, and native image bytes reach embedding. Do not carry forward tests whose only purpose is malformed-provider output repair or integer-to-float coercion under the perfect-output contract.
5. **Retrieval:** extend `tests/test_kms_source_search.py`, `test_kms_knowledge_search.py`, `test_kms_learner_observations.py`, `test_kms_search_selection.py` and semantic-hub tests. A text query must retrieve an image-only source block with a stored vector, send its image to the reranker and final selector, and preserve ownership/source/page/index identity through the result. Propositions/reasoning/observations still use text-only material through the same multimodal Protocols. Verify no inspection receipt is granted merely for internal retrieval.
6. **Factories/application/runtime:** adapt `tests/test_kms_embedding.py`, `test_kms_reranking.py`, `test_kms_reranker_batching.py`, `test_kms_application.py`, `test_kms_source_processing_application.py`, `test_kms_research_composition.py`, `test_kms_local_models.py` and config tests. Remove obsolete local retrieval tests and preserve LLM router profile switching, rollback and cleanup tests. Assert one matching resource acquisition per invocation and close behavior on failure/cancellation; source-learning-only execution acquires neither hosted capability. `SourceAdmissionService` profile and schema dimension use the same embedding descriptor.
7. **No old paths:** use specialized repository search to ensure no KMS caller imports the removed flat modules, uses `EmbeddingClient`/`RerankerClient`, selects retrieval tokenizers from `LanguageModelTokenizers`, calls local runtime `.embed/.rerank`, or sends `list[str]` to the new capability interfaces. No re-export/alias imports mask leftover callers.

Run the narrow suites for changed behavior first:

```text
uv run pytest -q tests/test_kms_token_counting.py tests/test_kms_embedding.py tests/test_kms_reranking.py tests/test_kms_reranker_batching.py tests/test_kms_source_embedding.py tests/test_kms_multimodal_content.py tests/test_kms_source_context.py tests/test_kms_source_search.py tests/test_kms_search_selection.py tests/test_kms_knowledge_search.py tests/test_kms_learner_observations.py tests/test_kms_application.py tests/test_kms_source_processing_application.py tests/test_kms_local_models.py
uv run pytest -q tests/test_kms_*.py
```

### Actual runtime proof and authorized data reset

Run a temporary Python smoke through the actual factories, not direct provider functions. Read `Settings()` without printing secrets; required new credential variables are `KMS_EMBEDDING__API_KEY` (Voyage) and `KMS_RERANKER__API_KEY` (IONOS), and the configured tokenizer repositories/files must be reachable. New provider keys are not interchangeable with the previous OpenRouter/Voyage rerank keys.

- Embed a packaged native diagram and an unrelated native image, each with a short caption, plus a text-only item, in document mode. Embed a text query and an image+text query in query mode. Observe real finite 1024-coordinate vectors and order; record provider/model identity, not credentials.
- Rerank those actual candidate materials using the selected IONOS component with a mixed query; print returned indexes and scores, confirming no image substitution or unsupported type. This is evidence of real request execution, not a guaranteed quality benchmark.
- After both calls succeed, carry out step 6's explicitly approved reset on the configured database only. With a new user/source identity, run public `application.ingest_source` using `/tmp/kms-web-ocr-PKWtNZ/web-capture.pdf` if it exists; otherwise build a temporary PDF with real text and a packaged figure using the already-installed PDF tooling. Run public `search_sources`, reload the source through the production repository, and confirm surviving figure/text blocks and 1024-dimensional vectors are persisted/searchable. Preserve original captures. Source processing still requires its normal Mistral OCR and generative model credentials; report those failures separately.
- Run existing opt-in Neo4j integration tests only against the now-authorized configured fresh graph using `KMS_NEO4J_IT=1` and its existing `KMS_DATABASE__URI`, `__USERNAME`, `__PASSWORD`, `__DATABASE`, with `tests/test_kms_source_search_integration.py`, `tests/test_kms_vector_search_integration.py` and `tests/test_kms_immediate_source_storage.py`. These synthetic-vector tests are database evidence, not hosted-model evidence.
- If any live prerequisite is absent, finish deterministic and actual-library/mock-transport smoke and record exactly which real calls/reset/ingestion were not executed. Never claim a fake client proves hosted modality support.

## Critical files and anchors

- `src/kms/application.py::_application_resources` — one owner for capability lifetimes and a single embedding descriptor used by schema setup.
- `src/kms/composition/teaching.py` normalization output budgets — currently reselects embedding capacity/tokenizer independently; replace that dependency without deleting the still-active normalizer.
- `src/kms/node/source_embedding.py::embed_source_pages` — current nonempty-text eligibility omits image-only material; replace with exact shared source projection.
- `src/kms/service/source_evidence.py::SourceSearchService.search_sources` — native candidate assets must survive reranking and final selection rather than reverting to description-only text.
- `src/kms/local_models/runtime.py::LocalModelRuntime/RuntimePredictor` — remove obsolete retrieval implementation while retaining LLM resource rollback and cancellation semantics.

## Assumptions and contingencies

- User selected hosted retrieval and authorized a fresh graph. Do not preserve a text-only local retrieval backend, add a Hugging Face reranker serving stack, or introduce an automatic re-embedding migration. Hugging Face tokenizer files are lightweight vocabulary resources, not local model inference.
- Text/image material, provider outputs and indexes are perfect. Resource limits, missing credentials, file integrity, authorization, HTTP errors and cancellation are real operational contracts and remain visible. No guessed vectors, repair branches, optional modality downgrades or new retries.
- Use the installed LiteLLM Voyage multimodal transform with explicit image_base64 parts and test its actual outgoing request. If a live key routes to an unavailable endpoint/model, report the failure; do not replace the pinned model with a text-only model or silently use another credential.
- Fresh-graph deletion is a one-time execution operation gated on actual new-provider smoke success and configured DB reachability. If prerequisites fail, leave graph data and credential files untouched and report the blocked live cutover.

