# KMS — Session Handoff

## Project summary

**KMS** is a greenfield pipeline that turns documents (PDFs via Mistral OCR) into
a knowledge graph (Neo4j). The graph has a provenance spine (`:Node` vertices in
document order) with semantic tiers hanging off it: `:Statement`/`:Procedure`
hubs (pedagogical units), `:Equation`/`:Variable` artifacts, and — as of this
session — an `:Fact` layer that decomposes the document into self-contained
atomic facts, embeds them, and persists them.

**Design mode:** GREENFIELD. Existing code is a reference, not a constraint.
Evaluate every architectural option from first principles. When the current code
conflicts with a cleaner design, surface the conflict and let first-principles
reasoning decide.

## Architecture conventions

### Dependency injection (DI)

Central services are env-driven, cached singletons injected at construction time:

- `kms.core.llm` — text LM (`DeepSeek V4 Pro`) and vision LM (`Qwen3-VL-235B`
  via OpenRouter), both `dspy.LM`
- `kms.core.embeddings` — embedding client (`voyage-multimodal-3.5` via
  OpenRouter's `/embeddings` endpoint, direct httpx, no dspy.Embedder)
- `kms.graph.db` — quarantined Neo4j async driver; the `session_factory`
  callable is injected into graph stages

### Pipeline (LangGraph)

A single straight path over `state.State` channels. Extraction passes accumulate
results in state; the terminal `IngestionPersisterNode` writes everything to
Neo4j in one idempotent pass (MERGE on deterministic uuids).

Current stage order:

```
corrector -> formatter -> extractor -> seam_merger -> splitter
          -> instruction_finder -> instruction_distributor
          -> pedagogical_component_finder -> hub_builder
          -> equation_variable -> atomic_facts -> fact_embedding
          -> ingestion_persister
```

### Graph tier

Every semantic tier has the same three-part shape:

1. **Mapping module** (`graph/<tier>.py`) — pure mapping: deterministic uuid,
   property maps, edge pairs. No driver imports.
2. **Writer** (`graph/writer.py`) — batched `MERGE`/`UNWIND` over the injected
   session factory.
3. **Schema** (`graph/schema.py`) — idempotent DDL constraints + indexes.

Edge conventions:

- `(:Node)-[:NEXT]->(:Node)` — pure provenance chain, every node, nothing skipped
- `(:Node)-[:EVIDENCE_FOR]->(:Statement|:Procedure|:Fact)` — raw material →
  construct anchor
- `(:Node)-[:MEMBER_OF]->(:Statement|:Procedure)` — hub membership
- `(:Node)-[:HAS_EQUATION]->(:Equation)`, `(:Node)-[:HAS_VARIABLE]->(:Variable)` —
  artifact siblings on the node
- `(:Equation)-[:HAS_VARIABLE]->(:Variable)` — bindings inside equations
- `HAS_*` edges point OUT of the statement; `EVIDENCE_FOR` points INTO the
  construct

### Coding style

`docs/STYLE.md` (Google-adapted Python). Enforced via `ruff format .` and
`ruff check .` (see §7). Module-qualified imports only (`from kms.core import
models`, never `from kms.core.models import ASTNode`). The `Recorder` symbol
import is the established codebase exception (every pass does it). 80-char
lines, single quotes, Google docstrings, no single-char names except loop
counters.

## What was built this session

### The fact pipeline (core delivery)

| Module | Purpose |
|---|---|
| `kms.ingestion.atomic_fact_extractor` | Decomposes the node stream into `AtomicFact(text, node_ids)` — ATOM-style fixed adjacent windows (WINDOW_BUDGET=2000), domain-agnostic, no kind taxonomy, no router (facts are dense). Receives equation/variable artifact names as context. Former "ALREADY-EXTRACTED CONTEXT" anti-restatement rule removed — the concept pass will deduplicate. |
| `kms.ingestion.fact_embedder` | Batched embedding stage. Embeds every fact text in one pass via the embedding client. Enriches `atomic_facts` channel with `embedding: list[float]`. No-op when embedding isn't configured. |
| `kms.graph.facts` | `FACT_LABEL`, `fact_uuid(source, node_ids, index=block_key)`, `fact_rows`, `evidence_pairs` — mirroring `equations.py`. |
| `graph/writer.py` (+46) | `persist_facts` — MERGE `:Fact` + `:EVIDENCE_FOR` edges per node id. |
| `graph/schema.py` (+5) | `fact_uuid` constraint + `fact_source` index. |
| `graph/persister.py` (+5) | Drains `atomic_facts` channel into `persist_facts`. |
| `pipeline.py` (+33) | `atomic_facts` + `fact_embedding` nodes, edges, embedder teardown in `finally`. |

### Infrastructure modules

| Module | Purpose |
|---|---|
| `kms.core.embeddings` | Env-driven DI singleton (`Embedder` class over httpx), defaulting to OpenRouter's `/embeddings` with `voyage-multimodal-3.5` (1024 dims, live-verified). Pure `cosine_similarity` and `top_k` helpers. Key falls back to `OPENROUTER_API_KEY` for zero-config use. `is_configured()` for graceful no-op. |
| `kms.graph.queries` | Read-side counterpart to `writer`: `existing_concepts(session_factory) -> list[ConceptDatum]` (name+description+source) and `relation_types(session_factory) -> list[str]`. Named, parameterised Cypher; returns plain data, never driver objects. |

### Key design decisions along the way

1. **Facts are the primitive.** Everything — claims, equations, variable bindings — is decomposable into atomic facts (ATOM's definition: "short, self-contained, exactly one piece of information"). The fact pass extracts them all.

2. **Living schema, not a registry.** New relation types are created at edge-creation time. No separate schema registry nodes, no ACTIVE/PROPOSED states. Write-time canonicalization: compare a proposed type against existing distinct types via embedding similarity; reuse existing name when similar, mint new one otherwise. The schema IS `MATCH ()-[r]->() RETURN DISTINCT type(r)`.

3. **Equations/variables stay sibling tiers on `:Node`** (not attributes on facts, not parent/child of facts). The enrichment pass stays node-anchored and runs before facts. Facts receive artifact names as context to write richer text. Dedup across fact-text and artifact-latex is the concept pass's job.

4. **Everything in stored strings is delimited LaTeX** (`$...$` inline, `$$...$$` display). Never plain-text or Unicode math. Enforced in fact extraction prompts. Covers chemical formulas, units, and any technical notation.

5. **No dspy.Embedder.** The embedding module is a direct httpx client over an OpenAI-compatible endpoint. This gives provider freedom (Voyage via OpenRouter, any model), multimodal-ready input shape, and controllable batching. The dspy version's Embedder is text-only and limited in provider routing.

6. **Embedding as a separate pipeline stage, not part of writing.** The embedder wants one large batch; extraction uses small windows. A separate stage computes vectors once for both the concept pass and the persister.

## What's next: the concept pass

### Purpose

Turn the `atomic_facts` channel into a canonical, deduplicated `:Concept` layer.

### What it does

1. **Cluster** fact vectors — facts about the same thing have similar embeddings.
   Use `embeddings.top_k` per fact vector or a clustering algorithm.
2. **Name and type** each cluster. Per cluster, pass the cluster's facts to an
   LLM: "What is the shared concept in these facts? What should it be called?
   What type is it (term, equation, person, constant, …)?"
3. **Dedup across books.** For each candidate concept name, embed it and search
   `queries.existing_concepts()` via `top_k`. If an existing concept is similar
   above a threshold, merge; otherwise mint a new `:Concept`.
4. **Persist.** `:Concept {name, description, embedding, source}` with
   `(:Concept)-[:GROUNDED_BY]->(:Fact)` edges.

### Design decision pending

**Mention extraction method** — two options on the table:

- **Option A (per-fact LLM):** DSPy call per fact ("what entities does this fact
  mention?"). O(facts) LLM calls, high precision, RAGA-style.
- **Option B (cluster-first):** Cluster fact vectors first (embedding
  similarity), then pass each cluster's facts to one LLM call for
  naming/typing. O(clusters) LLM calls, an order of magnitude cheaper.
  AutoSchemaKG-style.

Option B is the lean preference, but the cost/precision tradeoff hasn't been
decided. The concept pass will cluster fact vectors regardless — the question is
whether there's a per-fact mention extraction step before clustering.

### New capability needed

The concept pass is the **first pass that reads the graph mid-pipeline**. Inject
`session_factory` + `neo4j_configured` (exactly like the persister) and call
`queries.existing_concepts()`.

### Pipeline placement

```
hub_builder → equation_variable → atomic_facts → fact_embedding
            → concepts → ingestion_persister
```

### Graph shape (planned)

```
(:Concept {name, description, embedding, source})
  ↑
  GROUNDED_BY
  |
(:Fact)
```

### Beyond concepts: the relation pass

The relation pass will connect `:Concept` nodes with open-vocabulary typed edges
`(head, rel_type, tail)`. The living-schema canonicalization at write time (the
`relation_types` query) will reuse existing relation names when similar.
This pass is one step beyond the concept pass and hasn't been designed yet.

## Open design questions

1. **Concept pass mention extraction:** Option A (per-fact LLM) or B
   (cluster-first)?
2. **Concept embedding strategy:** name alone? Name + grounding description?
   The embedding learning says "name + grounding text."
3. **Cluster granularity:** similarity threshold for merging — tune after a live
   run.
4. **Pipeline ordering for concepts relative to embedding:** facts are already
   embedded; concept embedding happens in the concept pass itself.

## Code state (committed and pushed)

Branch: `main`  
Remote: `https://github.com/CrashHermit/kms.git`  
Commit: `0d68b75`

```
New files (5):
  src/kms/core/embeddings.py
  src/kms/graph/facts.py
  src/kms/graph/queries.py
  src/kms/ingestion/atomic_fact_extractor.py
  src/kms/ingestion/fact_embedder.py

Modified files (substantive, 7):
  src/kms/core/models.py          (+AtomicFact dataclass with embedding)
  src/kms/core/state.py           (+atomic_facts channel)
  src/kms/graph/__init__.py       (+queries bullet in docstring)
  src/kms/graph/persister.py      (+persist_facts drain)
  src/kms/graph/schema.py         (+fact_uuid constraint, fact_source index)
  src/kms/graph/writer.py         (+persist_facts function)
  src/kms/pipeline.py             (+nodes, edges, embedder teardown, imports)
```

All files pass `ruff format .` + `ruff check .`. Embedding client live-verified
with `voyage-multimodal-3.5` via OpenRouter (1024 dims). Fact extraction
live-verified (real LLM, real embedding).

## Starting a new session

To continue from here:

1. Run `uv run ruff check .` and `uv run ruff format --check .` to verify.
2. Review the learnings in code intelligence (the durable design decisions).
3. Pick up the concept pass at the "What's next" section above.
4. Start with the cluster-first approach (Option B); switch to per-fact if
   cluster granularity proves too coarse.
