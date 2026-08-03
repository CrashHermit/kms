# KMS — Session Handoff

Last updated: 2026-08-03 · Branch: `claude/neo4j-connectivity-68t3zy`

## Project summary

**KMS** turns documents (PDFs via Mistral OCR) into a knowledge graph (Neo4j).
The graph has a provenance spine — `:Node` vertices in document order — with
semantic tiers hanging off it:

- `:Statement` / `:Procedure` hubs — pedagogical units, membership by `:MEMBER_OF`
- `:Equation` / `:Variable` — node-anchored artifacts *(under review, see ADR 0001)*
- `:Fact` — self-contained atomic facts, the intended input to the concept pass

**Design mode: GREENFIELD.** Existing code is a reference, not a constraint.
Evaluate every architectural option from first principles. When the current code
conflicts with a cleaner design, surface the conflict and let first-principles
reasoning decide — and where possible, *measure* rather than argue.

## Architecture conventions

### Dependency injection

Central services are env-driven, cached singletons injected at construction time:

- `kms.core.llm` — text LM (`deepseek/deepseek-v4-pro`) and corrector/vision LM
  (`qwen3-vl-235b` via OpenRouter), both `dspy.LM`. `llm.gate(n)` bounds
  concurrency (`KMS_MAX_CONCURRENT_CALLS`, default 16).
- `kms.core.embeddings` — embedding client (`voyage-multimodal-3.5` via
  OpenRouter, direct httpx, no `dspy.Embedder`). **Currently orphaned — see
  Known defects.**
- `kms.graph.db` — quarantined Neo4j client; the `session_factory` callable is
  injected into graph stages.

### Pipeline (LangGraph)

A single straight path over `state.State` channels. Extraction passes accumulate
results in state; the terminal `IngestionPersisterNode` writes everything in one
idempotent pass (MERGE on deterministic uuids).

**Actual stage order, verified against `pipeline.py` edges:**

```
corrector -> formatter -> extractor -> seam_even -> seam_odd -> splitter
          -> instruction_finder -> instruction_distributor
          -> pedagogical_component_finder -> hub_builder
          -> equation_variable -> atomic_facts -> entity_extraction
          -> ingestion_persister
```

Passes come in two shapes, and the difference is the dominant cost factor:

- **Windowed** (`atomic_facts`, `entity_extraction`) — the stream is cut into
  fixed adjacent windows (~2000 chars) and each window is one LLM call. Cheap.
- **Per-node** (`equation_variable`) — one router call per content node plus
  conditional extractor calls. The only O(nodes) pass; measured at 12–13× the
  fact pass on the same document.

### Graph tier

Every semantic tier has the same three-part shape:

1. **Mapping module** (`graph/<tier>.py`) — pure: deterministic uuid, property
   maps, edge pairs. No driver imports.
2. **Writer** (`graph/writer.py`) — batched `MERGE`/`UNWIND` over the injected
   session factory.
3. **Schema** (`graph/schema.py`) — idempotent DDL, constraints + indexes.

Edge conventions:

- `(:Node)-[:NEXT]->(:Node)` — provenance chain, every node, nothing skipped
- `(:Source)-[:HEAD]->(:Node)` — the chain's root
- `(:Node)-[:EVIDENCE_FOR]->(:Statement|:Procedure|:Fact)` — raw material →
  construct
- `(:Node)-[:MEMBER_OF]->(:Statement|:Procedure)` — hub membership
- `(:Node)-[:HAS_EQUATION]->(:Equation)`, `(:Node)-[:HAS_VARIABLE]->(:Variable)`
- `HAS_*` points OUT of the construct; `EVIDENCE_FOR` points INTO it

### Coding style

`docs/STYLE.md` (Google-adapted Python), enforced with `ruff format .` and
`ruff check .`. Module-qualified imports only (`from kms.core import models`,
never `from kms.core.models import ASTNode`); `Recorder` is the established
exception. 80-char lines, single quotes, Google docstrings.

## Module inventory

| Package | Modules |
| --- | --- |
| `kms.core` | `llm`, `embeddings`, `models`, `state`, `walker`, `logs`, `recorder` |
| `kms.ingestion` | `ocr`, `corrector`, `formatter`, `extractor`, `seam_merger`, `splitter`, `instruction_finder`, `instruction_distributor`, `pedagogical_component_finder`, `hub_builder`, `variable_extractor`, `atomic_fact_extractor`, `entity_extractor`, `fact_embedder` |
| `kms.graph` | `db`, `nodes`, `statements`, `procedures`, `instructions`, `equations`, `variables`, `facts`, `schema`, `writer`, `queries`, `persister` |
| root | `pipeline`, `tui` |

## Design decisions

1. **Facts are the primitive.** Everything — claims, equations, variable
   bindings — is decomposable into atomic facts (ATOM's definition: short,
   self-contained, exactly one piece of information).

2. **Living schema, not a registry.** Relation types are created at edge-creation
   time; canonicalization happens at write time by embedding similarity against
   existing types. The schema IS `MATCH ()-[r]->() RETURN DISTINCT type(r)`.

3. ~~**Equations/variables stay sibling tiers on `:Node`.**~~ **Superseded by
   [ADR 0001](adr/0001-facts-as-the-only-semantic-primitive.md).** Measured
   double extraction against the fact → entity → concept path. Equations belong
   in the concept layer (cross-document identity); symbol bindings stay local.

4. **Everything in stored strings is delimited LaTeX** (`$...$` inline,
   `$$...$$` display) — never plain text or Unicode math. This is an *identity*
   rule, not cosmetics: symbol strings are part of `variable_uuid`, so `X` and
   `$X$` become two vertices for one thing.

5. **No `dspy.Embedder`.** Direct httpx over an OpenAI-compatible endpoint, for
   provider freedom, multimodal-ready input, and controllable batching.

6. **Embedding is its own stage, not part of writing.** The embedder wants one
   large batch; extraction uses small windows.

7. **Two Neo4j transports, one call shape.** `db.session()` yields a session
   whose `run(cypher, **params)` behaves identically over Bolt and over the HTTP
   Query API. `NEO4J_TRANSPORT=auto` (default) probes Bolt once and falls back.

## What this session established

All figures from live runs against real gold pages, N=3 per arm where an
A/B was involved. Both upstream passes are nondeterministic, so ranges matter.

| Finding | Evidence |
| --- | --- |
| The eq/var artifact context gives the fact pass **no** measurable benefit and **costs ~12% recall** | 89/87/91 facts without vs 85/71/80 with; perfect separation, p=0.05 permutation |
| ...including in the one case it was designed for (a symbol bound in an earlier window) | ~25 cross-window symbol uses per run in both arms; ~3 carry the meaning in both |
| The equation `name` field's premise doesn't hold | 7 of 34 equations named, and those were `'union definition'`, `'intersection definition'` — descriptive labels, 0 adoptions by the fact pass |
| The eq/var pass dominates cost | 92 LLM calls vs 7 for the fact pass on the same 57-node document |
| The router barely pays for itself and is nondeterministic | 56 router calls to gate 36 extractions; `has_variable` fired on 6 nodes then 3 on identical input |
| The entity pass was minting variables as entities | 44% of mentions on a proof page were bare symbols — fixed in `13568d5`, now 5 of 66 |
| Entity per-fact provenance is solid | 0 out-of-range `fact_index` across every run, including multi-batch global indexing |

The full argument and the proposed redesign are in
[ADR 0001](adr/0001-facts-as-the-only-semantic-primitive.md).

## Known defects

1. **The fact embedding stage is unwired (regression).** Commit `8cf48d4`
   ("Added entity extractor") removed the `fact_embedding` node, its edges, the
   `embeddings` import, and the `embedding` field from `models.AtomicFact`.
   `kms.ingestion.fact_embedder` and `kms.core.embeddings` are now imported by
   nothing, and `graph/facts.py` / `graph/writer.py` still document embeddings
   they never receive. **This blocks the concept pass**, which clusters fact
   vectors. Nothing crashes — facts persist without vectors.

2. **`variable_uuid` collides.** The key is
   `(source, node_id, symbol, value)`; a node that binds one symbol several ways
   with no explicit value collapses to one vertex, last write wins. Observed on
   two documents (`$X$` bound three ways in one node).

3. **Worked derivations fragment into per-step facts.** The fact pass emits one
   fact per algebra step, and the entity pass then names each transient
   expression as an entity. Not fixable from the entity prompt — it sees a lone
   step-fact and correctly reports what that fact is about.

## What's next

**The concept pass** — turn `atomic_facts` into a canonical, deduplicated
`:Concept` layer:

1. Cluster fact vectors (needs defect #1 fixed first).
2. Name and type each cluster with one LLM call over the cluster's facts.
3. Dedup across books: embed the candidate name, search
   `queries.existing_concepts()` via `top_k`, merge above threshold else mint.
4. Persist `:Concept {name, description, embedding, source}` with
   `(:Concept)-[:GROUNDED_BY]->(:Fact)`.

It is the **first pass that reads the graph mid-pipeline** — inject
`session_factory` + `neo4j_configured` exactly as the persister does.

Note the mention-extraction question is now partly answered: `entity_extractor`
implements Option A (per-fact LLM). Whether clustering still needs it, or can
work from fact vectors alone, is untested.

**ADR 0001 sequencing**, cheapest and most independent first:

1. Drop the artifact fields from the fact signature (recovers ~12% recall).
2. Fix the `variable_uuid` collision.
3. Window the symbol pass, delete the router.
4. Fold equations into facts, delete `:Equation` — **before the concept pass
   persists anything**, or this becomes a migration.
5. The `:Symbol`→`:Concept` binding model, after concepts exist.

## Open questions

1. **Is the symbol pass worth running at all?** Its only unique output is
   `:Variable.symbol` — `meaning` is concept-shaped and duplicated by the
   entity/concept path. Demonstrate the value of a queryable decoder ring
   against a real retrieval task before rebuilding it.
2. **Concept embedding strategy:** name alone, or name + grounding text?
3. **Cluster granularity:** merge threshold — tune after a live run.
4. **Does the derivation-granularity fix work?** Untested hypothesis from two
   pages; A/B it before trusting it.
5. **Does mention extraction survive clustering?** See above.

## Running things

**Environment.** Copy `.env.example` to `.env`. Required for a full run:
`MISTRAL_API_KEY` (OCR), `OPENROUTER_API_KEY` (corrector, embeddings),
`DEEPSEEK_API_KEY` (text passes).

**Neo4j.** `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE`.
In sandboxes that only allow egress on 443 (CI, Claude Code web sessions) the
Bolt port is unreachable; `NEO4J_TRANSPORT=auto` handles this automatically, and
`http` skips the ~5s Bolt probe. `NEO4J_DATABASE` is not always `neo4j` — on the
current Aura instance it is the instance id.

**Tests.** `uv run pytest` — 331 passed, 2 skipped, no network. The suite stubs
dspy/pydantic/langgraph/neo4j when absent. The live Neo4j test is opt-in behind
`KMS_NEO4J_IT=1`.

**Lint.** `ruff format .` and `ruff check .` — both clean.

**Real data for experiments.** `data/gold/extractor/pages/*.md` (22 single pages
with an `index.json` of provenance) and `data/gold/corrector/real/<book>_p*/`
(contiguous multi-page runs — useful when a test needs several windows).
`tests/fixtures/books/*.pdf` are the source PDFs.

## Starting a new session

Read this file, then `docs/adr/0001-*.md` for the live architectural question,
then `docs/STYLE.md` before writing code. The measurements above were expensive
to obtain — prefer extending them over re-deriving them, and prefer measuring
over arguing when a design question has an observable answer.
