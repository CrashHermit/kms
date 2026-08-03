# KMS

Turns documents (PDFs via Mistral OCR) into a knowledge graph (Neo4j): a
provenance spine of `:Node` vertices in document order, with semantic tiers
hanging off it.

**Design mode: GREENFIELD.** Existing code is a reference, not a constraint.
When the code conflicts with a cleaner design, surface the conflict and let
first-principles reasoning decide — and where a design question has an
observable answer, **measure it rather than argue it**.

## Where things are written down

- **Decisions** live in `docs/adr/`. Read them before proposing architecture.
  They carry the evidence, not just the verdict.
- **Style** lives in `docs/STYLE.md` (Google-adapted Python).
- **Pipeline order, module inventory, channel list** are NOT documented — read
  `pipeline.py`. A prose copy of them drifted out of date within two commits;
  don't recreate one.

## Conventions

**Dependency injection.** Central services are env-driven, cached singletons
injected at construction: `core.llm` (text + corrector LMs, plus `llm.gate(n)`
for concurrency), `core.embeddings` (direct httpx, no `dspy.Embedder`),
`graph.db` (the `session_factory` callable is injected into graph stages).

**Pipeline.** LangGraph, a single straight path over `state.State` channels.
Extraction passes accumulate into state; the terminal `IngestionPersisterNode`
writes everything in one idempotent pass (MERGE on deterministic uuids).

Passes come in two shapes, and the difference dominates cost:

- **Windowed** — the stream is cut into fixed adjacent windows (~2000 chars),
  one LLM call each. Cheap. Node/fact attribution rides along as an id the
  model returns per item; this is measured-reliable.
- **Per-node** — one call per content node. O(nodes), measured at 12–13× a
  windowed pass on the same document. Prefer windowed for anything new.

**Graph tier.** Every tier has the same three parts: a pure mapping module
(`graph/<tier>.py` — deterministic uuid, property maps, edge pairs, no driver
imports), a writer (`graph/writer.py` — batched MERGE/UNWIND over the injected
session factory), and idempotent DDL (`graph/schema.py`).

Edges: `(:Node)-[:NEXT]->(:Node)` is the provenance chain, nothing skipped;
`(:Source)-[:HEAD]->(:Node)` roots it; `EVIDENCE_FOR` points INTO a construct;
`HAS_*` points OUT of one; `MEMBER_OF` is hub membership.

**Stored strings are always delimited LaTeX** (`$...$`, `$$...$$`) — never plain
text or Unicode math. This is an identity rule, not cosmetics: symbol strings
feed `variable_uuid`, so `X` and `$X$` become two vertices for one thing.

**Imports are module-qualified** (`from kms.core import models`, never
`from kms.core.models import ASTNode`). `Recorder` is the established exception.

## Commands

```bash
uv run pytest          # 331 passed, 2 skipped; no network (heavy deps stubbed)
ruff format . && ruff check .
KMS_NEO4J_IT=1 uv run pytest tests/test_graph_db_integration.py   # opt-in, live DB
```

## Environment

Copy `.env.example` to `.env`. A full run needs `MISTRAL_API_KEY` (OCR),
`OPENROUTER_API_KEY` (corrector, embeddings), `DEEPSEEK_API_KEY` (text passes).
Every stage degrades gracefully when its service is unconfigured.

**Neo4j.** `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` / `NEO4J_DATABASE`.
Two transports behind one call shape — use `db.session()`, never
`db.driver().session(...)`. `NEO4J_TRANSPORT=auto` (default) probes Bolt once
and falls back to the HTTP Query API; `http` skips the ~5s probe. Sandboxes that
only allow egress on 443 (CI, Claude Code web sessions) cannot reach the Bolt
port, so the fallback is the normal path there. `NEO4J_DATABASE` is not always
`neo4j` — on the current Aura instance it is the instance id.

## Real data for experiments

- `data/gold/extractor/pages/*.md` — 22 single pages, real OCR output, with
  `index.json` giving book/page/licence provenance.
- `data/gold/corrector/real/<book>_p*/` — contiguous multi-page runs. Use these
  when a test needs several windows (e.g. anything about cross-window context).
- `tests/fixtures/books/*.pdf` — the source PDFs.

Prefer extending existing measurements over re-deriving them; the ones in
`docs/adr/0001` cost roughly 200 LLM calls to obtain.

## Open work

- **`docs/adr/0001`** proposes the live architectural change (facts as the only
  semantic primitive) with a 5-step sequencing. Item 4 has a deadline: it must
  land before the concept pass persists anything, or it becomes a migration.
- **Regression:** commit `8cf48d4` removed the fact embedding stage — its node,
  edges, the `embeddings` import, and `AtomicFact.embedding`. `fact_embedder`
  and `core.embeddings` are now imported by nothing. Nothing crashes, but the
  concept pass clusters fact vectors, so it is blocked until this is rewired.
- **Next deliverable:** the concept pass — cluster fact vectors, name/type each
  cluster with one LLM call, dedup across books via
  `queries.existing_concepts()` + `top_k`, persist
  `(:Concept)-[:GROUNDED_BY]->(:Fact)`. It is the first pass to read the graph
  mid-pipeline: inject `session_factory` + `neo4j_configured` as the persister
  does.
