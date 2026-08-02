# Next Step for KMS — Research Synthesis & Direction Plan

Status: **DRAFT — direction not yet chosen.** This plan records what the six research
documents say, where the kms project stands against them, and the candidate next steps.
The final recommendation depends on decisions the user must make (see *Open questions*).

---

## Context

The user asked: *research our current `.md` research files and figure out what should be
our next step.* The research corpus is six full-text paper scrapes under `docs/`:

| File | Paper | Core contribution |
|---|---|---|
| `docs/autoschemakg/markdown.md` | AutoSchemaKG (arXiv:2505.23628) | Autonomous KG construction: entity/event/concept kinds + **schema induction** (no predefined schema); web-scale ATLAS KGs; QA + factuality gains |
| `docs/papers/raga.md` | RAGA | **Agent-based** KG construction: Read–Search–Verify–Construct ReAct loop (LangGraph), full CRUD toolset, KG↔vector sync, evidence-anchored provenance, RRF fusion retrieval |
| `docs/papers/dial-kg.md` | DIAL-KG | **Incremental** construction: dual-track extraction (triple vs event), Meta-Knowledge Base governance (evidence/logical/evolutionary-intent checks), schema evolution, transactional soft-deprecation |
| `docs/papers/evorag.md` | EvoRAG | **Feedback-driven refinement**: response feedback → path utility → triplet contribution scores; relation fusion + suppression; hybrid priority retrieval |
| `docs/papers/atom.md` | ATOM | Atomic fact decomposition → exhaustive, **stable** extraction; LLM-free parallel merging; dual-time temporal KG |
| `docs/papers/autograph-r1.md` (+`.txt`) | AutoGraph-R1 (arXiv:2510.15339) | **RL for construction**: optimize graph building for downstream retrieval performance (GRPO; Knowledge-Carrying & Knowledge-Indexing rewards) |

All six are "raw" scrapes — none has been annotated with *what it means for kms*.

---

## What the papers say, distilled (research synthesis)

### The converging thesis: a KG earns its keep only when it is READ BACK

Every paper's headline result is a **downstream measure** — Answer F1 / Evidence F1
(RAGA), QA accuracy (AutoGraph-R1, AutoSchemaKG), reasoning accuracy (EvoRAG). None
evaluates a graph in isolation. The shared shape is a **closed loop**:

```
construct ──► read / retrieve ──► use (QA / review) ──► feedback ──► refine
   ▲                                                                │
   └────────────────────────────────────────────────────────────────┘
```

- **RAGA**: fusion retrieval (vector + graph, RRF) beats vector-only (0.587→0.615
  Answer F1) and KG-only (0.526); graph value is *realized in fusion with vectors*.
  Also: "quality over quantity" extraction (leaner extraction improved fusion +17%,
  cut construction time 30%); evidence-anchored provenance; CRUD + `mark_for_review`
  / `create_todo` for uncertain knowledge; schema auto-discovery.
- **EvoRAG**: feedback on *responses* is the supervision signal for *graph* refinement —
  contribution scores on triplets, relation fusion (shortcut edges), suppression
  (downweight, don't delete). This is the graph-level analog of what kms's FSRS
  dependency promises at the review level.
- **DIAL-KG**: a static one-shot graph rots. Incremental batches + a Meta-Knowledge
  Base + governance adjudication (evidence / logical consistency / **evolutionary
  intent → soft deprecation**) keep it truthful. Schema evolution replaces predefined
  schemas.
- **AutoGraph-R1**: construction should be optimized for *functional utility* (is the
  gold answer deducible? are gold passages retrievable?), not intrinsic "niceness".
  Naive F1 reward is brittle — use functional rewards.
- **ATOM**: decomposition into atomic facts before extraction fixes exhaustivity
  (forgetting in long contexts) and run-to-run stability; embedding+threshold merging
  (no LLM calls) makes it scale.
- **AutoSchemaKG**: the engine thesis kms already adopted in `UNIFIED-KG.md` /
  `GENERALIZATION.md` — schema induction as the default, hand-authored profiles for
  domains you care about.

### What this means for kms (gap analysis)

| kms today | The papers' target |
|---|---|
| Graph is **write-only** — nothing reads it back; no retrieval, no query path, no QA | Retrieval is where every paper shows value (RAGA fusion; EvoRAG hybrid; AutoGraph-R1 functional rewards) |
| Statement/Procedure hubs are **untyped** (role only — statement vs procedure) | Open `type` induction (AutoSchemaKG; DIAL-KG schema evolution; RAGA schema auto-discovery) |
| **No concept layer** (`:Concept`, `:INSTANCE_OF`, `:DEPENDS_ON`) — explicitly "dark" since the entity-layer rip-out | Conceptualization `φ`/`ψ` (AutoSchemaKG); concept-level prerequisite reasoning (kms's own curriculum goal) |
| **No entity/relation semantics** — no cross-block or cross-book edges | `:REFERENCES`, `:DEMONSTRATES`/`:PRACTICES`, canonical hubs (UNIFIED-KG edge design) |
| One-shot per book; **no update/deprecation path** | Incremental + governance + soft deprecation (DIAL-KG); feedback refinement (EvoRAG) |
| FSRS is a declared dependency but **unwired** | The metacognitive review loop — EvoRAG-style feedback feeding FSRS scheduling |
| Extraction quality has gold sets (corrector/extractor); **graph quality is never measured** | RAGA's Answer/Evidence F1 protocol; AutoGraph-R1's knowledge-indexing reward |

---

## Candidate next steps

1. **Retrieval / read side (RAGA-style)**
   Embed nodes (chunks) → graph expansion over `:NEXT`/`:MEMBER_OF` → RRF fusion →
   a query surface (TUI) + a QA eval over the fixture corpus (Answer F1 / Evidence F1).
   Buildable on what exists today; creates the first *graph-level* measurement; the
   prerequisite for feedback (you must read before you can refine).
2. **Semantic layer (type + concepts + relations)**
   Restore open `type` on hubs, add `:Concept` + `:INSTANCE_OF` + `:DEPENDS_ON`
   (probes already validated these on a frontier model), then cross-unit edges
   (`:DEMONSTRATES`, `:PRACTICES`, `:REFERENCES`). Makes the graph "knowledge" again;
   GENERALIZATION.md step 1–2.
3. **Feedback / refinement loop (EvoRAG + DIAL-KG + FSRS)**
   Review feedback → contribution scores on graph elements → relation fusion /
   suppression; governance + soft deprecation for incremental re-ingestion; wire FSRS.
   The "metacognitive learning" the project's description promises; needs 1 (and
   partly 2) first.
4. **Research-note pass only**
   Annotate each paper with a "what this means for kms" section (the current files
   are raw scrapes) before deciding anything.

Recommended default (pending user confirmation): **1 first, then 2, then 3** —
retrieval + eval harness unblocks the evidence-based decisions behind 2 and 3, and
its vector half (chunk embeddings, RRF) is reusable no matter which semantic layer
lands next.

---

## Files to modify (for the recommended direction, once confirmed)

To be filled in after the direction is chosen. Relevant existing code for reuse:
- `src/kms/graph/` — `db.py` (async driver + HTTP transport), `nodes.py`,
  `statements.py`, `procedures.py`, `equations.py`, `variables.py` (uuid/property/edge
  mappers), `writer.py`, `persister.py`, `schema.py`
- `src/kms/core/llm.py` — LM config, key loading (`MISTRAL`/`OPENROUTER`/`DEEPSEEK`)
- `src/kms/pipeline.py` — LangGraph wiring; `src/kms/tui.py` — user surface
- Fixture books: `tests/fixtures/books/`; gold sets in `data/gold/`
- Design docs in git history (not on main): `docs/GENERALIZATION.md`, `docs/UNIFIED-KG.md`,
  old `docs/HANDOFF.md` / `CLAUDE.md` — the previously recorded roadmap

## Verification (to be filled in)

---

## Open questions for the user

1. **Priority lens** — which matters most for the *next* step: user-facing value
   (query the graph / review flow), measurable progress (graph-level eval), or
   architectural completeness (semantic layer)?
2. **Deliverable scope** — is this session meant to produce (a) a decided roadmap /
   plan only, or (b) an actual implementation of the chosen next step?
3. **Embedding provider** — kms has no embedding infra today; is OpenRouter's
   `text-embedding-3-large` (reuses the existing key, per GENERALIZATION.md's frugal
   baseline) acceptable, or is another provider preferred?
4. **Second domain** — is physics/biology ingestion on the horizon? (GENERALIZATION.md
   gates the engine/profile split on the rule of three.)
5. **Research notes** — should the papers also be annotated with per-paper "what this
   means for kms" notes as part of this work, or stay as raw scrapes?
