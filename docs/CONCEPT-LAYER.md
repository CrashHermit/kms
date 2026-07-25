# Concept layer — conceptualization and fusion (deferred)

The design for the **concept layer**, the one tier of `SCHEMA.md` that is currently **dark**.
Nothing here is built. The schema, the built pipeline, and the rip-out that produced them are in
`SCHEMA.md` and `REBUILD.md`; this doc is the spec for what fills the empty tier.

**Why it is dark.** The concept layer's only source was `Entity.field`, drawn from AutoMathKG's
closed seven-value mathematical-field taxonomy. That taxonomy is deleted, so `graph/concepts.py`
was gutted down to the half that is *not* AutoMathKG — concepts and the `φ` mapping are
AutoSchemaKG's contribution — leaving `normalize_concept`, `concept_uuid`, `concept_properties`
and `concept_batch`. Identity survives; nothing writes through it yet.

**Why it matters more than it used to.** Block-to-block relation extraction is gone (it was
AutoMathKG's `refs`, not an AutoSchemaKG feature — their $P_{EE}$/$P_{EV}$/$P_{VV}$ all operate
*within* a text segment). So concepts are now the **only** connective tissue between blocks. Until
this layer is built, a book's blocks connect to nothing outside their own `:Source`.

---

## What to build

### 1. Flat multi-tag conceptualization (`:INSTANCE_OF`)

Adapt AutoSchemaKG's schema-induction prompts (their Figs 5/6/7, in `docs/autoschemakg/markdown.md`
§B.2) — **relaxing the "1–2 word" constraint** the 8B needed (a frontier model gives precise,
multi-word concepts). For each node, generate several concept phrases spanning **specific →
general**, and draw:

```
(:Entity)-[:INSTANCE_OF {phrase}]->(:Concept)
(:Procedure)-[:INSTANCE_OF {phrase}]->(:Concept)
(:Act)-[:INSTANCE_OF {phrase}]->(:Concept)
```

Multi-granularity comes from **attaching several flat tags of differing generality to each node**
(a coarse tag shared by many, a fine tag by few) — *not* from a concept tree. The layer is flat:
`:INSTANCE_OF` is its only edge type. `:BROADER`, MSC anchoring and `:DEPENDS_ON` are all cut.

**The induced phrase rides on the edge**, not on a mention node. A concept mention has no content
beyond the string, so nodehood would multiply the graph by (entities + procedures + acts) × ~3 for
no gain, and fusion can re-point an edge just as easily as it could re-point a mention.

**Context enhancement (AutoSchemaKG §B.2.2).** Feed a node's structural neighbours into its
conceptualization prompt so abstraction is grounded in its role — an `:Act`'s parent `:Procedure`
and grandparent `:Entity`, a `:Procedure`'s target `:Entity` and child `:Act`s. This is the
"Pass 2 uses Pass 1's hierarchy as context" step, and it is why the entity layer had to land first.

### 2. No corpus-global pass is needed

Phrase *generation* is per-node and fits inside the existing per-book LangGraph run. Cross-book
*convergence* is automatic: `concept_uuid` is a global uuid5 over the normalized name with no
source prefix, so a MERGE from book B lands on book A's node. Only **fusion** (below) needs a
corpus-wide execution mode, which the pipeline does not currently have.

---
## Evidence: the probes (DeepSeek V4 Flash, temp 0)

Two throwaway probes over curated math/physics/biology entities. Small (≈9 entities, one run) —
a signal, not a benchmark — but consistent.

**Probe 1 — concept induction.** The 8B "technical domain" limitation did **not** reproduce:
- Specific, not coarse: `normal subgroup` → `normal subgroup, group theory, abstract algebra`.
- Polysemy handled from context: the linear-map `kernel` → `kernel of a linear map, null space`
  (not integral/stats kernel); `normal subgroup` never drifted to normal vector/distribution.
- Inference beyond the literal text: an unlabelled `[[2,1],[1,2]]` exercise was tagged
  `symmetric matrix` — it noticed the matrix is symmetric.
- Cross-domain with **zero profile**: physics → `second law of thermodynamics, rotational
  dynamics`; biology → `biological catalyst, population genetics, evolutionary mechanism`.
- One blemish: a definition over-decomposed into its *parts* (`addition, scalar multiplication…`)
  rather than what it is *about* — a prompt-tuning fix.

**Probe 2 — `:DEPENDS_ON`.** Pairwise prerequisite judgments were crisp (8/8 defensible),
including **necessity reasoning** an 8B could not do: `compactness depends_on metric space?` → *No,
topological spaces suffice* (correct — compactness is topological, not metric). Reverse pairs and
unrelated pairs were correctly rejected (clean DAG behaviour). Batch/free-list concept-level
induction was good but occasionally listed a **co-concept** as a prerequisite (`eigenvector` for
`eigenvalue`) or drifted to *related* rather than *strict* prerequisites — hence: prefer
pairwise/grounded, define the edge tightly.

**Takeaway:** concept induction and `:DEPENDS_ON` are reliable enough with a frontier model to be
the **general path**; hierarchy quality rides on *method* (pairwise/grounded, not batch).

---

## Concept convergence: the embedding / fusion decision

Conceptualization *creates* concept phrases; it does not merge them ("linear algebra" vs "Linear
Algebra" vs "vector spaces"). Fusion is the tier that does — the dedup/canonicalization loop
AutoSchemaKG lacks. With the entity mention/canonical split deleted, fusion now operates on **one**
hub set: the `:Concept` nodes. It re-points `:INSTANCE_OF` edges onto the surviving hub, and the
edge's `phrase` property preserves what was originally extracted, so a bad merge is recoverable
without a mention tier. Design:

- **Division of labour:** the **embedder** does *candidate retrieval* (recall); a **conservative
  LLM judge** does the *merge decision* (precision), biased **against** merging (a duplicate hub is
  cheap; a wrong merge corrupts every reference routing through it). Benchmark false-merge rate on
  known-collision terms ("normal", "regular", "kernel", "field").
- **Embedder choice is unresolved and cannot be settled from public benchmarks** — there is **no
  independent math-specific embedding benchmark** (standard MTEB has no math category), and vendor
  charts are self-graded (zembed-1's "#1" is its own re-annotated benchmark; it is absent from the
  neutral MTEB leaderboard). Decide by a **bake-off on our own book**. Candidates:
  - **OpenRouter text model** (e.g. `openai/text-embedding-3-large`) — reuses the existing
    `OPENROUTER_API_KEY`, zero new provider; the frugal baseline.
  - **zembed-1** (ZeroEntropy) — STEM/math-tuned, hosted API (no GPU), 32k ctx, Matryoshka dims;
    text-only; needs a 4th key.
  - **voyage-multimodal-3.5** — shared text+image space (embeds figures with statements); general,
    not math-tuned; needs a 4th key. Better for *figure-aware retrieval* than for *dedup* (dedup is
    a text problem).
- **Recommendation:** build the candidate-scoring step **model-agnostic** (embedder behind a config
  swap), start with the free OpenRouter baseline, and only adopt a 4th-key model if it measurably
  beats the baseline on our entities. No GPU anywhere — hosted APIs only.
- **Note:** fusion makes concept identity **discovered** (embedding + judge) rather than
  deterministic — a real change to the persistence contract for this one layer, and the only part
  of the system needing a **corpus-wide execution mode** (each run becomes an "Input KG" fused into
  the "Existing KG"). Non-destructive: the extracted `phrase` survives on every `:INSTANCE_OF`
  edge, so a bad merge can be undone by re-pointing.

---

---

## Deferred within this layer

- **`:DEPENDS_ON`** (concept → concept prerequisite) is **cut**, not merely unbuilt. Its purpose
  was curriculum sequencing, and its grounded evidence source was the concept-level rollup of the
  entity-level reference graph — which died with the referencers. What remained was ungrounded
  pairwise LLM judgment producing an edge nothing consumed. Two signals could ground it if it
  returns: concept **co-occurrence** for candidate recall, and **first-appearance ordering** across
  the corpus (every `:Node` carries an `index`, so `MATCH (c:Concept)<-[:INSTANCE_OF]-(x)
  -[:DERIVED_FROM]->(n:Node) RETURN n.source, min(n.index)` is one aggregation). Ordering is
  acyclic by construction, which also answers the cycle risk. Note the distinction that must be
  settled first: a **definitional** prerequisite ("the derivative is defined via limits") is not
  the same as a **pedagogical** one ("what a learner needs first"), and the old design defined the
  edge as the former while justifying it by the latter.
- **Per-book concept sense** ("in Hefferon 'linear map' means X; in Axler it means Y") is the one
  feature that would justify promoting the edge property to a `:ConceptMention` node.
