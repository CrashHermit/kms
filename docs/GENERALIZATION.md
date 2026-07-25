# Generalization — from math-first to a general (AutoSchemaKG-style) engine

Plan for taking the pipeline from **math-first** to a **general textbook knowledge-graph
engine** that also handles **physics and biology** (and any textbook), modelled closer to
AutoSchemaKG (arXiv:2505.23628). This is the concrete build-out of the "generalization era"
that `UNIFIED-KG.md` deferred — now **warranted**, because a second and third domain (physics,
biology) make the engine/profile split real rather than premature (rule of three).

**Status: design only.** Nothing here is built yet. Two live probes (below) validated the
*concept* layer of the plan; the *extraction* layer is still to be proven. **Do not rip out the
math-specific extractors before the general path is built and measured against them** — the
reason is correctness (quality parity), not migration. Greenfield removes migration cost, not
the risk of deleting validated code before its replacement works.

---

## Why

The pipeline is math-only because the entity layer hardcodes math's declarative vocabulary
(definition/theorem, the `FIELDS` taxonomy, the 9 `ACTIONS_ALL` tactics). To ingest a physics or
biology textbook we need the domain-specific parts to become either **induced** (AutoSchemaKG
schema induction) or **profile-supplied**, while the genuinely reusable parts stay put.

The key enabling change since `UNIFIED-KG.md`: **we run a frontier model (DeepSeek V4 Flash, and
can go stronger), not Llama-3-8B.** AutoSchemaKG's headline limitation — schema induction
"struggles with extremely technical domains" (their §8) — was measured on an 8B. Our probes show
it does **not** reproduce on V4 (see Evidence). So the argument that "math must stay profiled"
weakens to an *empirical* question, decided per-layer by measurement, not inherited from the paper.

---

## The organizing model: kind / genre / domain

Three layers of generality (from `UNIFIED-KG.md`), which decide what moves and what stays:

- **Kind** (universal, permanent): the node kinds — `:Entity` / `:Event` / `:Concept` — plus the
  `:Procedure` / `:Source` / `:Node` containers. Domain-free. Unchanged.
- **Genre: the textbook** (cross-domain pedagogy): worked examples, exercises, exposition,
  statements-of-fact, and the lead-in that governs a run of problems. **These generalize to any
  textbook for free.**
- **Domain: math / physics / biology** (pluggable): only the *vocabulary* — what declarative
  statements are called (definition/theorem vs law/model), the relation labels, the concept set.

Asymmetry to keep in mind: **procedures generalize better than statements.** Steps are steps
("example vs exercise" is just "shown vs withheld"); a *declarative statement* comes in
domain-specific flavours. So the procedural layer and the pedagogy layer are the most reusable;
the declarative extractor is where per-domain work concentrates.

---

## What stays, what generalizes, what we keep because it is richer than AutoSchemaKG

| Component | Layer | Fate |
|---|---|---|
| `splitter`, `instruction_finder`, `instruction_distributor` | genre | **Keep** — pedagogy, already domain-free |
| **The three per-type finders** (problem / definition / theorem) **+ their attributors + referencers** | genre + domain | **Collapse into one `block finder`** + universal attributor + procedure finder (see "Entity layer" below). The per-type walk was already triplicated; the block finder is that walk with a generalized "what is a block" clause |
| `:Procedure` / `:Event` spine (proofs/solutions → step chain) | kind | **Keep — richer than AutoSchemaKG.** Their events are flat triples; ours is a real derivation spine. Adopt AutoSchemaKG's *openness*, do **not** downgrade to their flat event model |
| `:Source` / `:Node` provenance | kind | **Keep** — unchanged |
| Attributor `field` | domain | **Replace** — → conceptualization (concept layer) |
| Referencers' `tactic` / `REFERENCE_KINDS` | domain | **Replace** — → open relations (`:DEPENDS_ON` / `:USES`) |
| `EntityType` / `FIELDS` / `ACTIONS_ALL` closed vocab | domain | **Open up** — the entity `type` becomes an open property (finder-tagged); the closed field/tactic taxonomies give way to conceptualization + open relations |

What is **not** touched: the pedagogy stages (splitter/instruction), the `:Procedure`/`:Event`
procedural spine, and the `:Source`/`:Node` provenance layer.

---

## Entity layer: the block finder (decided)

The three per-type finders (problem / definition / theorem) — with their attributors and referencers —
collapse into **three general stages**: **detect → attribute → procedure-find**. Everything the finder
produces is an **`:Entity`**; a procedure is *extracted from within* an entity, never a
separately-detected block. **There is no separate task/statement classify stage** — see below.

**1. Block finder (detect).** Literally the current finder's cursor-walk with only the "what is a
block" clause of its Signature generalized (from "a posed math task" to "any labeled pedagogical
block"), emitting each block as a **span only** — **no type**. (The original per-type finders emit
only spans too: `problem.py`'s output is `list[ProblemSpan]` of `{start, end}`; the entity's type was
hardcoded by *which* finder ran — `Entity(type=PROBLEM)` — never classified. Keep the finder pure
detection; the **type is extracted downstream by the attributor**, not here.) Span/boundary is the
finder's *existing* machinery (growing-window structural banking; "start at own label, stop at next
label") — kept **verbatim**.
- *Validated by probe:* one prompt found every block kind across math/physics/biology and excluded
  prose/headers/remarks; when *asked*, it also typed them correctly (incl. a non-math `law` and an
  induced `key concept`) — so open typing is reliable (it now lives in the attributor, stage 2). A
  second boundary probe confirmed the finder's boundary rules cleanly separate adjacent blocks a naive
  one-shot prompt overlapped (theorem+proof to its own span; 8/9 spans correct, the lone miss a
  start-of-stream over-reach the full machinery covers).

**2. Universal attributor.** One pass (not per-type) reads each entity's content and fills its
attributes: **label, number, title, content, and `type`** — what kind of block it is (definition /
theorem / law / example / mechanism / …), as an **open, induced property** (not a closed enum, and
not a Neo4j label — open types would explode the label set; `kind = label, type = property`). The
attributor already reads the content to fill label/number/title, so typing is one more field — **not
a separate classify stage and not the finder's job**.

**3. Procedure finder.** A procedure is **extracted from within an entity** — the entity's shown steps
reify into `:Procedure`/`:Event` via `:HAS_PROCEDURE` (the bi-modal spine holds: entity = declarative,
extracted procedure = procedural; the split is *produced by extraction*, not decided by a classifier —
matching the old validated pipeline). The procedure finder runs over **all** entities and makes one
direct semantic call each — *is there something to work out (prove / solve / derive), shown or
absent?* — rather than deriving it from the type. Routing:
- **shown** (theorem's proof, example's solution) → **extract**: split statement from steps, reify.
- **absent + posed problem to solve** (exercise, no worked solution) → **create**: the procedure
  creator generates the steps (AutoMathKG-style completion, **task-first**).
- **absent + asserted claim** (theorem, no shown proof) → **defer** (generating proofs is deferred).
- **nothing to work out** (a definition, a bare fact) → **skip**.

This is the generalized `solution_start`/proof split the old attributor did per-type — one pass over
any entity, proven machinery.

**No separate classify stage.** The two things a "classifier" might do are handled without one: (a)
the **`type`** (what kind of block) is just one of the attributes the universal attributor extracts —
it reads the content anyway; (b) the **task-vs-statement** routing lives *inside* the procedure finder
(posed-problem-to-solve vs asserted-claim, in its create/defer branch). So the finder stays pure
detection and there is no standalone classify step — but the open `type` metadata is still produced,
by the attributor.

**The same "kind general, type a property" pattern at every level:**

| Level | Kind (general) | Type (property) | Made of |
|---|---|---|---|
| block | `:Entity` | definition / theorem / example / law / … | — |
| worked part | `:Procedure` | proof / solution / derivation / … | steps |
| unit | `:Event` (step) | — | — |

`solution` and `proof` are just `:Procedure` **types**, not separate fields — the entity's old
`solutions` / `proofs` fields unify into one `procedures` list.

**This replaces the per-type finders/attributors/referencers.** The file-level removal + build steps
are in `ENTITY-LAYER-REBUILD.md`, with this section as the build design.

## Concept layer redesign (the validated core)

Replace the closed `FIELDS` taxonomy with AutoSchemaKG-style **conceptualization**, plus a
prerequisite edge that fits this graph better than a taxonomy.

### 1. Flat multi-tag conceptualization (`:INSTANCE_OF`)

Adapt AutoSchemaKG's schema-induction prompts (their Figs 5/6/7, in `docs/autoschemakg/markdown.md`
§B.2) — one each for **entity**, **event**, **relation** — **relaxing the "1–2 word" constraint**
the 8B needed (a frontier model gives precise, multi-word concepts). For each element, generate
several concept phrases spanning **specific → general**:

- `φ`: node (entity/event) → concept — `(:Entity|:Event)-[:INSTANCE_OF]->(:Concept)`
- `ψ`: relation → concept-type

Concepts are **born canonical**. Multi-granularity comes from **attaching several flat tags of
differing generality to each node** (coarse tag shared by many, fine tag by few) — *not* from a
concept tree. Entities get **graph-context enhancement** (AutoSchemaKG §B.2.2): feed a node's
neighbours (its refs, members, field) into its conceptualization prompt so abstraction is grounded
in the node's role. This subsumes the current `field` attribute — `concepts.py` already mints a
`:Concept` from a string, so it generalizes with no schema change.

### 2. Drop `:BROADER` / MSC; add `:DEPENDS_ON`

`:BROADER` (taxonomic "is-a-kind-of") was only ever MSC-derived and is **not** an AutoSchemaKG
edge. Cut it. In its place, a **prerequisite** edge, which is more descriptive for a math/science
KG whose point is dependency reasoning:

- `(:Concept)-[:DEPENDS_ON]->(:Concept)` — "you need Y to define/prove/understand X."

`:DEPENDS_ON` beats `:BROADER` here because:
1. It answers the actual goal — curriculum / prerequisite sequencing — that a taxonomy only
   approximates.
2. It is **groundable**: it is the *concept-level rollup* of the entity-level `:REFERENCES` /
   `:USES` graph we already extract (many eigenvalue-entities cite vector-space-entities ⇒ concept
   `eigenvalue` depends on `vector space`). So the graph converges on **one relationship —
   dependency — at two granularities** (mentions and concepts), mirroring the existing
   `:REFERENCES` (entity) vs `:USES` (step) pattern.
3. It fits AutoSchemaKG's open-relation model natively; `:BROADER` was an imported taxonomy.
4. Flat multi-tag concepts already cover the *categorization* `:BROADER` would have served.

Build `:DEPENDS_ON` from **pairwise / reference-grounded** judgments (reliable — see Evidence),
**not** batch free-listing (noisier). Define it strictly as *definitional prerequisite* (not
"related to"). Guard against cycles (co-defined pairs like eigenvalue↔eigenvector can loop; a
prerequisite graph should be a DAG).

### 3. Concept convergence is a separate layer (fusion)

Conceptualization *creates* concept phrases; it does not merge them ("linear algebra" vs "Linear
Algebra" vs "vector spaces"). Merging them into canonical `:Concept` hubs is the **embedding /
fusion** layer (below), reusing the `:Entity:Canonical` name-normalization + embedding mechanism.

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

## The engine / profile / genre architecture

- **Genre layer** (universal, keep): pedagogical extractors — splitter, instruction stages, and the
  **block finder** itself (detecting labeled blocks is genre — every textbook has them). Reliable
  structure with no per-domain induction.
- **Domain vocabulary** (open, per-run): the entity `type` (definition/theorem/law/…), the relation
  labels, and the concept set are **induced** by the frontier model (validated for concepts; see
  Evidence), not drawn from a closed math taxonomy. A hand-authored **profile** remains available for
  a domain that measurement shows needs one; math is the first candidate profile.
- **Engine** (semantic layer): open relation extraction (AutoSchemaKG §3.1 / B.1 — **LLM-named open
  relations**) + conceptualization, run over the block finder's entities. This is the fine-grained
  dependency/concept graph *inside and between* the document-level blocks.
- **Rule (unchanged):** a domain *with* a profile is **never** downgraded to induction. Generality
  lets a new domain in cheaply; it must not degrade a domain that has a profile.

"Find dependencies on its own" = the engine's open relation extraction (no fixed tactic list),
rolled up to concept-level `:DEPENDS_ON`.

---

## Concept convergence: the embedding / fusion decision

The dedup/canonicalization layer (AutoMathKG's update loop — the piece AutoSchemaKG lacks) merges
concept phrases and entity mentions into canonical hubs. Design:

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
- **Note:** fusion makes canonical identity **discovered** (embedding + judge), not deterministic —
  a real change to the persistence contract for that one layer (canonicals become stateful; each
  run is an "Input KG" fused into the "Existing KG"). Non-destructive: mentions preserved via
  `:REALIZES`, so a bad merge is recoverable.

---

## Build sequence (strangler-fig: build → validate → remove)

Each step is usable alone; nothing validated is deleted before its replacement is proven.

1. **Concept layer (start here — validated, additive).** Conceptualization stage (entity + event +
   relation, context-enhanced) → `:Concept` + `:INSTANCE_OF`; concept-level `:DEPENDS_ON` grounded
   in the existing `:REFERENCES` rollup, pairwise-induced, cycle-guarded. Deletes nothing; gives
   math richer concepts immediately and is the exact stage physics/biology reuse.
2. **Open relations.** Referencers → open `:DEPENDS_ON` / `:USES`, retiring the fixed `tactic` /
   `REFERENCE_KINDS` vocab.
3. **Entity `type`: label → property.** So physics/biology types induce. Forward-compatible —
   entities already *write* the `type` property (`graph/entities.py`), so reads work today.
4. **Block finder + universal attributor + procedure finder** (the "Entity layer" design), built
   **alongside** the three per-type finders and **measured against them on real math books**
   (extraction quality is the one thing the probes did *not* validate). See
   `ENTITY-LAYER-REBUILD.md` for the file-level removal + build steps.
5. **Remove** the three per-type finders/attributors/referencers + closed vocab **only once** step 4
   reaches quality parity.
6. **Embedding / fusion** (concept + mention convergence) — independent track; start with the
   model-agnostic candidate-scoring bake-off.

---

## Open questions & risks

- **General extraction quality is unvalidated.** Probes covered concepts, not open entity/relation
  extraction over raw markdown. Math textbook structure (numbered theorems, proofs, exercises) is
  what the profiled finders exploit; AutoSchemaKG's web-general extraction may under-deliver. Gate
  the rip-out (step 5) on measured parity.
- **`:DEPENDS_ON` cycles** from co-defined concepts — need a DAG guard or a co-definition merge.
- **Fusion false-merge** on polysemy — conservative judge + known-collision benchmark.
- **Keep a math profile, or trust induction?** Decide per-layer by measurement. Concepts: induction
  looks sufficient. Declarative extraction: TBD.
- **`:Procedure`/`:Event` must not be flattened** to AutoSchemaKG's event model — keep the spine.

## What we keep from AutoMathKG (not discarded by going general)

- The **procedural richness** (`:Procedure` container + event chain) — richer than AutoSchemaKG.
- The **update loop** (embedding fusion + Math-LLM completion) — deferred future work, wraps the
  whole engine; the concept-convergence half is the embedding/fusion track above.
