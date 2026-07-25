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
| **`problem` finder** (+ most of the problem attributor) | genre | **Keep** — "a posed task" is a pedagogical universal; already neutralized to any textbook. Gives a reliable cross-domain `problem` type *without* induction |
| `:Procedure` / `:Event` spine (proofs/solutions → step chain) | kind | **Keep — richer than AutoSchemaKG.** Their events are flat triples; ours is a real derivation spine. Adopt AutoSchemaKG's *openness*, do **not** downgrade to their flat event model |
| `:Source` / `:Node` provenance | kind | **Keep** — unchanged |
| **`definition` & `theorem` finders** | domain | **Generalize / replace** — math's declarative types. Physics = law/model, etc. Replace with open extraction or a per-domain profile |
| Problem attributor `field` | domain | **Replace** — → conceptualization (concept layer) |
| Referencers' `tactic` / `REFERENCE_KINDS` | domain | **Replace** — → open relations (`:DEPENDS_ON` / `:USES`) |
| `EntityType` / `FIELDS` / `ACTIONS_ALL` closed vocab | domain | **Open up** — induced or profile-supplied |

The rip-out target is therefore **narrow**: the def/theorem declarative extractors + the domain
attribution (`field`, `tactic`) + the closed vocab. The problem chain, the pedagogy stages, the
procedural spine, and the provenance layer are **not** on the chopping block.

---

## Entity layer: one type-agnostic block finder (decided)

The per-type finders collapse into **one type-agnostic "block finder"** — literally the current
finder's cursor-walk with only the "what is a block" clause of its Signature generalized (from "a
posed math task" to "any labeled pedagogical block"), plus a rough `type` + `mode` (asserts/poses) on
the output. Detection and typing unify: a physics `law` or an induced `key concept` is found the same
way as a `definition`.

- **One finder, not per-type.** Detecting a block and typing it are different jobs; only typing
  differs across definition/theorem/example/exercise/law. **Validated by probe:** one prompt found +
  typed every block kind across math/physics/biology (incl. a non-math `law` and an induced
  `key concept`) and excluded prose/headers/remarks. Span/boundary is the finder's *existing*
  machinery (growing-window structural banking; "start at own label, stop at next label") — keep it
  verbatim. A second boundary probe confirmed those rules cleanly separate adjacent blocks that a
  naive one-shot prompt overlapped (theorem+proof resolved to its own span; 8/9 spans correct, the
  lone miss being start-of-stream over-reach the full machinery covers).
- **Separate classify step — kind first, then type.** Its *primary* job is the **bi-modal kind
  split**: is this block **declarative (→ `:Entity`)** or **procedural (→ `:Procedure`)**? That is the
  "statement vs procedure" distinction, and it is fundamental — it decides which node kind the block
  becomes and how it is persisted, so it must happen *before* attribution. Then the fine `type` within
  the kind: Entity → definition/theorem/law/example/… (asserts/poses is a grouping); Procedure →
  proof/solution/derivation/…. Type is a **property**, not a per-type finder (`kind` structural,
  `type` open) — applied at the genre layer. This lives in the classifier, **not** the attributor
  (routing precedes attribution; the attributor fills `label/number/title/content` once the kind is
  known).
- **The general shape: detect → classify → resolve.** Three general verbs:
  1. **Detect** — find any labeled **block** (the finder), uniformly, including "Proof."/"Solution."
     units (drop the probe's "absorb the proof into the parent span" rule).
  2. **Classify** — *kind* (declarative → `:Entity` vs procedural → `:Procedure`), then within
     declarative the *mode* (asserts = statement vs poses = task) and the fine *type*.
  3. **Resolve** procedures — attach a detected procedure to its declarative neighbour (link,
     `instruction_distributor` shape), and for a block that *needs* worked steps but has none,
     **generate** one. `resolve = extract-if-shown OR generate-if-absent`; the generate half is
     **completion** (AutoMathKG's Math-LLM completion), generalized to any block.
- **"Task or statement" is not the branch point for procedures.** Both are **declarative blocks**;
  the task/statement split is just the asserts/poses *mode* (a property). Procedure handling keys on
  "does this block have/need worked steps?", not on task-vs-statement — a theorem has a proof, an
  example has a solution, both are declarative blocks with a procedure; a definition has none; an
  exercise has one to be generated.
- **One universal attributor** (not per-type) runs after the finder and fills the genre-universal
  fields only: **label, number, title, content**. Every labeled block has these regardless of
  subject, so it is one pass over the unified block stream — no forking.
- **No procedural split yet.** The block's worked-out part (proof/solution/…) is deferred. When it
  returns, it is **one general pass**, not per-type — see below.
- **Generalize "solution"/"proof" → a `procedure` made of `steps`.** "Solution" (task) and "proof"
  (theorem) are too narrow: the general notion is a **procedure** — the worked-out sequence attached
  to a block — whose units are **steps**. This already exists in the graph as `:Procedure`
  (open `type`: proof / solution / derivation / protocol / …) rooting an `:Event` step chain. So it
  is the **same "kind general, type a property" pattern as the block, one level down**: `procedure`
  is the kind, proof/solution/derivation are its types. Concretely, the entity's two narrow fields
  (`solutions`, `proofs`) unify into one general `procedures` list (a block may have several), each
  carrying a `type` + its `steps` — snapping straight onto the existing `:Procedure`/`:Event` layer.
  When this pass returns it is one general "find this block's procedure(s) and steps", any block type.

**This supersedes** the "rename `problem`→`task` + keep the task attributor" shape in
`REMOVAL-def-thm-rename-task.md`: the finder becomes `block` (not `task`), and the attributors are
removed rather than kept. Reconcile that spec before implementing.

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

- **Genre layer** (universal, keep): pedagogical extractors — splitter, instruction stages,
  **problem finder**. Give reliable pedagogical structure (exercises, worked examples, lead-ins)
  with no induction.
- **Domain profiles** (optional, per-domain): the existing math profile (def/thm finders, the
  math relation set) is the **first profile**. Physics/biology may start fully induced and grow a
  light profile only where measurement shows a gap.
- **Engine** (default/general path): open triple extraction (AutoSchemaKG §3.1 / B.1 — entity–
  entity, entity–event, event–event, with **LLM-named open relations**) + conceptualization.
  Unprofiled domains route here.
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
4. **General declarative extractor** (open triple extraction), built **alongside** the def/theorem
   finders, and **measured against them on real math books** (extraction quality is the one thing
   the probes did *not* validate).
5. **Remove** the def/theorem finders + closed vocab **only once** step 4 reaches quality parity.
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
