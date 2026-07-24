# Unified KG — the graph substrate

Design for merging **AutoMathKG** (arXiv:2505.13406) and **AutoSchemaKG** (arXiv:2505.23628)
into one knowledge-graph model. This doc records the **vertex substrate** (settled) and the
reasoning behind it; edges and the semantic mechanisms are scoped at the end as open work.

Status: **substrate settled, edges not yet designed.** Nothing here is built yet — the current
pipeline (see `HANDOFF.md`) persists a `:Source`/`:Node` provenance layer, a math-specific
`:Entity` overlay, and `:GeneralEntity` reference hubs. This doc supersedes that graph schema;
the migration notes at the bottom say what folds into what.

---

## Scope — math-first; generalization deferred

Decided after the initial design: **adopt the entity/event/concept substrate now, concretely for
mathematics; defer the general engine.** The two are separable, and only one is a large change:

- **Adopted now — this is just a better *math* graph.** The `:Entity` / `:Event` / `:Concept`
  model, the generic `:Procedure` container, MSC-anchored concepts, and a canonical hub (the
  refactor of today's `:GeneralEntity`). None of this needs the engine — it is the concrete
  AutoMathKG + AutoSchemaKG **content** merge for math, and it is the actual answer to the original
  question: concepts give the multi-hop / curriculum win, events give step-level reasoning. Math
  types and tactics are **hardcoded** (labels, as the current code already does — `:Entity:Theorem`),
  not open properties.
- **Deferred — the "huge change," kept below as the north star, not the next build.** The
  engine/profile split, open `type` / `role` properties *for schema induction*, the explicit
  genre/domain layering, `stance`, any second domain, and the heavyweight update loop (MathVD
  embedding-fusion + Math-LLM completion). Building a plug-in engine with exactly one plug is
  premature abstraction; adopt it only when a second domain is real (rule of three). Because the
  substrate is identical either way, that later step is a **refactor, not a rewrite** — the math
  build extracts cleanly into a profile if and when it's warranted.

Everything below describes the full/general design so the north star is on record. Sub-sections
tagged **_(generalization-era)_** are deferred; the plain substrate is what's in scope now.

---

## Thesis _(generalization-era)_: AutoSchemaKG is the *engine*, AutoMathKG is a *profile*

The two papers are not peers to fuse symmetrically. They contribute different things, and the
clean way to state the merge is one sentence:

> A general **Entity/Event/Concept** engine that **induces** a domain schema by default
> (AutoSchemaKG), but accepts a **hand-authored profile** for domains you care about
> (AutoMathKG/math being the first).

- **AutoSchemaKG** gives the general machinery: the Entity/Event/Concept node kinds, and — the
  part we must not throw away — **schema induction**, the ability to *discover* the types of a new
  domain with no predefined schema. Hardcoding Definition/Theorem/Problem as schema primitives
  would keep only its concept layer and discard its actual thesis.
- **AutoMathKG** is then not "the substrate" — it is the **first hand-authored domain profile**:
  fixed entity/procedure *types* (definition/theorem/problem, proof/solution), a fixed action
  vocabulary (its 9 tactic labels), a concept ontology (MSC), and per-type extractors (today's
  finders/attributors). It is a *specialization* of the general engine, not its core. AutoMathKG's
  genuinely unique piece — the **update loop** (MathVD fusion + Math-LLM completion), which
  AutoSchemaKG lacks entirely — wraps the whole engine.

The engine never mentions math. Quality on math comes from the *profile*, not from induction —
which matters because AutoSchemaKG's own limitations section reports that schema induction
"struggles with extremely technical domains," and math is the canonical one. **The rule: a domain
with a profile is never routed through induction.** Generality lets a *new* domain in cheaply; it
must never downgrade a domain that has a profile.

## Three layers of generality: kind / genre / domain

The design separates three things that are easy to conflate:

- **Kind** (universal, permanent): the node kinds — Entity, Event, Concept — plus containers.
  Domain-free. This is AutoSchemaKG's formalism `G = (V_N ∪ V_E, E, C, φ, ψ)`.
- **Genre: the textbook** (cross-domain, mostly domain-free): worked examples, exercises,
  exposition, statements-of-fact, and the lead-in ("In the following exercises…") that governs a
  run of problems. These are **pedagogical universals** — a physics text, a CS text, and a math
  text share them structurally. Much of what the current pipeline calls "math" is really this
  layer: the **splitter**, **instruction_finder**, and **instruction_distributor** are
  exercise-pedagogy, not math, and generalize to any textbook for free.
- **Domain: math (or physics, …)** (pluggable): only the *vocabulary* — what declarative
  statements are called (definition/theorem vs law/model), the tactic labels, the concept ontology.

Asymmetry worth internalizing: **procedures generalize better than statements.** A procedure is a
procedure — steps are steps, "example vs exercise" is just "shown vs withheld." A *declarative
statement* comes in domain-specific flavors (theorem vs law vs proposition). So the procedure
extractor is the most reusable piece; the declarative extractor is where per-domain profiles earn
their keep.

## The bi-modal spine: declarative = Entity, procedural = Event

The declarative/procedural split is the organizing idea and it is **not** math-specific:

| Domain | Entity (declarative — "what is true / asked") | Event (procedural — "what you do") |
|---|---|---|
| Math | definition, theorem statement, problem statement | proof steps, solution steps |
| Physics | law, model, constant | derivation, experiment procedure |
| CS | data structure, complexity claim | algorithm steps, correctness proof |
| Law | statute, precedent | argument, ruling derivation |
| Cooking | ingredient, equipment | recipe steps |

Every knowledge domain with *statements and methods* has this split; math is an unusually clean
instance. A **theorem** decomposes into a statement-**entity** plus a proof **event**-chain,
linked. A **problem/example** decomposes into a statement-**entity** plus a solution
**event**-chain. A **definition** is a pure entity (no events). This makes the "bi-modal graph"
free: the declarative view is the entity subgraph, the procedural view is the event subgraph, and
they meet only at the ownership edge (a theorem *has* its proof). No dual-role storage, no
query-time projection — bi-modality is just which node-kind you traverse.

Dividing rule that keeps entity/event from blurring:

> **Statement structure = entity attributes. Derivation steps = events.**

A theorem's statement has its own premise/assumption/conclusion structure, but that is the *claim*
— it stays as attributes on the `:Entity`. Only the *proof* (the doing) becomes events. A
definition's internal structure likewise stays as entity attributes.

---

## The vertex substrate (SETTLED)

### The governing principle: labels for closed structural facets, properties for open semantic ones

Neo4j gives two ways to annotate a node — labels and properties — and they are not
interchangeable here:

- **Labels are a closed, structural set.** Fast to `MATCH`, constrainable, schema-ish. Correct for
  facets the *engine* defines, of which there are few. Minting a new label per LLM-induced type
  would cause label explosion and destroy the skeleton — this is the failure mode to avoid.
- **Properties are open.** Any string, no schema change. Correct for anything a domain profile or
  the LLM fills in freely.

So the dividing line — and this is what keeps AutoSchemaKG's dynamic-schema property alive at the
vertex level:

| Facet | Owner | Cardinality | Lives in |
|---|---|---|---|
| **Kind** (Entity/Event/Concept/Procedure/Source/Node) | engine | ~6, fixed | **label** |
| **Role** (mention / canonical) | engine | 2, fixed | **label** |
| **Type** (theorem, proof, conjecture, algorithm, …) | profile *or* induced by LLM | unbounded | **property** |

**Kind and role are labels; type is a property.** Typing-as-labels is exactly what would stop the
LLM from inducing types, so type must be a property. Roles are a closed 2-set, so both get labels
(no implicit "absence of label ⇒ mention" rule).

> **Math-first note.** Open-property typing exists to serve *induction*, so it travels with the
> deferred generalization. Since math's type set is closed and known, the near-term math build
> keeps types as concrete **labels** (`:Entity:Theorem`, as the current code does) — simpler, and it
> buys nothing to make them open until a second domain is on the table. The *kinds* (`:Entity` /
> `:Event` / `:Concept` / `:Procedure`) are labels in both worlds; only the type facet differs.

### Vertex inventory

```
CONTAINER kinds (structural roots; not semantic):
  :Source     — one per book; roots the :Node text chain                     (built)
  :Procedure  — roots an :Event step chain; type ∈ {proof, solution, …}      (NEW, generic)

PROVENANCE kind:
  :Node       — one raw markdown chunk; :HEAD / :NEXT reading-order chain     (built)

SEMANTIC kinds (the AutoSchemaKG trio):
  :Entity     — a declarative, static object (definition / theorem stmt / problem stmt)
                role label:  :Mention  |  :Canonical
                type property: "definition" | "theorem" | "problem" | …
  :Event      — one procedural step (owned by a :Procedure). Never canonicalized.
  :Concept    — an abstract category (field / method / topic). Born canonical.
```

Concrete nodes:

```
(:Entity:Mention   {type:"theorem"})       # Hefferon's Theorem 1.5, this book, this wording
(:Entity:Canonical {type:"definition"})    # the corpus-level "vector space"  (replaces :GeneralEntity)
(:Procedure        {type:"proof"})         # one proof of a theorem (a theorem may have several)
(:Event            {action:"calculation"}) # one proof/solution step
(:Concept          {type:"field"})         # "linear algebra"
```

### Node kinds explained

**`:Entity`** — the declarative half of the bi-modal spine. Definitions, and the *statements* of
theorems and problems/examples. Static, referenceable. Its domain type is an **open property**
(`type`) so a new domain induces `type:"conservation law"` with no schema change; the math profile
constrains the property to its enum and ships per-type finders.

*Task vs fact is a type, not a kind.* An exercise ("show that…") and a definition ("a group is…")
feel different — one requests, one asserts — but both are static declarative objects. Same kind,
different `type`. The *doing* is Events either way.

**`:Event`** — the procedural half. One step of a proof/solution/derivation (the reified
`bodylist` step the current attributors already produce). Carries the step text and an `action`
(the tactic label). Owned by exactly one `:Procedure`; never deduplicated across books (a step is
proof-specific). This is where AutoSchemaKG's event modeling and AutoMathKG's `bodylist` fuse into
one thing.

**`:Concept`** — the abstract category (the vertical `φ` axis). Fields ("group theory"), methods
("proof by induction"), topics. Field- vs method-concept is a `type` property, not a new kind.
Concepts are *born canonical* — a concept is inherently a corpus-independent hub — so they do not
carry the mention/canonical split (they handle their own dedup if needed). Concept induction should
be **anchored to an existing math ontology** (MSC / nLab / ProofWiki categories) rather than pure
open induction, to dodge AutoSchemaKG's technical-domain weakness; open induction supplements what
the ontology misses.

**`:Procedure`** — a *generic container* (not a fourth semantic kind). It is the structural sibling
of `:Source`: `:Source` roots a `:Node` chain, `:Procedure` roots an `:Event` chain. One generic
kind with an open `type` (proof / solution / derivation / algorithm / protocol / recipe) so it fits
any domain — proof and solution were merely the math names for "a named, ordered sequence of steps
toward an end." It earns nodehood over a bare edge because (a) a statement can own **several**
procedures (a theorem with two proofs = two `:Procedure` nodes) and (b) it is the home for
procedure-level metadata — notably the flag that a procedure was **generated by completion** rather
than read from the source.

**`:Source` / `:Node`** — the provenance tier, already built and validated. Every semantic node
points back to its `:Node` chunks via `:DERIVED_FROM`. Not reopened here.

### Roles: mention vs canonical

The `:Entity` mention/canonical split is where both papers' core value lands, expressed as a
composable **role label** rather than a separate kind:

- **`:Entity:Mention`** — book-specific, has `:DERIVED_FROM` provenance, an immutable extraction
  record ("Hefferon's Definition 1.2 of a vector space").
- **`:Entity:Canonical`** — corpus-independent identity, may have no source, mutable
  (merged/synthesized by the fusion loop) ("the object *vector space*"). References and `φ`
  converge here; this is what decouples the reference graph from corpus completeness (a reference
  can point at a canonical that has no mention yet — the "missing knowledge" case both papers cite).

**Why a role *label*, not a `:Canonical` kind.** Canonical is a cross-cutting role: it marks
"corpus-independent identity hub" and must compose with a base kind — `:Entity:Canonical` today,
plausibly `:Procedure:Canonical` later if proofs are ever deduplicated. A label composes with any
base kind; a standalone kind cannot (it would lose whether the hub is an entity or a procedure).
This also **subsumes and improves the built `:GeneralEntity`**: that is currently a *disjoint* kind
that references point at but which *is not* an entity, so mentions and their canonical cannot be
queried together. `:Entity:Canonical` unifies them — the hub is still an `:Entity`, so
`MATCH (:Entity {type:"definition"})` sees mention and canonical alike, while `MATCH (:Canonical)`
isolates the hubs.

*Applies to `:Entity` only.* `:Concept` is born canonical; `:Event` never canonicalizes.

---

## What this subsumes from the current build

| Current (see `HANDOFF.md`) | Becomes |
|---|---|
| `:GeneralEntity` reference hub (disjoint kind) | `:Entity:Canonical` (role label on the entity kind) |
| `bodylist`/`proofs`/`solutions` stored as JSON-string blobs on the entity | reified `:Event` steps under a `:Procedure` |
| implicit "proof/solution" nesting inside the theorem/problem entity | explicit `:Procedure {type:…}` container rooting the event chain |
| `:Entity:Theorem` (type as a secondary *label*) | `:Entity:Mention {type:"theorem"}` (type as an open *property*) |
| `:Source` / `:Node` provenance layer | unchanged |

The extraction front-end (OCR → corrector → extractor → splitter → instruction_finder → finders →
attributors → referencers) is **largely untouched** — it already produces statements, `bodylist`,
proofs/solutions, and tactic-labeled refs, i.e. the raw material for entities *and* events. The
change is mostly at the graph-persistence boundary (reify events, relabel canonical, open the type
property) plus reframing the math-specific finders as the *math profile* rather than the core.

---

## Open work (not yet designed)

**Edges — the next session's focus.** The relationships fall into families, sketched but not
settled:
- **Containment** (structural, easy): `(:Entity)-[:HAS_PROCEDURE]->(:Procedure)-[:FIRST]->(:Event)-[:THEN]->…`
- **Aboutness / the "anchor"** (semantic, hard): a worked example `:DEMONSTRATES` a theorem; an
  exercise `:PRACTICES` a concept. Difficulty tracks where it points — a **proof** anchors to its
  own theorem (trivial, positional); an **example** anchors to a specific statement (medium,
  positional); an **exercise** anchors *up* to a concept (hard, thematic, needs the concept layer
  to exist first → a later linking pass, consistent with how `instruction_distributor` and the
  `referencer`s already draw cross-entity edges).
- **Step-use / references** (medium): `(:Event)-[:USES {tactic}]->(:Entity)`. AutoMathKG attaches
  refs at the *entity* level; the unified model refines them to the *step* level (which step
  invoked which prior result), with the entity-level ref recoverable as a rollup. Open question:
  push the `referencer` stage down to per-step, or keep it entity-level and defer.
- **Conceptualization** (`φ`): `(:node)-[:INSTANCE_OF]->(:Concept)`.
- **Dedup / identity**: `(:Entity:Mention)-[:REALIZES]->(:Entity:Canonical)`.

**Other deferred decisions:**
- **Procedure ownership.** Assume `:Procedure` hangs off its owning `:Entity` via `:HAS_PROCEDURE`;
  confirm whether procedures ever stand alone (a derivation with no single owning statement).
- **Statement decomposition.** Keep theorem statements atomic (default) vs decompose
  premises/conclusions into first-class nodes to enable hypothesis-matching queries ("given these
  hypotheses, which theorems apply?").
- **The horizontal fusion loop** (AutoMathKG): MathVD embeddings → candidate retrieval → LLM
  merge-or-add, producing/updating `:Entity:Canonical`. Each run is an "Input KG" fused into the
  Neo4j "Existing KG".
- **The completion mechanism** (AutoMathKG): Math-LLM synthesizes a missing `:Procedure` (event
  chain) for a statement that has none — reframed as *procedural-layer generation*.
**Deferred to the generalization era (not the next build):**
- **The engine/profile split and profile interface** — what a domain profile supplies
  (entity/procedure `type` vocabularies, per-type extractors, action/tactic vocabulary, concept
  ontology). Math is the first profile; the procedure extractor is shared across profiles.
- **Open `type` / `role` properties and `stance`** — the schema-induction machinery. Math-first
  uses concrete labels instead.
- **The heavyweight update loop** — MathVD embedding-fusion and Math-LLM completion (above). The
  canonical *hub* stays in scope as a cheap name-normalized cluster (as today's `:GeneralEntity`);
  the embedding-based fusion *mechanism* is deferred.

**Accuracy gate (whenever fusion/canonicalization lands).** Hold a math validation set and require
the unified pipeline's extraction precision/recall to *match today's math-only numbers* before
migrating; benchmark fusion with a false-merge rate on known-collision terms ("normal", "regular",
"kernel", …). Bias the fusion judge toward *not* merging — a duplicate hub is cheap, a wrong merge
corrupts every reference routing through it. Canonicals are non-destructive (mentions preserved via
`:REALIZES`), so a bad merge is recoverable.
