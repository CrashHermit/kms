# Schema — the knowledge graph

The single source of truth for **what the graph is**: its node kinds, their properties, the
edges between them, and the rules that govern what gets stored. Supersedes `UNIFIED-KG.md`
(deleted), whose mention/canonical split, `:Event`-as-step, `:BROADER`/MSC taxonomy, and
per-kind labels are all gone.

The build order that gets us here is `REBUILD.md`. The deferred concept work is
`CONCEPT-LAYER.md`.

---

## Principles

These are the rules the schema is derived from. Every one of them was violated somewhere in
the previous design, and each violation cost a dead property or a dead node kind.

**1. Persisted vs. transient.** The in-memory models are the **pipeline's working state**.
The graph is the **deliverable**. A field is persisted only if something reads it *from the
graph*. Fields that exist to pass information between stages are **transient** and are
marked as such. This rule alone would have caught `ASTNode.role`, `BodySegment.action`, and
`Entity.bodylist` — three properties that were written to Neo4j and never once read back.

**2. No write-only properties.** If nothing reads it, it does not exist. Applies to the
model layer too: a field with no consumer is deleted, not kept "for later."

**3. Strict provenance.** Source text is never destructively rewritten. Nodes hold the
document verbatim; an `:Act`'s text is a verbatim slice of its procedure's content, and the
slices **partition** it — concatenating them in order reproduces the source exactly, with
nothing added or removed.

**4. Structural over semantic.** Where a boundary can be decided by structure, it is never
decided by asking the model to introspect. The finder's window advances on "a node was seen
to follow this span," not on a self-report. The presence or absence of a procedure is
*observed*, not reasoned about.

**5. Kind is a label, type is a property.** A node kind is a Neo4j label. A subtype is a
property — and only exists if it is **non-constant** and **not derivable** from a neighbour.
Open type sets never become labels, because the label set would explode.

---

## Node kinds

Six labels: two provenance, four semantic.

| Kind | Properties | Notes |
|---|---|---|
| `:Source` | `uuid`, book metadata (`title`, `author`, …) | One per book. Roots everything. |
| `:Node` | `uuid`, `source`, `type`, `content`, `index`, `segment_index` | The verbatim document stream. |
| `:Entity` | `uuid`, `source`, `type`, `label`, `number`, `title`, `contents`, `instruction` | A pedagogical block. |
| `:Procedure` | `uuid`, `source`, `contents`, `index` | A worked derivation. Bare — no subtype label. |
| `:Act` | `uuid`, `source`, `text`, `index` | One step. Verbatim slice. |
| `:Concept` | `uuid`, `name` | Global hub. **Currently dark** — see below. |

### `:Entity` — the pedagogical block

A macro-level document region: a theorem statement, a definition, a worked example, an
exercise. **Macro only.** It is *not* the fine-grained entity of AutoSchemaKG (a noun phrase
inside a paragraph) — that kind does not exist here, and the `Act → Entity` "modified
variable" edge is dropped with it.

`type` is an **open, induced** string filled by the statement extractor: `definition`,
`theorem`, `example`, `law`, `mechanism`, whatever the source presents. It is a property on a
bare `:Entity`, never a Neo4j label. It survives (where `Act.action` did not) because it
records the block's *genre*, which concept tags cannot recover: a definition of "normal
subgroup" and an exercise about "normal subgroup" carry identical concepts and are entirely
different objects. `give me the exercises` runs on this axis.

`instruction` is the shared directive of a grouped exercise set ("Graph the following
relations"), copied onto each governed block by the instruction distributor. It is
deliberate denormalization: an atomic exercise retrieved alone is meaningless without it.

### `:Procedure` — the worked derivation

A proof, a solution, a derivation. Bare label, **no `type` property**: a procedure hanging
off `(:Entity {type:"theorem"})` is a proof and one off `type:"example"` is a solution, so
the subtype is derivable from the parent and violates principle 5. This is the one place we
knowingly discard live information — the procedure extractor reads the "Proof." / "Solution."
marker and we choose not to store what it saw. Cheap to reinstate.

An `:Entity` may own **several** procedures (two proofs of one theorem); `index` orders them.

> *Open, not decided:* `contents` is derivable, since the `:Act` chain partitions it.
> Carried for now; a candidate cut.

### `:Act` — the atomic unit

One step of a procedure. `text` is a verbatim slice; `index` is its position.

Named `:Act` rather than `:Step` or `:Event` because it must eventually host two populations:
**prescribed** actions ("apply the chain rule") and **asserted** occurrences ("Napoleon
invaded Russia"). "Act" is honestly neutral between them; "step" and "event" each read wrong
for one side. Today 100% are prescribed, so the distinguishing `type` property is **not
stored** — it would be constant, and a constant property is dead weight (principle 2). It
returns as one property when a narrative corpus does.

No `action` / tactic role. That was AutoMathKG's closed 9-value taxonomy, it was written and
never read, and the concept layer supersedes it richly — the same Act will carry `chain
rule`, `differentiation`, `calculus` as open, cross-corpus-linkable tags.

### `:Concept` — the global hub (dark)

The only connective tissue between blocks, since block-to-block relations no longer exist.
Global identity: a uuid5 over the normalized name with **no source prefix**, so the same
concept from any book converges on one node.

**Currently dark.** Its only source was `Entity.field` from AutoMathKG's closed 7-value
taxonomy, which is deleted. `concepts.py` retains its identity/dedup half
(`normalize_concept`, `concept_uuid`, `concept_properties`, `concept_batches`) and loses the
field-sourced edge builders. No `:Concept` nodes and no `:INSTANCE_OF` edges are written
until conceptualization lands. Design: `CONCEPT-LAYER.md`.

---

## Edges

```
(:Source)-[:HEAD]->(:Node)-[:NEXT]->(:Node)-[:NEXT]->…      document order
(:Source)-[:HAS_ENTITY]->(:Entity)

(:Entity)-[:DERIVED_FROM]->(:Node)                          provenance
(:Procedure)-[:DERIVED_FROM]->(:Node)                       provenance

(:Entity)-[:HAS_PROCEDURE]->(:Procedure)                    1-to-many
(:Procedure)-[:FIRST]->(:Act)-[:THEN]->(:Act)-[:THEN]->…    ordered chain

(:Entity)-[:INSTANCE_OF {phrase}]->(:Concept)               dark
(:Procedure)-[:INSTANCE_OF {phrase}]->(:Concept)            dark
(:Act)-[:INSTANCE_OF {phrase}]->(:Concept)                  dark
```

`:Procedure` gains `:DERIVED_FROM` — new. The old design routed procedure provenance
transitively because a procedure was carved out of an entity's members. Now the finder emits
procedure spans directly, so a procedure has member node ids of its own.

`:Act` has no direct provenance edge: its text is a sub-node slice that need not align to
node boundaries. Provenance is transitive — `Act → Procedure → Node`.

**`:INSTANCE_OF` carries the induced phrase as an edge property**, and concept mentions are
*not* nodes. A concept mention has no content of its own — it is one string a model emitted
about another node — so giving it nodehood would multiply the graph by
(entities + procedures + acts) × ~3 for nodes whose entire payload is that string. On the
edge, fusion re-points to merged hubs while `phrase` preserves what was actually extracted,
which is the recoverability a mention tier would have provided.

---

## What is gone

**Kinds and labels:** `:Entity:Mention` · `:Entity:Canonical` · `:GeneralEntity` ·
`:Entity:<Type>` per-type labels · `:Procedure:Proof` / `:Procedure:Solution` ·
`:Concept:Field` · `:Event` (renamed `:Act`)

**Edges:** `:REFERENCES` · `:USES` · `:REALIZES` · `:DEPENDS_ON` · `:BROADER`

**Model vocabulary:** `EntityType` · `ProcedureType` · `FIELDS` · `ACTIONS_ALL` ·
`REFERENCE_KINDS` · `Reference` · `Proof` · `Solution` · `BodySegment` ·
`Entity.field` · `Entity.bodylist` · `Entity.proofs` · `Entity.solutions` · `Entity.refs`

**Properties:** `:Node.role` (transient now — still in memory, no longer persisted) ·
`:Act.action` · `:Entity.bodylist` (JSON blob)

The mention/canonical split went with the referencers: canonical hubs existed *only* as
`:REFERENCES` targets. The one capability they uniquely offered — representing a citation to
something absent from the corpus — moves to the concept tier for free, as a `:Concept` that
no `:Entity` defines.

Block-to-block relation extraction is **not** an AutoSchemaKG feature and was not kept.
AutoSchemaKG's $P_{EE}$/$P_{EV}$/$P_{VV}$ operate *within* a text segment, over noun-phrase
entities and single-sentence events; nothing in that paper relates one document block to
another. Block-to-block citation was AutoMathKG's `refs`.

---

## Deferred

Each parked with a reason, none blocking.

| Deferred | Why | Unblocks when |
|---|---|---|
| Conceptualization | Entity layer must land first — Pass 2 samples Pass 1's hierarchy | Entity rebuild ships |
| `:DEPENDS_ON` | No grounded evidence source once reference-rollup is gone | A grounding signal is chosen |
| Pedagogical ordering | Parked | Multi-book corpus makes first-appearance meaningful |
| `:Act {type:"occurrence"}`, micro-entities | No producer, no corpus | A narrative corpus arrives |
| Procedure generation | Generated Acts have no source text, so they break principle 3 | A distinct provenance marking is designed |
| Fusion | Needs a corpus-wide execution mode the pipeline does not have | Concept layer is populated |

Conceptualization itself does **not** need a corpus-global pass: phrase generation is
per-node, and cross-book convergence is automatic through the global `concept_uuid`. Only
fusion needs corpus-wide execution.
