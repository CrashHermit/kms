# ADR 0001 — Facts as the only semantic primitive

- **Status:** Proposed
- **Date:** 2026-08-03
- **Supersedes:** HANDOFF "Key design decisions" #3 (equations/variables as sibling
  tiers on `:Node`)
- **Affects:** `ingestion.variable_extractor`, `ingestion.atomic_fact_extractor`,
  `ingestion.entity_extractor`, `graph.equations`, `graph.variables`, `pipeline`

## Context

The pipeline currently runs **two extraction philosophies over the same text**:

- **Node-anchored enrichment** — `equation_variable` walks every content node, mints
  `:Equation` and `:Variable` vertices, and anchors them to the node.
- **Fact-anchored semantics** — `atomic_facts` → `entity_extraction` → (planned)
  concepts → relations.

Both read the same source text and both mint vertices. Anything that is simultaneously
notation *and* something a fact is about therefore gets extracted twice, under two
identity schemes that can never merge.

This contradicts the project's own stated first principle. HANDOFF decision #1 says
"Facts are the primitive. Everything — claims, equations, variable bindings — is
decomposable into atomic facts." Decision #3 then carves equations and variables out as
sibling tiers. The overlap measured below is precisely the seam between those two
decisions.

### Measurements

All figures from live runs against real gold OCR pages
(`data/gold/extractor/pages`, `data/gold/corrector/real`) using the configured text LM.
Both upstream passes are nondeterministic, so A/B arms were run N=3 and are reported
with ranges.

**Cost.** The equation/variable pass is the only O(nodes) semantic pass; every other one
is windowed.

| Document | Nodes | eq/var calls | Fact-pass calls | Ratio |
| --- | --- | --- | --- | --- |
| Lebl p260 | 15 | 25 (15 router + 4 eq + 6 var) | 2 | 12× |
| Grinstead & Snell p00–p04 | 57 | 92 (56 router + 24 eq + 12 var) | 7 | 13× |

**The artifact context does not help the fact pass, and costs recall.** Arms: A = no
`equations`/`variables` input, B = both supplied. Same nodes, same windows.

| Metric | A (no ctx) | B (with ctx) | A's range |
| --- | --- | --- | --- |
| facts (Grinstead, 7 windows) | **89.0** | **78.7** | 87–91 (B: 71–85) |
| % facts with an opaque symbol | 36.0 | 35.7 | 30.8–41.6 |
| equation-name adoptions | 0.0 | 0.0 | 0–0 |
| cross-window symbol uses | 25.7 | 23.7 | 24–27 |
| cross-window uses carrying the meaning | 3.0 | 2.7 | 3–3 |

All three A runs exceeded all three B runs on fact count — perfect separation, p = 1/20
= 0.05 by permutation test. The likely cause is the instruction the context arrives
with ("Read-only context — do not restate them as facts") applied to a list of 34
equations and 25 variables.

The cross-window row is the decisive one. It counts facts that use a symbol bound in an
*earlier* window, where the fact pass cannot see the binding and only the artifact
channel could carry the meaning forward. Supplying the exact symbol→meaning table did
not increase resolution (2.7 vs 3.0). That was the one mechanism that could have
justified the ordering.

**The equation `name` field's premise does not hold.** Of 34 equations extracted from
the Grinstead pages, 7 carried a name, and those were `'union definition'`,
`'intersection definition'`, `'set difference definition'`, `'complement of a set'` —
descriptive restatements, several duplicating each other, none resolvable as a
cross-document identity. The fact pass adopted zero of them, in either arm.

**The double extraction is at the semantic layer, not the symbol layer.** After the
entity pass was taught to skip bare symbols, the overlap persisted:

- `'discrete metric'` and `'diameter of a set'` were minted both as `:Equation.name` and
  as entity names.
- Every `:Variable.meaning` was a paraphrase of entities the entity pass emitted
  independently (`'the subspace metric'` ↔ `subspace metric`).
- The variable extractor emitted `'sample space'` and `'discrete'` as bindings — those
  are concepts, not stand-ins.

An `:Equation.name` *is* a concept name; a `:Variable.meaning` *is* a concept
description. Only `:Variable.symbol` is genuinely unique to the pass.

**Two defects surfaced by the same runs.**

- The LLM router is nondeterministic: on identical input `has_variable` fired on 6 nodes
  in one run and 3 in the next, silently dropping five bindings. It also saves only ~19%
  of calls (56 router calls to gate 36 extractions) while adding a serial round-trip per
  node.
- `variable_uuid(source, node_id, symbol, value)` collides. `$X$` bound three ways in one
  node ("a set", "the set containing 0", "the set containing a and b") produces one
  vertex; last write wins. Reproduced on both test documents.

**What is working.** The entity pass's per-fact provenance held at **0 out-of-range
`fact_index` across every run**, including multi-batch runs where indices must be global
rather than batch-local. Its windowing is the template the other passes should follow.

## Decision

Adopt **facts as the only semantic primitive**. Node-anchored extraction survives only
for things that genuinely are not facts: document-local scaffolding. Tier membership is
decided by **identity scope**, not by extraction mechanism.

| | Scope | Home |
| --- | --- | --- |
| An equation | means the same in every book | a `:Concept` |
| A symbol binding | true only within its passage | a local `:Symbol` |

### 1. Reorder: symbols run after concepts, not before facts

```
current:  hub_builder → equation_variable → atomic_facts → entity_extraction → persist

adopted:  hub_builder → atomic_facts → fact_embedding → entity_mentions
                      → concepts → relations → symbols → persist
```

Running symbol extraction first forces it to emit prose meanings that the concept layer
independently rediscovers — that is the double extraction. Running it last lets it
**link** rather than describe: the concepts already exist, so a binding becomes an edge
to one instead of a competing vertex.

This also demotes symbols from a blocking prerequisite to a late enrichment. A skipped
or failed symbol pass leaves the graph complete.

### 2. Remove

- **The equation extractor and the `:Equation` tier.** Equations become
  `:Concept {type: 'equation', latex}` grounded by the facts that state them, inheriting
  the cross-book dedup the concept pass already needs. Deletes an identity space
  (`source, node_id, index`) that can never merge across books.
- **The `equations` and `variables` input fields on the fact signature**, and the "do not
  restate them as facts" instruction with them.
- **The LLM router.** If a skip filter is wanted, a regex over math delimiters is free
  and deterministic.

### 3. Change

- **Window the symbol pass** as the fact and entity passes are windowed. Node attribution
  travels as a `node_id` on each binding, exactly as `fact_index` does today.
- **Put meaning (or a scope discriminator) in the variable uuid.** Live bug, independent
  of this ADR.
- **Treat a worked derivation as one fact**, with its steps as provenance rather than as
  sibling facts. The entity pass cannot fix this from its own prompt — it sees a lone
  step-fact and correctly reports that the fact is about that expression.

### 4. Add

- **A binding model** that links rather than describes:

  ```
  (:Symbol {text: '$X$'})-[:BINDS {scope}]->(:Concept {name: 'random variable'})
  ```

- **A durability gate** on whatever pass owns symbols, to keep concepts
  (`'sample space'`) out of the local tier. The model does not draw that line unprompted.

### 5. Unchanged

The node/provenance spine, the `:Statement`/`:Procedure` hub overlay, facts as the
primitive, and the entity mention pass.

## Consequences

**Gained.** One extraction philosophy instead of two. Equations merge across books for
free. The symbol pass drops from ~13× the fact pass to roughly parity. The fact pass
recovers ~12% of its output. Two live bugs are closed.

**Lost.** `:Equation` and its writer are deleted; anything reading that label must move
to the concept layer. Equation LaTeX becomes a concept property rather than its own
vertex.

**Timing constraint.** Item 2 must land **before the concept pass persists anything**.
That is the moment the two identity schemes fork in the database, after which this
becomes a migration rather than a design change. Nothing is corrupt today: the
`entity_mentions` channel is state-only, and the persister does not write it.

## Sequencing

1. Drop the artifact fields from the fact signature — smallest change, recovers ~12%
   recall, independently measurable.
2. Fix the variable uuid collision — live bug, independent of the rest.
3. Window the symbol pass and delete the router — the cost lever.
4. Fold equations into facts and delete `:Equation` — **before the concept pass writes**.
5. The `:Symbol`→`:Concept` binding model — after concepts exist.

## Open questions

- Items 4 and 5 are designed against the concept pass as described in HANDOFF, not
  against code. If clustering changes how concept identity works, the
  equation-as-concept shape may need adjusting.
- The derivation-granularity change (§3) is a hypothesis from two observed pages and has
  not been A/B tested. It should be, before it is trusted.
- Whether the symbol pass is worth running at all is now an open question rather than an
  assumption. Its only unique output is `:Variable.symbol`; the value of a queryable
  decoder ring should be demonstrated against a real retrieval task before the windowed
  rewrite is built.
