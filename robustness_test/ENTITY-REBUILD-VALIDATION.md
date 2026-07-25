# Entity-layer rebuild — first live validation

**Date:** 2026-07-25 · **Scope:** the stages changed by #21 (`group_finder →
statement_extractor → procedure_extractor`) and the graph tier they feed.
**Status of the thing under test before this run:** unit-green, compiling, and *never run
against a real PDF* — HANDOFF next step 1.

This is the companion to [`REPORT.md`](REPORT.md), whose numbers describe the **retired**
three-chain entity layer. Where a count is compared below, `REPORT.md` is the baseline.

---

## TL;DR

The rebuilt chain **runs end to end on real pages and its structural contracts hold
exactly**: 5/5 books completed, and across them **0** dangling member refs, **0** span
overlaps, **0** duplicate-member entities, **0** mis-ordered attachments, **8/8** procedures
decomposed into an exact verbatim partition, **0** stepless procedures.

Two behavioural gaps are real, reproducible, and worth fixing before the concept layer:

1. **Unmarked derivations are invisible to the finder.** A worked example whose solution
   runs straight on from the statement with no `Solution.` marker yields **zero** procedure
   spans. On Lebl this is deterministic across 3 runs, and it means the procedural spine —
   the headline deliverable of the rebuild — is **empty for that book**.
2. **`type` induction keys off a block's embedded content, not its nature.** On Hammack,
   **14 of 16** exercises are typed `theorem`/`example`/`quote`. The evidence needed to type
   them correctly exists in the run but arrives one stage too late.

Neither is a regression — both are new-surface behaviour — but #1 undercuts the change the
rebuild was principally for.

---

## Method

Five of the six `robustness_test/books/` slices, chosen to hit the changed stages:
theorem+proof exposition (Stein), worked examples with solutions (Lebl, Morris, Levin), and
a lead-in-heavy exercise set (Hammack). Grinstead & Snell was skipped — its documented gap
is a front-end figure-extraction issue, upstream of anything #21 touched.

Runs were **DB-less** (see [Environment](#environment-neo4j-is-not-reachable)), driven by a
harness that mirrors `pipeline.run()` exactly — same `ocr.extract` + `build_graph().ainvoke`
— and additionally keeps the final state so the overlay can be dumped and checked. The
retired `entities.json` / `nodes.json` artifacts are reproduced *for validation only*; the
pipeline itself still writes just `document.md`.

Because the graph tier could not be exercised live, `graph/{entities,procedures,writer}.py`
were driven **offline against the real extracted output** — they are deterministic and
driver-free by design, so the exact payloads that would be MERGEd can be built and checked
without a database.

---

## Results

| Book | Nodes (base) | Entities | Procs | Steps | Baseline entities | Induced types |
|---|--:|--:|--:|--:|---|---|
| Stein, *Number Theory* | 36 (47) | 8 | **4** | 17 | 1 def, 4 thm, 0 prob | definition 1, example 3, proposition 2, lemma 1, theorem 1 |
| Hammack, *Book of Proof* | 51 (50) | 16 | 0 | 0 | 0 def, 0 thm, 16 prob | theorem 4, example 9, exercise 2, quote 1 |
| Lebl, *diffyqs* | 57 (51) | 22 | **0** | 0 | 0 def, 1 thm, 21 prob | exercise 15, example 3, problem 3, theorem 1 |
| Morris, *Topology* | 38 (37) | 10 | 1 | 16 | 1 def, 0 thm, 8 prob | exercise 6, definition 2, example 2 |
| Levin, *Discrete Math* | 53 (50) | 7 | 3 | 4 | 8 def, 2 thm, 3 prob | example 3, definition 2, lemma 1, proposition 1 |

Totals: 235 nodes → 63 entities, 8 procedures, 37 `:Act` steps, 145 member assignments.
Wall time 66–110 s per 4–5 page slice.

### Structural contracts — all clean

Checked on every book:

| Invariant | Result |
|---|---|
| Member ids resolve to real nodes (provenance) | **0** dangling / 145 |
| One partition — no node in two spans | **0** overlaps |
| Duplicate-member entities (pre-splitter failure mode) | **0** |
| Procedure starts after the entity it hangs off | **0** violations |
| Steps reproduce procedure contents verbatim | **8/8** exact |
| Stepless procedures | **0** |
| Orphan procedures | **0** |
| `role="instruction"` lead-in absorbed into a span | **0** |

The verbatim-partition check is whitespace-normalised concatenation of `steps` against
`contents` — the strict reading of the extractor's contract. All 8 procedures pass exactly.

### What the rebuild claimed, and what actually happened

**Theorem and proof are two adjacent spans — confirmed, 4/4 on Stein.** Every
statement/derivation pair came out as disjoint adjacent spans, correctly attached:

```
[2] proposition 2.5.3  members=[14,15]   -> PROCEDURE members=[16,17,18]  steps=4
[4] proposition 2.5.5  members=[22]      -> PROCEDURE members=[23,24,25]  steps=3
[6] lemma       2.5.7  members=[32]      -> PROCEDURE members=[33]        steps=7
[7] theorem     2.5.8  members=[34]      -> PROCEDURE members=[35]        steps=3
```

This is the structural detection replacing the old attributors' semantic
`proof_start`/`solution_start` call, and on marked prose it works cleanly.

**Open `type` induction — confirmed, and it is a visible gain.** The baseline collapsed
Proposition/Lemma/Theorem into a single `theorem` bucket; the induced vocabulary now
separates them (`proposition` 2, `lemma` 1, `theorem` 1 on Stein) with no profile and no
enum. Stein's three SAGE examples, which the baseline captured as **0 problems**, are now
picked up as `example`.

**Domain-generality — confirmed by direct probe.** No fixture book is non-math, so the
"a physics or biology book types itself with no profile" claim had no coverage. Feeding
hand-built physics and biology node streams through the live chain:

- Physics → `type='law'` for Newton's Second Law (section header correctly excluded as a
  boundary); the marked `Solution.` became a procedure with 3 verbatim steps.
- Biology → `type='definition'` for glycolysis and **`type='mechanism'`** for the
  energy-investment phase.

Both types are outside any closed enum the old schema had. The claim holds.

**Definition counts drop — confirmed, as predicted.** Levin falls from 8 definitions to 2.
HANDOFF called this: 6 of Levin's 8 baseline definitions were *inline bold terms* inside
larger blocks, and under one partition they are absorbed by the containing span. Intended,
but it is the cost side of the one-partition trade and it is a large fraction here.

**Lead-in handling — unchanged and still correct.** Hammack has 2 `instruction` nodes;
**neither** is absorbed into any span, and the distributor stamps **14/16** blocks —
matching the baseline exactly. The splitter/finder/distributor pedagogy path is unaffected
by the rebuild, as expected.

---

## Finding 1 — unmarked derivations produce no procedure spans

**Severity: high.** This is the rebuild's headline capability failing on a book it targets.

Lebl's §1.2 has three worked examples that *do* contain real derivations, but the book never
writes `Solution.` — the working simply continues after the posed problem. The finder
returned **0 procedure spans**, so all of that content was absorbed into the entity's
`contents` and **nothing decomposed into steps**.

The boundary is not a granularity problem — the extractor already put the derivation in its
own nodes:

```
node 20 [header]    **Example 1.2.3:** For some constant $A$, solve:
node 21 [math]      $$y' = y^2, \quad y(0) = A.$$
node 22 [paragraph] We know how to solve this equation. First assume that $A \neq 0$, ...
node 23 [math]      $$y = \frac{1}{1/A - x}.$$
node 24 [paragraph] If $A = 0$, then $y = 0$ is a solution.
```

The finder emitted one span `[20..25]` where it could have emitted entity `[20,21]` +
procedure `[22..25]`. A clean cut existed and was available.

**It is systematic, not variance.** Re-running the group finder in isolation over the saved
node stream gives `entities=22, procedure_spans=0` on **3/3** runs.

**Cause.** The `Signature` tells the model derivations are "usually marked explicitly
(`Proof.`, `Solution.`)", and `docs/SCHEMA.md` principle 4 ("absence is structural") means
there is deliberately no fallback question. Against a book that marks nothing, the detector
has no cue. This is precisely the drift HANDOFF next step 3 predicted — a reliable
structural task fused with a softer unmarked-prose one.

**Consequence in the graph.** Lebl and Hammack both persist **0 `:Procedure` and 0 `:Act`
vertices**. Hammack's zero is *correct* (a pure exercise set has nothing worked out); Lebl's
is a miss. For Lebl the outcome is the same one the rebuild set out to eliminate — a book
whose procedural spine is empty — reached by a different mechanism (no marker → no span,
rather than a type restriction on the step list).

**Suggested direction.** Keep the structural detection as the primary path and add an
explicit unmarked-turn cue to the finder's Signature: within a block that poses a task, a
shift from *posing* to *working* starts a procedure even with no marker word. The
`procedure`-follows-`entity` post-check HANDOFF already names is the cheap detector, and a
worked example with no procedure is a directly countable regression metric.

---

## Finding 2 — `type` keys off embedded content rather than the block's nature

**Severity: medium-high.** `type` is now the *only* genre signal in the graph.

Hammack's slice is a problem set: 16 numbered items under two lead-ins. All 16 are
exercises. They were typed:

| Items | Induced type | What they actually are |
|---|---|---|
| 1–4 | `theorem` ×4 | exercises whose *body* is a maths sentence ("For matrix $A$ to be invertible…"), under the lead-in "Without changing their meanings, convert each of the following sentences…" |
| 5 | `quote` ×1 | same lead-in — a Sartre quotation to be converted |
| 1–9 (2nd group) | `example` ×9 | bare formulas under "Write a truth table for the logical statement…" |
| 10–11 | `exercise` ×2 | correct — these are prose-stated tasks |

**14 of 16 mis-typed.** The extractor reasonably read `$P \vee (Q \Rightarrow R)$` as an
example and "For matrix $A$ to be invertible…" as a theorem: on its own, with no context,
that is what each *looks* like.

**Cause is architectural, not prompt-level.** `statement_extractor.Identify` receives **only
the block's own member nodes**. The two things that disambiguate these — the governing
lead-in text and the bare-number label — are (a) computed by `instruction_distributor`,
which runs **after** it (`pipeline.py:159` vs `:161`), and (b) not surfaced as a signal. The
run *had* the answer: 14 of these blocks carry a correct `instruction` field by the time the
graph is written. The pipeline knows they are exercises and types them anyway.

The ordering is not free to simply swap: `pipeline.py:155-157` records that the distributor
runs last *because* it reads the contents/number the statement extractor fills. So the two
stages have a genuine mutual dependency, and resolving it means either splitting the
distributor (lead-in detection is already done by `instruction_finder` upstream), passing
the block's lead-in context into `Identify`, or re-typing after the distributor.

This is the same failure the baseline logged as gap #3 ("field can key off embedded
content", 3/16, *minor*) — but it has moved from `field`, which was advisory, onto `type`,
which is the property that distinguishes "a definition of X" from "an exercise about X".
At 14/16 it is no longer minor.

---

## Environment: Neo4j is not reachable

The graph tier could **not** be exercised live.

- **Bolt** (`neo4j+s://901d982e.databases.neo4j.io`) — `ServiceUnavailable: Unable to
  retrieve routing information`. Expected; port 7687 is blocked from this sandbox and this
  is the documented reason `NEO4J_TRANSPORT=http` exists.
- **HTTP Query API** — reaches Aura and gets a clean protocol-level response, then fails
  `Neo.ClientError.Security.Unauthorized: Invalid credential`. The credentials carry no
  stray whitespace (verified); the endpoint works and the secret is simply stale or the
  instance was recreated.

So `NEO4J_*` is *set but invalid*, which is worse than unset: `persister.py:33` gates only
on `db.is_configured()` (i.e. `NEO4J_URI` present), so a live run with these values would
crash at `node_persister` rather than skipping. Runs here were made DB-less by unsetting the
vars. **Refreshing the Aura credential is a prerequisite for validating the graph tier**,
and a new session is needed to pick it up (secrets inject at session start).

**What was validated instead.** The mapping layer, driven offline on the real output:

| Check | Result |
|---|---|
| uuid uniqueness within each layer (node/entity/procedure/act) | pass, all books |
| uuid namespaces disjoint across layers | pass, all books |
| Edge endpoints resolve to a written vertex | **0** dangling, all books |
| `:Act` chain shape — n steps ⇒ 1 `:FIRST` + n−1 `:THEN` | pass, all books |
| `:Procedure` carries no `type` property (schema principle 5) | pass |
| `type` written as a property, never a label | pass |

Stein, for example, maps to 1 `:Source` + 36 `:Node` + 8 `:Entity` + 4 `:Procedure` +
17 `:Act`, with 35 `:NEXT`, 12 entity `:DERIVED_FROM`, 4 `:HAS_PROCEDURE`, 8 procedure
`:DERIVED_FROM`, 4 `:FIRST`, 13 `:THEN`. The mapping tier looks correct; only the write path
is unverified.

---

## Smaller notes

- **Doc drift — trace capture is gone.** `CLAUDE.md:41` still lists `core/tracing.py`, and
  `REPORT.md` documents a `KMS_TRACE_DIR` workflow feeding `training/*/dataset.py`. Neither
  exists after #21: there is no `tracing` module and no `KMS_TRACE_DIR` reference anywhere in
  `src/`. **No traces could be captured for this run**, so it produced no training data — a
  loss worth restoring deliberately given the DSPy-optimisation plans for the splitter and
  the finder Signature. Left unfixed here; it is a decision, not a typo.
- **Node counts differ from baseline** (e.g. Stein 36 vs 47). The front-end is untouched by
  #21, so this is OCR/corrector/extractor run-to-run variance, not an entity-layer effect.
  It does mean baseline entity counts are not a like-for-like comparison.
- **Unit suite** is green at the documented figure: 134 passed, 3 skipped.
- **Levin's Handshake Lemma (2.1.8) carries no procedure** while the two examples following
  it do. Consistent with Finding 1 — worth a look when that is addressed.

## Suggested order

1. **Finding 1** (unmarked derivations) — it is the rebuild's core deliverable and it is
   silently empty on a targeted book.
2. **Refresh the Neo4j credential**, then re-run one book with `NEO4J_TRANSPORT=http` to
   validate the write path end to end.
3. **Finding 2** (`type` context) — cheap if the lead-in is threaded into `Identify`.
4. Restore trace capture before doing any Signature tuning, so the tuning has data.
5. Then the concept layer (`docs/CONCEPT-LAYER.md`), per HANDOFF.

## Reproducing

Harness, invariant checker, graph-mapping checker and the domain probe used for this run are
in the session scratchpad, not committed — they reproduce the retired JSON artifacts purely
for inspection. The pipeline itself was run unmodified.
