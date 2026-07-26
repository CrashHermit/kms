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

## Update 2026-07-26 (later) — the entity chain split into one question per stage

The entity layer was refactored from three stages into five, each asking a single question:

```
group_finder        -> spans (boundaries only, untyped)
role_typer     NEW  -> entity | procedure          (closed, binary)
block_typer    NEW  -> type                        (open, induced)
statement_extractor -> label, number, title, contents   (transcription)
procedure_extractor -> steps
```

The motivation was the one HANDOFF next-step 3 named: the finder fused a reliable structural
task (where do units start and stop?) with a softer semantic one (which of these is the
working?), and that fusion is what let a book with no `Solution.` markers lose its whole
procedural spine. `statement_extractor` was fused the same way — a genre *judgement* riding
along in a call that is otherwise *transcription*.

**Where it landed (5 books, versus the fused chain measured the same way):**

| | Fused | Granular |
|---|--:|--:|
| Entities | 63 | 62 |
| Procedures | **15** | **13** |
| Steps | 62 | 58 |
| Verbatim-exact partitions | 15/15 | 13/13 |
| Module probes | 17/17 | **24/24** |
| Structural invariants | 0 | 0 |

**What the split clearly bought.** Each stage is separately testable, and the probe suite grew
from 17 to 24 checks because there are now three isolated decisions to pin instead of one
fused one. `role_typer` is accurate in isolation (5/5, including an unmarked worked solution
and an unlabelled computation session). `block_typer`, given nothing to do but type, now
handles the case the fused pass never could — an exercise whose body is a bare mathematical
assertion comes back `exercise`, not `theorem`.

**What it cost.** Two books still find one fewer procedure than the fused chain (Lebl 2 vs 3,
Levin 3 vs 4). The deficit is in the *finder*, not the typer: stripping the role output also
removed scaffolding that had been helping it decide where to cut. Telling it to cut without
telling it what it is cutting is a harder instruction to follow, and the remaining gap is the
price of that.

**Two regressions found and fixed during the split**, both worth recording because both were
self-inflicted and caught only by the book sweep, never by the probes:

1. *A bare labelled definition got no span at all* — Stein's Definition 2.5.1 vanished from
   the document. The "boundaries only" rewrite lost the emphasis that every labelled unit
   starts a span even when nothing is worked out after it.
2. *Labelled examples containing a worked session were demoted to procedures*, attaching
   Stein's SAGE examples to the preceding proposition as second proofs. Fixing that by
   deleting "a worked session and its output" from the procedure definition then
   over-corrected in the opposite direction — *unlabelled* sessions became entities, which is
   what the 11-procedure intermediate result was. The correct fix is the label rule alone: a
   span that opens with its own label is a block whatever follows; an unlabelled session is
   the working. Both directions are now pinned as probes.

**Recommendation.** Keep the split — the isolated stages are more accurate at what they each
do, and far easier to tune. The open item is the finder's cut rate, which is now a
single-stage prompt problem rather than a whole-chain one.

---

## Update 2026-07-26 — both findings fixed, in the prompts only

Both findings were resolved by editing **Signature text only**: no new functions, no new input
fields, no stage reordering. Three docstrings changed
(`group_finder.Signature`, `statement_extractor.Identify`, `procedure_extractor.Decompose`);
the diff is 3 files, prompt text.

| Measure (5 books) | Before | After |
|---|--:|--:|
| Procedures found | 8 | **15** |
| `:Act` steps | 37 | **62** |
| Verbatim-exact partitions | 8/8 | **15/15** |
| Module probes passing | 14/16 | **17/17** |
| Dangling / overlaps / dupes / bad-attach / orphans / absorbed lead-ins | 0 | **0** |

- **Finding 1 (unmarked derivations).** The finder's Signature said derivations are "usually
  marked explicitly"; it now says a marker is *common but never required*, and to cut at the
  turn from **posing/stating** into **working**. Lebl — the book that defined this finding —
  goes from **0 to 3** procedures, one per worked example. Stein gains 2 (its SAGE sessions,
  which genuinely do work the example out) while keeping all 4 proofs exact. Hammack stays at
  **0**, which remains correct: a pure exercise set has nothing worked out. The fix adds
  procedures where there is working, not everywhere.
- **Finding 2 (`type` keys off embedded content).** `Identify` now says to type the block, not
  its subject matter, with the Hammack failures as worked counter-examples, plus an
  example-vs-exercise test based on *whether the working is shown*. Hammack goes from **2/16**
  correctly typed to **15/16**. Notably this did **not** need the lead-in threaded in — the
  earlier recommendation to add that plumbing was the wrong shape.

**A regression the fix exposed, and its fix.** Once Stein's SAGE examples had procedures for
the first time, one decomposition dropped its trailing sentence ("The output of the roots
command above lists each root…") — 94 of 192 characters, a partition violation. `Decompose`
now states that surrounding and trailing prose is part of the partition and a closing remark
is its own final step. That case is pinned in `module_probes.py`. All 15 procedures are exact.

**What is still imperfect:**

- **1 of 16** Hammack items is still typed `theorem` ("If a function has a constant derivative
  then it is linear, and conversely") — down from 14, but not 0.
- Span boundaries moved slightly on two books: Morris 10 → 9 blocks (one absorbed), Levin
  7 → 8. Morris also *gained* correctness — an exercise previously typed `definition` is now
  `exercise`.
- Morris's exercises number 1, 2, 3, 4, 4, 6. This duplicate **predates** these changes (it is
  in the original run too) and is the known `number` format sensitivity, not a new fault.

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
node stream gives `entities=22, procedure_spans=0` on **3/3** runs with DSPy's caches
explicitly disabled (`dspy.configure_cache(enable_disk_cache=False,
enable_memory_cache=False)`, ~13 s of real calls per trial).

> **Methodology correction.** An earlier version of this section cited 3/3 identical runs
> *without* disabling the cache. Those repeats were served from `~/.dspy_cache` — the whole
> 3-trial probe returned in 4.5 s — so they demonstrated cache determinism, not model
> determinism. The numbers above are the re-verified, genuinely uncached result; the
> conclusion is unchanged, its evidence is not. **Anyone re-running a book on this machine
> should assume the DSPy disk cache is on and will silently serve a prior run's answers.**

**Cause — the finder sees the derivation and declines to mark it.** This is not a perception
failure. The finder's own recovered reasoning for the Lebl window says so outright:

> "Nodes 4-5: The example statement and discussion. **This example doesn't have a separate
> solution marker** — it's just presented and discussed. So the example itself is the entity."
>
> "Nodes 7-9: The equation and solution discussion. **Again no separate 'Solution.' marker —
> the solution is integrated.**"
>
> "Nodes 21-25: The example **with its solution integrated**. End entity at 25."

The model identifies the solution, names it as a solution, and then folds it into the entity
because the `Signature` tells it derivations are "usually marked explicitly (`Proof.`,
`Solution.`)" — and `docs/SCHEMA.md` principle 4 ("absence is structural") deliberately
provides no fallback question. Against a book that marks nothing, the rule fires exactly as
written. This is precisely the drift HANDOFF next step 3 predicted — a reliable structural
task fused with a softer unmarked-prose one — and it makes the fix cheap: the Signature needs
to say that an *integrated* solution is still a procedure span.

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

## Per-module contract probes (2026-07-26)

`module_probes.py` drives each DSPy module **in isolation** against hand-built input and
asserts the contract its Signature claims, including the hazards HANDOFF records as known.
It complements the book-level sweep above: a book run tells you the pipeline's output, a
probe tells you which module owns a behaviour. DSPy's caches are disabled, so every probe is
a real call.

```bash
# the 8 text modules; add a PDF + out dir to also run the vision probe
PYTHONPATH=src uv run --extra mistral python robustness_test/module_probes.py \
    robustness_test/books/topology_morris.pdf /tmp/probe
```

**18/20 passing.** The two failures are the two open findings, and this is the useful part:
each now reproduces on **three synthetic nodes** instead of a whole book, which makes them
fast regression tests for a fix.

| Module | Probe | Result |
|---|---|---|
| `corrector` | stays within the tolerance guard | pass |
| `corrector` | repairs injected `f^{-2}`, `\bigcap`, `\infty` against the page image | pass ×3 |
| `extractor` | emits only structural node types | pass |
| `extractor` | preserves a display-math block verbatim | pass |
| `seam_merger` | merges a sentence split across the page break | pass |
| `seam_merger` | leaves a genuine boundary (proof end → next definition) unmerged | pass |
| `splitter` | splits a node packing three exercises | pass |
| `splitter` | leaves a single worked example unsplit | pass |
| `splitter` | breaks an embedded lead-in onto its own piece | pass |
| `instruction_finder` | tags the lead-in and nothing else | pass |
| `group_finder` | **marked** derivation → separate procedure span | pass |
| `group_finder` | **unmarked** derivation → procedure span | **FAIL** (Finding 1) |
| `statement_extractor` | takes the block's own number, not an in-text cross-reference | pass |
| `statement_extractor` | induces an open non-math type (`law`) | pass |
| `statement_extractor` | types a bare-formula exercise as an exercise | **FAIL** (Finding 2) |
| `procedure_extractor` | steps are a verbatim partition | pass |
| `procedure_extractor` | decomposes into more than one step | pass |
| `instruction_distributor` | governs two prove-problems, excludes a compute one | pass |

Notes on what the probes settle that the book sweep could not:

- **The corrector does its job.** Three subtle single-token math errors injected into a real
  page's transcription — an inverse-image index, a union flipped to an intersection, an empty
  set flipped to infinity — were **all** repaired against the page image, with no collateral
  rewriting. This is the first direct test of the claim in `core.llm`'s docstring.
- **The `number` hazard is not currently firing.** "2.1.12 Prove Proposition 2.1.13" yields
  `number='2.1.12'`. The guard carried into the statement extractor's Signature works.
- **Finding 2 is confirmed as a context problem, not a prompt-quality one.** The same pass
  types `Law 4.1 (Ohm's Law)` correctly as `law` — it is perfectly capable — but types a bare
  `$P \vee (Q \Rightarrow R)$` as `example`, because nothing tells it the block sits under
  "Write a truth table for…".
- **Step granularity varies run to run.** The same proof decomposed into 4 steps on one run
  and 6 on another. Both were exact verbatim partitions, so the contract holds; only the
  cut-points move. Worth knowing before treating step counts as a stable metric.

## Observability: what could and could not be seen

Worth stating plainly, because it shaped how much of this validation had to be reconstructed.

**There are no logs.** The pipeline emits no logging at all outside a single line in
`cli.py` (`Wrote assembled document to: …`). No stage — DSPy or otherwise — logs its inputs,
outputs, timings, or decisions. The per-run `.log` files from this sweep are one line each,
and that line came from the harness, not the pipeline.

**Tracing is gone.** `CLAUDE.md:41` still lists `core/tracing.py`, and `REPORT.md` documents
a `KMS_TRACE_DIR` workflow feeding `training/*/dataset.py`. Neither survived #21: there is no
`tracing` module and no `KMS_TRACE_DIR` reference anywhere in `src/`. The previous sweep
captured 227 trainable `{stage, inputs, outputs}` trace lines; **this one captured none.**

**What stood in for it.** Stage *outputs* were recovered two ways:

1. The harness kept the final `State`, giving the fully-attributed overlay per book — enough
   to check every structural invariant in this report, but only end-state, not per-call.
2. `~/.dspy_cache` retained **78 raw model responses** spanning **all nine** DSPy signatures
   — including the ChainOfThought reasoning, which is what produced the Lebl evidence above.

| Stage | Cached responses |
|---|--:|
| `statement_extractor.Identify` | 30 |
| `ingestion.extractor` | 8 |
| `ingestion.seam_merger` | 8 |
| `entity.group_finder` | 7 |
| `entity.splitter` | 7 |
| `ingestion.corrector` | 7 |
| `procedure_extractor.Decompose` | 6 |
| `entity.instruction_finder` | 3 |
| `entity.instruction_distributor` | 2 |

**The cache is not a replacement for tracing**, for three reasons: entries are keyed by a
hash of the request, so **the inputs are unrecoverable** and the pairing needed for training
data does not exist; the count is far short of the calls actually made (30 cached vs 67
statement extractions), so it is not a complete record; and it lives in an ephemeral
container. It is also an active hazard for validation — see the methodology correction under
Finding 1.

Restoring trace capture should come **before** any Signature tuning, since the tuning has no
data without it.

## Smaller notes
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
4. Restore trace capture before doing any Signature tuning, so the tuning has data — and add
   at least minimal per-stage logging, since the pipeline currently emits none.
5. Then the concept layer (`docs/CONCEPT-LAYER.md`), per HANDOFF.

## Reproducing

Harness, invariant checker, graph-mapping checker and the domain probe used for this run are
in the session scratchpad, not committed — they reproduce the retired JSON artifacts purely
for inspection. The pipeline itself was run unmodified.
