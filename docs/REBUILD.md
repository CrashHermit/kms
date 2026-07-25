# Rebuild — ripping out AutoMathKG, building the block layer

**Audience: an implementing agent.** The file-level work order for moving the pipeline from
the three per-type AutoMathKG chains to the four-kind schema in `SCHEMA.md`. Read `SCHEMA.md`
first — it is the target; this is the route.

Supersedes `ENTITY-LAYER-REBUILD.md` (deleted), which specced procedures as extracted *from
within* an entity and gated the rip-out behind a build-alongside parity comparison. Both are
repealed: the finder emits procedure spans directly, and the old chains are deleted outright.

**No strangler fig.** The previous plan said build the new path beside the old one, measure,
then remove. That gate was never executable — there is no gold annotation, no metric, and no
scoring harness in the repo, so "reach parity" had no way to be satisfied. Maintaining two
entity layers, two channel sets, and two persister paths to service an unenforceable
condition is pure cost. Delete first.

What replaces it: the **twelve PDFs stay** (`robustness_test/books/` and
`tests/fixtures/books/`), and `robustness_test/REPORT.md` is kept as the written record of
known-good behavior. Re-run and compare against that narrative (Phase 6).

---

## Phase 1 — Delete

Nothing here is replaced in place; all of it goes.

```
src/kms/entity/finders/definition.py      src/kms/entity/finders/theorem.py
src/kms/entity/attributors/               (whole package: definition, problem, theorem)
src/kms/entity/referencers/               (whole package: definition, problem, theorem)
src/kms/graph/references.py               src/kms/graph/uses.py
src/kms/graph/realizes.py
```

Tests:
```
test_definition_finder.py       test_theorem_finder.py
test_definition_attributor.py   test_theorem_attributor.py    test_problem_attributor.py
test_definition_referencer.py   test_theorem_referencer.py    test_problem_referencer.py
test_graph_references.py        test_graph_uses.py            test_graph_realizes.py
```

Generated output and stale docs:
```
robustness_test/runs/     robustness_test/traces/
docs/UNIFIED-KG.md        docs/ENTITY-LAYER-REBUILD.md        docs/STYLE-CLEANUP-TASKS.md
```

`STYLE-CLEANUP-TASKS.md` declares itself disposable and targets files being deleted.
Keep `robustness_test/books/`, `robustness_test/REPORT.md`, `tests/fixtures/books/`,
`docs/STYLE.md`, and both paper mirrors — AutoMathKG stays as reference material even though
its structures are gone (its completion loop is still deferred future work).

**Renames:**
```
src/kms/entity/finders/problem.py  →  src/kms/entity/group_finder.py
tests/test_problem_finder.py       →  tests/test_group_finder.py
```
The `finders/` package disappears; `group_finder.py` sits directly under `entity/`.

---

## Phase 2 — Core models (`src/kms/core/models.py`)

**Delete:** `EntityType` · `ProcedureType` · `FIELDS` · `ACTIONS_ALL` · `REFERENCE_KINDS` ·
`BodySegment` · `Proof` · `Solution` · `Reference`.

**`Entity`** — drop `field`, `bodylist`, `proofs`, `solutions`, `refs`. Add `procedures:
list[Procedure]`. `type` becomes `str | None` (open, induced) instead of the closed enum.
Keep `members`, `id`, `label`, `number`, `title`, `contents`, `instruction`.

> The dataclass-`field` shadowing hazard in the current class body disappears with the
> `field` attribute, so the "declare above" ordering constraint can go too.

**`Procedure`** — new dataclass: `members: list[int]` (its own node ids, for
`:DERIVED_FROM`), `contents: list[str]`, `steps: list[str]`, `index: int`. No `type` — it is
derivable from the owning entity (`SCHEMA.md`, principle 5).

**`ASTNode`** — unchanged in memory, **including `role`**. It is now *transient*: the
instruction finder writes it, the group finder and instruction distributor read it, and it is
no longer persisted. Update the docstring to say so — this is the first field carrying the
persisted/transient distinction and it should be labelled.

**`flatten_entities`** — collapses from three overlay arguments to one list. With a single
finder producing a single partition, entities no longer overlap and the concatenate-don't-
merge rationale in the docstring is obsolete; ordering by first member position still stands.

---

## Phase 3 — Entity layer (`src/kms/entity/`)

Three stages replace nine.

### `group_finder.py` (from `finders/problem.py`)

**Keep verbatim:** the growing-window cursor walk, structural banking (`bank what a node is
seen to follow; grow when the only span reaches the edge`), `_window_from`, the budget
constants, the clamp/sort logic. This machinery is the reliable part and is not being
redesigned.

**Change only the Signature and the output type:**

- *What is a block* — widen from "a posed task" to any labeled pedagogical block:
  definitions, theorems, laws, worked examples, exercises, mechanisms. Keep the existing
  boundary rules verbatim — start at the block's own label, stop at the next label / section
  header / lead-in, subparts stay together, distinct base numbers are distinct blocks.
- *Emit two span roles.* Each span carries `role: "entity" | "procedure"`. This is a
  **closed, binary, structural** distinction — not the open `type` taxonomy, which the
  statement extractor induces later. A theorem and its proof are **two adjacent spans**, not
  one fused span.
- *Required tag.* The role must be a required output field so the model commits rather than
  defaulting.

**Keep:** exercise lead-ins (`role="instruction"`) are boundaries, never members.

**One partition, no overlap.** A node now belongs to at most one span. Accepted consequence:
an inline definition embedded mid-proof is absorbed into the containing block rather than
surfacing separately, where three independent finders would each have caught it. `REPORT.md`
praises exactly this ("6 of 8 definitions are inline bold terms… all found"), so **expect
definition counts to drop on the Phase 6 re-run** — that is intended, not a regression.

**Post-check (cheap drift detector):** every `procedure` span should follow an `entity` span.
Log violations; do not fail on them.

### `statement_extractor.py` (new)

One pass over each `entity` span. Fills `type` (open, induced), `label`, `number`, `title`,
`contents`.

Crib the label/number/title prompt language and `_strip_label_prefix` / `_contents` helpers
from the deleted `attributors/problem.py` — they are good and tested. **Drop** `field`,
`field_choices`, `solution_start`, and all member-splitting: the extractor no longer
restructures the entity, it reads a span and fills attributes.

### `procedure_extractor.py` (new)

One pass over each `procedure` span. Produces a `Procedure` with `contents` and `steps`, and
attaches it to an entity.

- **Decomposition is universal.** Every procedure decomposes into steps, whatever it derives.
  AutoMathKG restricted `bodylist` to Thm/Def, which left every solution stepless — the
  procedural spine was empty for exactly the exercise-heavy books the pipeline targets.
- **Steps are a verbatim partition.** Carry the partition + verbatim language from the
  deleted `attributors/theorem.py` `ProofBodylist` signature — *"every part belongs to exactly
  one piece, no repeats and no omissions"* / *"copy each piece's text VERBATIM"*. Drop the
  `action` label and the role taxonomy.
- **Attachment:** nearest preceding `entity` span. A procedure span with no preceding entity
  (a proof deferred pages after its theorem) is left **unattached** — persisted as an orphan
  `:Procedure` with no `:HAS_PROCEDURE` edge and a flag. Do **not** drop it; the text and its
  steps are real extracted content, and an attachment pass can find orphans later. The
  cross-reference rule ("Proof of Theorem 2.4") is deliberately **not** implemented yet.
- **Absence is structural.** No procedure span means no procedure. There is no "is there
  something to work out?" call — generation for unsolved exercises is deferred (`SCHEMA.md`).

### Unchanged

`splitter.py`, `instruction_finder.py`, `instruction_distributor.py` — pedagogy, already
domain-free. The distributor retargets from `problem_entities` to the single channel.

---

## Phase 4 — Graph layer (`src/kms/graph/`)

- **`entities.py`** — delete `entity_label` and the `:Mention` role label; entities get the
  bare `:Entity` label with `type` as a property. Delete `_segment` and the `bodylist`
  JSON-string property.
- **`procedures.py`** — `EVENT_LABEL` → `ACT_LABEL = 'Act'`. Delete `procedure_label` and the
  per-kind labels. Drop `kind` from `procedure_uuid` and `event_uuid` (→ `act_uuid`): it
  existed to disambiguate a shared entity, which one partition makes impossible. Drop
  `action` from the act properties. Replace `_derivations` (which walked `proofs` then
  `solutions`) with a walk over the unified `procedures` list. Add `:DERIVED_FROM` pairs for
  procedures — new, and possible now that procedures have their own members.
- **`concepts.py`** — **gut, don't delete.** Keep `normalize_concept`, `concept_uuid`,
  `concept_properties`, `concept_batches`. Delete `FIELD_CONCEPT`, `_entity_concepts`, and
  `instance_rows` (it hardcodes `entity_uuid`, so it cannot emit edges from procedures or
  acts, and has no slot for the `phrase` property). The layer goes dark; the identity scheme
  survives for `CONCEPT-LAYER.md` to build on.
- **`schema.py`** — drop the reference/canonical constraints and indexes; rename the `:Event`
  constraint to `:Act`.
- **`writer.py`** — delete `persist_references`, `persist_uses`, `persist_realizes`. Update
  the procedure/act writes for the renamed label and dropped properties.
- **`persister.py`** — `EntityPersisterNode` persists entity → procedure → (concept, dark).

`nodes.py` is untouched **except** dropping `role` from `node_properties`.

---

## Phase 5 — Pipeline (`state.py`, `pipeline.py`)

**State:** `problem_entities` / `definition_entities` / `theorem_entities` → one `entities`
channel. Add whatever channel the finder uses to hand spans to the extractors.

**Wiring** — nine entity nodes become three:

```
… → splitter → instruction_finder → node_persister
  → group_finder → statement_extractor → procedure_extractor
  → instruction_distributor → entity_persister → END
```

Sequential, not parallel. The two extractors are logically independent, but both write the
`entities` channel, and sequencing avoids a reducer clash for no meaningful latency cost.

---

## Phase 6 — Verify

```bash
uv run ruff format . && uv run ruff check .
PYTHONPATH=src uv run pytest -q
PYTHONPATH=src uv run python -c "from kms.pipeline import build_graph; build_graph(); print('ok')"
```

Grep `src/` for stragglers — all should be empty:
```
EntityType|ProcedureType|FIELDS|ACTIONS_ALL|REFERENCE_KINDS
problem_entities|definition_finder|theorem_finder
canonical|REFERENCES|REALIZES|bodylist|\.proofs|\.solutions|\.refs
```

Then re-run the twelve fixture PDFs and read the output against `robustness_test/REPORT.md`.
Its documented behaviors are the checklist. Specifically confirm:

- theorem + proof produce **two spans**, correctly attached
- solutions now carry **steps** (this was empty before — the headline change)
- exercises stay atomic; lead-ins are never absorbed
- `instruction` still stamps on grouped exercises and **not** on self-contained books
- provenance integrity holds: no dangling member refs, no empty-content nodes
- **expected:** definition counts drop where inline definitions sat inside larger blocks

---

## Not touched

Ingestion (`ocr`, `corrector`, `extractor`, `seam_merger`), the pedagogy stages, the
`:Source`/`:Node` provenance layer, `output/assembler.py`, and `docs/STYLE.md`.

`HANDOFF.md` and `CLAUDE.md` are rewritten **after** this lands — they describe what *is*, so
writing them first would document a system that does not exist. Two sections must survive
verbatim into the new `HANDOFF.md`: the **validation results** and the **container gotchas**.
