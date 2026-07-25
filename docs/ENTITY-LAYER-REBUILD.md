# Entity-layer rebuild — the block finder (removal + build)

**Audience: an implementing LLM/agent.** File-level steps for replacing the three per-type entity
chains with the **block-finder** entity layer. The **design and rationale live in
`GENERALIZATION.md`** ("Entity layer"); this doc is the *what-changes-where*. Read that section first.

**Supersedes** the earlier "rename `problem`→`task`, keep the task attributor" spec — that framing is
dead. The new shape is **detect → attribute → procedure-find** with all three per-type chains gone.

## What the rebuild does

Replace the three per-type finders + attributors + referencers with **three general stages**:

1. **Block finder** — the current finder's cursor-walk (`entity/finders/problem.py` is the base — it
   already has the generalized "posed task in any textbook" prompt) with only the "what is a block"
   clause widened to "any labeled pedagogical block", emitting a **span only** (no type — the per-type
   finders emit only spans too; the type was hardcoded by which finder ran, never classified).
2. **Universal attributor** — one pass reading each entity's content, filling `label / number / title
   / content` **and `type`** (what kind — definition/theorem/law/…, an **open induced property**, not
   the finder's job and not a separate classify stage).
3. **Procedure finder** — one pass over all entities: *is there something to work out, shown or
   absent?* → extract shown steps / create for a posed problem / defer a proofless statement / skip.
   (No separate task/statement classify — the procedure finder's routing subsumes it.)

## Strangler-fig ordering (correctness gate)

**Do not delete the per-type finders until the block finder reaches quality parity on real math
books** (extraction quality is the one thing the probes did not validate — `GENERALIZATION.md`,
Evidence/Open questions). Build the three stages **alongside** the existing chains, measure, then
remove. Greenfield removes migration cost, not the risk of deleting validated code before its
replacement works.

## What is NOT touched

- Pedagogy: `splitter`, `instruction_finder`, `instruction_distributor`.
- Provenance: the `:Source`/`:Node` layer and `graph/nodes.py`, `graph/persister.py` node stage.
- Procedural + semantic graph modules: `graph/procedures.py`, `concepts.py`, `references.py`,
  `uses.py`, `realizes.py` — they consume the flattened entity list and keep working (a definition
  entity simply has no procedure; `entity_label` yields `:Entity:<Type>` from whatever `type` holds).

## Files removed (only after parity — step 5)

```
src/kms/entity/finders/definition.py        src/kms/entity/finders/theorem.py
src/kms/entity/attributors/definition.py    src/kms/entity/attributors/theorem.py
src/kms/entity/attributors/problem.py       (→ replaced by the one universal attributor)
src/kms/entity/referencers/definition.py    src/kms/entity/referencers/theorem.py
```
`entity/finders/problem.py` is **repurposed** into the block finder (not deleted). The referencers
are replaced by open-relation extraction in a **separate** step (build-sequence step 2 in
`GENERALIZATION.md`); leave `entity/referencers/problem.py` until then.

Tests: delete the def/thm finder/attributor/referencer tests; retype/rename the problem-* tests to
the block finder + universal attributor + procedure finder.

## Cascade to handle (`src/kms/core/models.py`)

- **`EntityType` / entity `type`**: the closed enum (DEFINITION/THEOREM/PROBLEM) gives way to an
  **open `type` property filled by the attributor** (an induced string: definition/theorem/law/…).
  Because it is open, store it as a **property on a bare `:Entity`** — do **not** mint a per-type
  Neo4j label (open labels explode; `kind = label, type = property`). `graph/entities.py` currently
  builds `:Entity:<Type>` from the closed enum via `entity_label` — change it to drop the per-type
  label and rely on the `type` property (already written to the vertex).
- **`Entity`**: `field` → moves to the concept layer (drop once conceptualization lands, not before —
  `concepts.py` still reads it). The two narrow fields **`solutions` + `proofs` unify into one
  `procedures` list** (each `{type, steps}`) that the procedure finder fills and `procedures.py`
  reifies — this is a rename/merge, `procedures.py` already models `:Procedure` with an open `type`.
- **`flatten_entities`** + `state.py` channels + `pipeline.py` wiring + `graph/persister.py`: collapse
  the three `*_entities` channels/chains into the single block-finder channel; the persister flattens
  one overlay instead of three.

## Verification

```
uv run ruff format . && uv run ruff check .          # clean
PYTHONPATH=src uv run pytest -q                        # green
PYTHONPATH=src uv run python -c "from kms.pipeline import build_graph; build_graph(); print('ok')"
```
Grep for stragglers in `src/` (should be empty): `definition_finder|theorem_finder|problem_entities|
EntityType\.(DEFINITION|THEOREM|PROBLEM)`.
