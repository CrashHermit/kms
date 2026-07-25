# Entity-layer rebuild — the block finder (removal + build)

**Audience: an implementing LLM/agent.** File-level steps for replacing the three per-type entity
chains with the **block-finder** entity layer. The **design and rationale live in
`GENERALIZATION.md`** ("Entity layer"); this doc is the *what-changes-where*. Read that section first.

**Status (2026-07-25): the BUILD half is done; only the REMOVAL half is left.** The three general
stages exist and are wired as the `block` entity layer (`KMS_ENTITY_LAYER=block`, or
`build_graph('block')`), running alongside the per-type chains, which are still the default. What
remains is the correctness gate below — measure, then remove.

**Supersedes** the earlier "rename `problem`→`task`, keep the task attributor" spec — that framing is
dead. The new shape is **detect → attribute → procedure-find**, and the end state has all three
per-type chains gone.

## What the rebuild does — BUILT

Replace the three per-type finders + attributors + referencers with **three general stages**:

1. **Block finder** — `entity/finders/block.py`: the finder's cursor-walk, verbatim, with only the
   "what is a block" clause widened to "any labeled pedagogical block", emitting a **span only** (no
   type — the per-type finders emit only spans too; the type was hardcoded by which finder ran,
   never classified).
2. **Universal attributor** — `entity/attributors/universal.py`: one pass reading each entity's
   content, filling `label / number / title / contents` **and `type`** (what kind —
   definition/theorem/law/…, an **open induced property**, not the finder's job and not a separate
   classify stage).
3. **Procedure finder** — `entity/procedure_finder.py`: one pass over all entities: *is there
   something to work out, shown or absent?* → extract shown steps / create for a posed problem
   (marked `generated`) / defer a proofless statement / skip. (No separate task/statement classify —
   the procedure finder's routing subsumes it.)

The referencers are already gone: build-sequence step 2 replaced all three with the single
open-relation `entity/referencers/open.py`, which serves every channel.

## Strangler-fig ordering (correctness gate)

**Do not delete the per-type finders until the block finder reaches quality parity on real math
books** (extraction quality is the one thing the probes did not validate — `GENERALIZATION.md`,
Evidence/Open questions). The three stages are built **alongside** the existing chains; what is left
is to measure, then remove. Greenfield removes migration cost, not the risk of deleting validated
code before its replacement works.

**How to measure.** Both layers are selectable on the same book — `KMS_ENTITY_LAYER=block` versus the
default `per-type` — and everything downstream of the collector is identical, so the comparison is
clean. Run both over the same pages and compare: entity count and span boundaries (does the block
finder find the same blocks, with the same extents?), the induced `type` against the type the
per-type chain hardcoded, label/number/title fidelity, and whether the procedure finder splits
statement from derivation where the per-type attributors' `proof_start`/`solution_start` did. The
Hefferon §III.1 pages used in `HANDOFF.md`'s Validation section are the obvious first case, since the
per-type numbers there are already recorded.

## What is NOT touched

- Pedagogy: `splitter`, `instruction_finder`, `instruction_distributor`.
- Provenance: the `:Source`/`:Node` layer and `graph/nodes.py`, `graph/persister.py` node stage.
- Semantic stages and graph modules: `collector`, `conceptualizer`, `dependency_finder`,
  `referencers/open.py`, and `graph/procedures.py`, `concepts.py`, `dependencies.py`,
  `references.py`, `uses.py`, `realizes.py`. They all consume the collected entity list and are
  layer-agnostic by construction (a definition entity simply has no procedure; the open `type` is a
  property, so nothing derives a label from it).

## Files removed (only after parity — step 5)

```
src/kms/entity/finders/problem.py           src/kms/entity/finders/definition.py
src/kms/entity/finders/theorem.py
src/kms/entity/attributors/problem.py       src/kms/entity/attributors/definition.py
src/kms/entity/attributors/theorem.py       (→ replaced by the one universal attributor)
```
The referencers are already replaced (step 2), so nothing is left to delete there.

Tests: delete `test_{problem,definition,theorem}_{finder,attributor}.py` — the block finder,
universal attributor, and procedure finder already have their own.

## Cascade to handle (`src/kms/core/models.py`)

Most of this cascade is **already done** (build-sequence steps 1–3): `Entity.type` and
`Procedure.type` are open strings stored as properties (no per-type Neo4j labels), `field` is gone in
favour of the conceptualizer's `concepts`, `proofs` + `solutions` are unified into one `procedures`
list, and `flatten_entities` takes a list of overlays so the count is a wiring detail.

What is left, at removal time:

- **`EntityType`** can be deleted outright — it survives only as the values the per-type finders
  stamp.
- **`state.py` channels + `pipeline.py` wiring**: drop `problem_entities` / `definition_entities` /
  `theorem_entities`, delete `_wire_per_type_layer` and the `ENTITY_LAYERS` / `KMS_ENTITY_LAYER`
  selection with it, and wire the block chain unconditionally between the node persister and the
  collector. `collector.OVERLAY_CHANNELS` then names one channel.

## Verification

```
uv run ruff format . && uv run ruff check .          # clean
PYTHONPATH=src uv run pytest -q                        # green
PYTHONPATH=src uv run python -c "from kms.pipeline import build_graph; build_graph(); print('ok')"
```
Compare both layers on the same book before removing anything:
```
PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/per-type
KMS_ENTITY_LAYER=block PYTHONPATH=src uv run --extra mistral python -m kms.cli book.pdf out/block
```
Grep for stragglers in `src/` after the removal (should be empty): `definition_finder|theorem_finder|
problem_entities|EntityType`.
