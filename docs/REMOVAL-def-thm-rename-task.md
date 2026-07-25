# Removal spec — delete definition/theorem finders, rename `problem` → `task`

**Audience: an implementing LLM/agent.** This is an exact, self-contained change spec. Execute it
as written. Context lives in `docs/GENERALIZATION.md` (the why); this doc is the *what*.

## Goal (two operations, one commit-worthy change)

1. **Remove the math-typed declarative extractors**: the `definition` and `theorem` finders,
   attributors, and referencers — and everything that only exists to serve them.
2. **Rename `problem` → `task`** everywhere (finder/attributor/referencer, `EntityType`, the state
   channel, the graph label, pipeline wiring, tests). "Problem" is too STEM-specific; the entity is
   a *posed task* (worked example or exercise), a pedagogical universal.

After this change the entity layer has **one finder chain: `task`** (finder → attributor →
referencer → instruction_distributor → entity_persister). The pipeline still runs end to end and
produces `document.md`; when Neo4j is configured it persists `:Entity:Task` vertices with their
solution procedures, concepts, and references.

## Explicitly OUT of scope (do NOT do these)

- **Do not build the replacement** (no "statement finder", no AutoSchemaKG triple extractor, no
  conceptualization stage). Declarative extraction (definitions/theorems/laws) is intentionally
  *absent* after this change — it will be rebuilt later as open triple extraction (see
  `GENERALIZATION.md`). This is a deliberate, greenfield interim state.
- **Do not strip the `field` attribute** from the task attributor. `field` still feeds
  `graph/concepts.py`; it is removed only when the concept layer is redesigned (a later, separate
  change). Leave it.
- **Do not delete the `proofs` / `bodylist` fields** on `models.Entity`, and **do not modify the
  graph modules** `procedures.py` / `uses.py` / `realizes.py` beyond `EntityType` renames in their
  tests. They degrade to no-ops gracefully (see "Dormant, keep as-is"). Removing them is a separate
  future cleanup.
- **Do not touch** the ingestion stages, the splitter, the instruction_finder, the provenance
  (`:Source`/`:Node`) layer, or the procedural/reference/concept graph layers' logic.

---

## A. Files to DELETE (6)

```
src/kms/entity/finders/definition.py
src/kms/entity/finders/theorem.py
src/kms/entity/attributors/definition.py
src/kms/entity/attributors/theorem.py
src/kms/entity/referencers/definition.py
src/kms/entity/referencers/theorem.py
```

And their tests (6):

```
tests/test_definition_finder.py
tests/test_definition_attributor.py
tests/test_definition_referencer.py
tests/test_theorem_finder.py
tests/test_theorem_attributor.py
tests/test_theorem_referencer.py
```

## B. Files to RENAME `problem` → `task` (3 + 3 tests)

Rename the file **and** the node class inside it (`ProblemXNode` → `TaskXNode`), and scrub
`problem`/`Problem` from docstrings/identifiers to `task`/`Task`:

| From | To | Class rename |
|---|---|---|
| `src/kms/entity/finders/problem.py` | `finders/task.py` | `ProblemFinderNode` → `TaskFinderNode` |
| `src/kms/entity/attributors/problem.py` | `attributors/task.py` | `ProblemAttributorNode` → `TaskAttributorNode` |
| `src/kms/entity/referencers/problem.py` | `referencers/task.py` | `ProblemReferencerNode` → `TaskReferencerNode` |
| `tests/test_problem_finder.py` | `test_task_finder.py` | update imports + `EntityType.PROBLEM` → `TASK` |
| `tests/test_problem_attributor.py` | `test_task_attributor.py` | same |
| `tests/test_problem_referencer.py` | `test_task_referencer.py` | same |

Inside these files: the DSPy signatures already say "task" in prose (the problem finder was
neutralized earlier) — keep that, just fix the node/channel names. The attributor writes
`entity.solutions` and reads `entity.field` — **both stay** (solutions → procedures; field →
concepts).

---

## C. Files to EDIT

### 1. `src/kms/core/models.py`

- **`EntityType`** (around line 45): remove `DEFINITION` and `THEOREM`; rename
  `PROBLEM = 'problem'` → `TASK = 'task'`. Update the class docstring (no longer "three math-semantic
  categories" — now the pedagogical `Task` entity; definitions/theorems are handled by the future
  general layer). The value string `'task'` becomes the `:Entity:Task` graph label and the `type`
  property value — intended.
- **`Entity` dataclass** (line 146): keep all fields. Update the `solutions` comment
  `# Problem-only` → `# Task-only` and `instruction` `# Problem-only` → `# Task-only`. **Leave
  `proofs` and `bodylist`** (now unused by any producer, but retained — see Dormant).
- **`flatten_entities`** (line 265): change the signature from three per-type lists to one:
  ```python
  def flatten_entities(task: list['Entity'], nodes: list[ASTNode]) -> list['Entity']:
  ```
  Body: `entities = list(task)` (drop the `+ definition + theorem` concat at line 282). Keep the
  ordering-by-first-member logic and id assignment unchanged. Update the docstring (one overlay, not
  three; "problem finder" → "task finder").
- **`FIELDS`, `ACTIONS_ALL`, `REFERENCE_KINDS`**: **no change** — still used by the task attributor
  (`field`) and referencer (`tactic`, target kinds). `REFERENCE_KINDS = ['definition', 'theorem']`
  stays: references still *target* definition/theorem canonical hubs even though we no longer extract
  them as in-corpus entities.

### 2. `src/kms/core/state.py`

- Remove channels `definition_entities` and `theorem_entities` (lines 49–50).
- Rename `problem_entities` → `task_entities` (line 48); update its comment to "written by the task
  finder".
- Update the class docstring: "three `*_entities` channels" → one `task_entities` channel; drop the
  parallel-overlay language that assumed three finders.

### 3. `src/kms/pipeline.py`

- **Imports** (lines 53–63): delete the six `Definition*`/`Theorem*` imports; rename the three
  `Problem*` imports to `Task*`.
- **Instantiation** (lines 87–95): delete def/thm instances; rename `problem_*` → `task_*`.
- **`add_node`** (lines 113–121): delete the six def/thm node registrations; rename the three
  `problem_*` node names → `task_*`.
- **`chains`** (lines 176–180): reduce to a single tuple:
  ```python
  chains = [('task_finder', 'task_attributor')]
  ```
- **Edges** (lines 191–197): delete the `definition_*`/`theorem_*` → referencer → persister edges.
  Keep the task chain: `task_attributor → task_referencer → instruction_distributor →
  entity_persister` (rename from `problem_*`).
- **Docstrings** (module docstring + `build_graph` docstring): update the stage-order line and the
  "three per-type chains" language to a single `task` chain. Remove "problem/definition/theorem".

### 4. `src/kms/graph/persister.py`

- `EntityPersisterNode.run` (lines 66–71): change the `flatten_entities` call to the new one-channel
  signature:
  ```python
  entities = models.flatten_entities(state.get('task_entities', []), state.get('nodes', []))
  ```
- Update the class + module docstrings: "three per-type overlays" → the single `task` overlay;
  "problem chain's instruction distributor" → "task chain's".

### 5. `src/kms/entity/instruction_distributor.py`

- It reads/writes the problem channel and stamps `Problem.instruction`. Update: the state channel
  `problem_entities` → `task_entities`, and any `Problem`/`problem` identifiers/docstrings → `Task`/
  `task`. It still keys on `EntityType.TASK` entities and `role="instruction"` nodes. (The
  `instruction` attribute name on the entity is unchanged.)

### 6. `src/kms/entity/instruction_finder.py`

- Verify only: it tags nodes `role="instruction"` and should have no per-type entity references. If
  it names the `problem` channel anywhere, update to `task`; otherwise no change.

---

## D. Dormant — keep as-is (do NOT edit logic, only test retypes)

After removal, only `EntityType.TASK` entities exist, and no entity carries `proofs`/`bodylist`. The
following graph modules keep working because they iterate empty lists / filter by type and degrade to
no-ops. **Leave their logic untouched:**

- `graph/procedures.py` — `_derivations` still yields `solutions` (tasks have them); the `proofs`
  branch just yields nothing. `proof_events`/`first_pairs` for proofs go empty.
- `graph/uses.py` — proof-step `:USES` needs proof events; produces none. No-op.
- `graph/realizes.py` — ties definition/theorem mentions to canonicals; with none, produces none.
  No-op.
- `graph/entities.py`, `graph/concepts.py`, `graph/references.py` — unchanged; `entity_label`
  yields `:Entity:Task`, `field`→concept and refs→canonical still work.

These are intentional dormancy, not bugs. A later change removes the proof machinery when the
declarative layer is rebuilt.

---

## E. Tests to UPDATE (not delete)

Retype `EntityType.DEFINITION`/`THEOREM`/`PROBLEM` → `EntityType.TASK` and fix channel names/imports
in:

- `tests/test_flatten_entities.py` — update to the one-channel `flatten_entities(task, nodes)`
  signature; drop the def/thm inputs.
- `tests/test_graph_persister.py` — `problem_entities` → `task_entities`; drop def/thm channels.
- `tests/test_instruction_distributor.py` — `Problem` → `Task`, channel rename.
- `tests/test_graph_procedures.py`, `tests/test_graph_uses.py`, `tests/test_graph_realizes.py` —
  these exercise the (now dormant but still present) proof machinery. **Retype** the constructed
  entities from `EntityType.THEOREM` → `EntityType.TASK` to keep exercising `procedures.py`/`uses.py`/
  `realizes.py` (the code still exists). Do not delete these tests.
- `tests/test_graph_entities.py`, `tests/test_graph_concepts.py`, `tests/test_graph_references.py`,
  `tests/test_graph_db_integration.py` — retype any def/thm/problem `EntityType` usages to `TASK`.
- `tests/test_corrector.py` — the match is almost certainly an incidental "problem" in prose; verify,
  likely no change.

Test count will drop (the 6 deleted files). That is expected.

---

## F. Verification (must all pass)

```
uv run ruff format . && uv run ruff check .        # both clean
PYTHONPATH=src uv run pytest -q                      # green
```

Then a smoke check that the graph builds with one chain:

```
PYTHONPATH=src uv run python -c "from kms.pipeline import build_graph; build_graph(); print('ok')"
```

Grep for stragglers (should return nothing in `src/`):

```
rg -n "definition_finder|theorem_finder|DefinitionFinderNode|TheoremFinderNode|problem_entities|EntityType.PROBLEM|EntityType.DEFINITION|EntityType.THEOREM" src/
```

## G. Suggested order

1. `models.py` (`EntityType`, `flatten_entities`) → 2. delete the 6 modules → 3. rename the 3
   `problem` modules → 4. `state.py` → 5. `pipeline.py` → 6. `persister.py` +
   `instruction_distributor.py` → 7. tests (delete 6, rename 3, retype the rest) → 8. run F.

## H. Docs to update for consistency (secondary, after code is green)

`CLAUDE.md`, `docs/HANDOFF.md`, `docs/ARCHITECTURE.md` describe "three per-type finders
(problem/definition/theorem)" throughout. Update those references to the single `task` chain and note
that the declarative layer is pending re-implementation (per `GENERALIZATION.md`). Non-blocking for the
code change, but do not leave them contradicting the code.
