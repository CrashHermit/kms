# Style-cleanup work order

A punch list of the `docs/STYLE.md` rules the styling refactor (commit `a5e5c51`)
applied **inconsistently** or **missed**. Hand this to a local model to execute.
It is disposable — delete it once the work lands.

The machine-enforced rules (formatting, quotes, line length, import *ordering*,
`E`/`F`/`I`/`B`/`UP`) already pass. Everything here is a rule ruff **cannot**
enforce — §7.1 of STYLE.md says these "must be caught in review," and they weren't.

## How to apply & verify

Work top to bottom. After **each** part, run:

```bash
PYTHONPATH=src uv run ruff check src/ tests/     # F821 catches a mis-qualified import
PYTHONPATH=src uv run ruff format src/ tests/     # reflow to 80 cols after edits
PYTHONPATH=src uv run pytest -q                    # 147 passed, 4 skipped == green
```

The import conversions are the risky part, and they are **self-verifying**: if a
symbol is renamed to `module.symbol` but the module isn't imported (or vice-versa),
ruff reports `F821 undefined name` / `F401 unused import` and pytest fails. Do not
proceed to the next part until all three commands are clean.

---

## Part A — DONE (fixed in commit on branch; skip). Broken docstrings in `persister.py`

The refactor left **two stacked triple-quoted strings** on both `run` methods.
The first is the docstring; the second is dead code (a no-op string expression).
Worse, `NodePersisterNode.run`'s *active* docstring is **wrong** — it describes the
entity layer, not the node layer.

`src/kms/graph/persister.py`, `NodePersisterNode.run` (~line 36) — replace:

```python
    async def run(self, state: state.State) -> dict:
        """Flattens the three per-type overlays and upserts as the :Entity layer."""
        """Upserts the run's node stream as the graph's provenance layer."""
```

with:

```python
    async def run(self, state: state.State) -> dict:
        """Upsert the run's node stream as the graph's provenance layer."""
```

`EntityPersisterNode.run` (~line 58) — replace:

```python
    async def run(self, state: state.State) -> dict:
        """Flattens the three per-type overlays and upserts as the :Entity layer."""
        """Upserts the run's node stream as the graph's provenance layer."""
```

with:

```python
    async def run(self, state: state.State) -> dict:
        """Flatten the three per-type overlays and upsert them as the :Entity
        layer, then the procedural, concept, reference, and :USES layers."""
```

---

## Part B — §1.1 module imports in the `graph/` package

Rule (STYLE.md §1.1): **import the module, not its symbols.** Reference every
symbol module-qualified at the call site. The refactor did this for `core`
consumers but skipped all of `graph/`.

**Mechanical procedure, per file below:** replace the listed `from kms.graph.X import …`
lines with a single grouped `from kms.graph import …`, then prefix **every**
occurrence of each imported symbol in that file with its module (`entity_uuid`
→ `entities.entity_uuid`). Leave `from kms.core import …` lines untouched — they
are already module-style. Re-run `ruff format` afterward; a grouped import may
reflow.

Symbol → module reference map (applies across all these files):

| Symbol | Becomes |
|---|---|
| `NODE_LABEL`, `SOURCE_LABEL`, `node_uuid`, `node_label`, `node_properties`, `source_uuid`, `source_properties` | `nodes.<symbol>` |
| `ENTITY_LABEL`, `MENTION_LABEL`, `CANONICAL_LABEL`, `entity_uuid`, `entity_label`, `entity_properties` | `entities.<symbol>` |
| `PROCEDURE_LABEL`, `EVENT_LABEL`, `procedure_batches`, `event_rows`, `has_procedure_pairs`, `first_pairs`, `then_pairs`, `proof_events` | `procedures.<symbol>` |
| `CONCEPT_LABEL`, `concept_batches`, `instance_rows` | `concepts.<symbol>` |
| `canonical_batches`, `reference_rows`, `canonical_uuid` | `references.<symbol>` |
| `uses_rows` | `uses.<symbol>` |
| `database`, `driver`, `is_configured` | `db.<symbol>` |
| `ensure_schema` | `schema.<symbol>` |
| `persist_nodes`, `persist_entities`, `persist_procedures`, `persist_concepts`, `persist_references`, `persist_uses` | `writer.<symbol>` |

Per-file import-line replacements:

**`src/kms/graph/entities.py`** — replace `from kms.graph.nodes import source_uuid`
with `from kms.graph import nodes`. (One call site: `source_uuid(source)` → `nodes.source_uuid(source)`.)

**`src/kms/graph/references.py`** — replace `from kms.graph.entities import entity_uuid`
with `from kms.graph import entities`.

**`src/kms/graph/concepts.py`** — replace `from kms.graph.entities import entity_uuid`
with `from kms.graph import entities`.

**`src/kms/graph/procedures.py`** — replace the two lines
`from kms.graph.entities import entity_uuid` / `from kms.graph.nodes import source_uuid`
with `from kms.graph import entities, nodes`.

**`src/kms/graph/uses.py`** — replace the two lines
`from kms.graph.procedures import proof_events` / `from kms.graph.references import canonical_uuid`
with `from kms.graph import procedures, references`.

**`src/kms/graph/schema.py`** — replace the five `from kms.graph.X import …` lines with:

```python
from kms.graph import concepts, db, entities, nodes, procedures
```

**`src/kms/graph/writer.py`** — replace the whole `from kms.graph.…` import block
(concepts, db, entities, nodes, procedures, references, uses) with:

```python
from kms.graph import (
    concepts,
    db,
    entities,
    nodes,
    procedures,
    references,
    uses,
)
```

Then qualify every symbol from the map above throughout the file. This is the
largest file — the label constants (`NODE_LABEL`, `ENTITY_LABEL`, …) appear inside
many Cypher f-strings; each must become `nodes.NODE_LABEL`, `entities.ENTITY_LABEL`,
etc. Note `db.database` / `db.driver` inside every `async with driver().session(...)`.

**`src/kms/graph/persister.py`** — replace the three lines
`from kms.graph.db import is_configured` / `from kms.graph.schema import ensure_schema` /
`from kms.graph.writer import (…)` with:

```python
from kms.graph import db, schema, writer
```

Then: `is_configured()` → `db.is_configured()`, `ensure_schema()` →
`schema.ensure_schema()`, `persist_nodes(...)` → `writer.persist_nodes(...)` (and
the other five `persist_*`). Leave `from kms.core import models, state` as-is.

`nodes.py` and `db.py` define these symbols but import none of them from
`kms.graph`, so their import blocks need no change.

---

## Part C — `pipeline.py` and `cli.py` (READ THIS — not purely mechanical)

**`cli.py`** is clean to convert: replace `from kms.pipeline import run` with
`from kms import pipeline`, and `run(...)` → `pipeline.run(...)`.

**`pipeline.py` has a genuine collision** that pure module imports cannot resolve.
It imports node classes from three modules all named `definition` (and `problem`,
`theorem`): `entity.finders.definition`, `entity.attributors.definition`,
`entity.referencers.definition`. `from kms.entity.finders import definition` and
`from kms.entity.attributors import definition` would bind the same name twice.

Two honest options — **pick one, don't leave it half-done**:

- **Option 1 (recommended): document an exception, keep the symbol imports.**
  `pipeline.py` is pure wiring; every imported name is a unique, self-descriptive
  `*Node` class. Add a short exception clause to STYLE.md §1.1 ("Entry-point and
  graph-wiring modules that import uniquely-named classes are exempt") and leave
  `pipeline.py`'s imports as they are. This is the pragmatic, readable call.

- **Option 2: aliased module imports for full literal compliance.** e.g.

  ```python
  from kms.entity.finders import definition as definition_finder
  from kms.entity.attributors import definition as definition_attributor
  from kms.entity.referencers import definition as definition_referencer
  ```

  …and then `definition_finder.DefinitionFinderNode()` at each call site. Verbose
  and, in this file, arguably less readable than the status quo — offered only if
  strict literal compliance is the goal.

If you don't know which the maintainer wants, do Option 1 and note it.

---

## Part D — missing docstrings (§4). Drop-in text below.

**`src/kms/cli.py` `main()`** — currently no docstring. Insert as the first line
of the body:

```python
def main(argv: list[str] | None = None) -> None:
    """Run the pipeline from the command line: ``python -m kms.cli book.pdf out/``.

    Args:
        argv: Argument vector (pdf path, then output dir). Falls back to
            ``sys.argv[1:]`` when None, then to ``test.pdf`` / ``output``.
    """
```

**`src/kms/ingestion/ocr.py` `_require_key()`** — currently uses a `#` comment where
its sibling `core/llm.py:_require_key` uses a proper docstring. Replace the leading
comment block with a docstring so the two match:

```python
def _require_key() -> str:
    """Return the Mistral API key, raising a clear error if it is unset.

    Prefers ``MISTRAL_API_KEY`` (the documented ``.env.example`` name) and falls
    back to ``MISTRAL_OCR_API``, the name the hosted environment injects the
    secret under, so the front-end runs in either place. Raised on use rather
    than at import, so the module stays importable without credentials.
    """
    key = os.environ.get(MISTRAL_ENV_KEY) or os.environ.get('MISTRAL_OCR_API')
    ...
```

---

## Part E — cryptic local names (§3.5 / §3.7). Optional, low value.

The variable-rename pass was thorough; only a few short nested-scope names remain.
Rename only if you want full §3.5 compliance — none affect behavior:

- `src/kms/graph/db.py:143` — nested generator `def gen():` inside `__aiter__` →
  `def records_gen():` (or inline it). `gen` is the one genuinely cryptic name.
- `src/kms/entity/instruction_distributor.py:179` — `def pos(entity)` → `def position(entity)`.
  Update the two `key=pos` / `pos(p)` call sites.
- `src/kms/graph/entities.py:91` — comprehension `[_segment(s) for s in entity.bodylist]`
  → `for segment in entity.bodylist`.

Leave `repl` / `index_of` in `ocr.py` — `index_of` is descriptive and `repl` is the
conventional `re.sub` callback name; both are fine.

---

## Done when

`ruff check`, `ruff format --check`, and `pytest -q` (147 passed, 4 skipped) are all
green, and `grep -rnE '^from kms\.(graph|pipeline)\.[a-z_]+ import [a-z]' src/`
returns nothing (Part C Option 1 leaves `pipeline.py`'s `*Node` class imports, which
are capitalized, so that grep stays clean either way).
