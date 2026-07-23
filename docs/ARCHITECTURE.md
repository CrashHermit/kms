# KMS — Target Architecture

The package organization the pipeline uses. This layout is **realized** — the tree below
matches `src/kms/` as it stands. Read `HANDOFF.md` first for what the pipeline actually does
and why; this doc is only about *where the code lives* and *which direction dependencies point*.

**One rule from this doc is intentionally deferred** (a pragmatic MVP call; cheap to do later
because stages don't cross-import):

- **No `core/errors.py` hierarchy yet.** There's little ad-hoc failure signaling to justify one;
  `MistralOCRError` (in `ingestion/ocr.py`) covers the one real domain error today.

---

## The organizing principle: phases, keyed to the backbone

The most stable seam in this system is the **backbone data structure**, and it changes exactly
at the phase boundaries:

```
segments  ──▶  nodes  ──▶  entities  ──▶  graph
(per page)     (flat        (sparse        (cross-entity
               global       overlays)      edges, fusion,
               stream)                     completion)
```

Everything else — prompts, models, the number of entity types — is more volatile than these
boundaries. So the top level is organized around the phases, not around file types or entity
types. Each phase is a package that owns one span of that pipeline and nothing else.

---

## Target layout

```
src/kms/
  __init__.py              # public API: run(); __all__ = ["run"]
  pipeline.py              # the LangGraph DAG wiring — the ONLY file that knows the full order
  cli.py                   # __main__ entry: arg parsing + logging setup (no library logic on import)

  core/                    # shared center; depends on nothing, everything depends on it
    __init__.py
    models.py              # domain data: ASTNode, Segment, Entity, … + flatten/merge helpers (dspy/langgraph-free)
    state.py               # the LangGraph State TypedDict + reducer channels (imports models)
    llm.py                 # text_lm / teacher_lm / corrector_lm config
    tracing.py             # opt-in trace capture (the data→compile loop's raw material)
                           # (deferred: errors.py KmsError hierarchy)

  ingestion/               # PHASE 1: PDF → healed per-page nodes    (backbone: segments)
    __init__.py
    ocr.py                 # was mistral_ocr.py
    corrector.py
    extractor.py           # purely structural; no math-semantic typing
    seam_merger.py         # flatten_segments lives here — this is the segments→nodes boundary

  entity/                  # PHASE 2: nodes → sparse entity overlays  (backbone: nodes)
    __init__.py
    splitter.py            # was exercise_splitter.py; runs first, makes exercises atomic
    finders/               # the cursor-walk shape, one self-contained copy per type
      __init__.py
      problem.py
      definition.py
      theorem.py
    attributors/           # the enrichment shape, one self-contained copy per type
      __init__.py
      problem.py
      definition.py
      theorem.py
    instruction_distributor.py   # Problem-only; the lone per-type exception, kept at entity level

  graph/                   # PHASE 3: NOT STARTED — package exists so the seam is physical
    __init__.py            # (refs / fusion / completion modules added when the tier is built)

  output/
    __init__.py
    assembler.py           # entities + nodes → document.md / entities.json / nodes.json

training/                  # compile-time only (DSPy optimize); consumes traces, OUTSIDE the runtime pkg
tests/
```

---

## The load-bearing rules

### 1. Dependencies point backward only

This is the invariant the whole layout exists to encode:

```
ingestion ─┐
entity    ─┼─▶ core          core imports NONE of them.
graph     ─┤                 No stage imports a LATER stage.
output    ─┘                 Sibling stages meet only through the backbone in core.
```

- Every stage imports **from** `core`. `core` imports from no stage — it has no idea any stage
  exists.
- `ingestion` never imports `entity`; `entity` never imports `graph`; and so on. The graph tier
  reads entity *outputs*, never entity *code* — that direction is the point of giving it its own
  package now.
- Stages don't import each other's internals. They communicate only through the backbone
  channels in `core/state.py`. The single place that knows the full ordering is `pipeline.py`.

The payoff is testability: because a stage depends only on `core` (which `conftest` already
stubs), each stage stays independently testable, no keys or GPU required — the property the
current 46-test suite already relies on.

### 2. `core/` is the shared center, and it's plain (not `_core`)

`core/` holds the things that aren't a *stage* but that every stage reaches for: the domain
models, the orchestration state, LM config, tracing, and the error hierarchy. It's the floor
nothing is allowed to import upward from. Named `core/` (not `_core/`) because
`kms/__init__.py` is already the one public door — the underscore would be redundant signal.

### 3. `models.py` is separate from `state.py`

Two concerns, two modules: `models.py` holds the pure data containers (`Entity`, `ASTNode`, …)
and the pure helpers over them (`flatten_segments`, `merge_results_into_segments`); `state.py`
holds the LangGraph `State` TypedDict with its `operator.add` reducers. The models are what the
system is *about*; `State` is a mechanism of the runner we happen to use.

Keeping them apart keeps the domain types free of any LangGraph/dspy import, so a test, the graph
tier, or a future non-LangGraph runner can use `Entity`/`ASTNode` in isolation. `models.py`
imports only stdlib + pydantic; `state.py` imports `models` + langgraph. The one dspy-using
helper, `_load_dspy_image` (loads a page image at the corrector's LLM boundary), lives in
`ingestion/corrector.py` — its only caller — rather than contaminating `models.py`.

### 4. Entity layer is grouped by **stage**, not by type

`finders/{problem,definition,theorem}.py` rather than `problem/{finder,attributor}.py`. The
reusable unit here is the *shape* — the cursor-walk finder, the attributor pattern — and what
varies between types is prompt plus a little schema, not architecture. Grouping by stage keeps
the "one shape, three self-contained copies" honest and positions us to later collapse the
copies into a single parameterized module + per-type specs as a **local** change, without files
moving across the tree.

**When this flips to by-type:** if the per-type logic genuinely diverges — e.g. theorems grow
real proof-decomposition machinery and problems grow solution-handling until they no longer
share a shape. Today they share the shape, so by-stage wins. Because stages don't cross-import
(rule 1), switching later is a folder move, not a rewrite — a reversible bet, not a one-way
door.

Do **not** unify the three copies into one parameterized module yet. Keeping them self-contained
while the prompts are still being validated is deliberate: three copies you tune independently
beat one abstraction you fight. Extract the shared shape only after splitter/distributor
validation settles and the prompts stop moving.

### 5. Scaffold `graph/` before it's written

It's the next big piece and the one place rule 1 earns its keep — the graph tier must read
entity outputs and never be imported by the entity layer. Giving it a directory now (even as
stubs) makes that direction physical instead of a convention to remember.

### 6. Library and CLI are separate

`cli.py` owns `__main__`, argument parsing, and logging setup. Importing the library
(`kms.run`) must never run logic or configure logging. This is what lets `print()` disappear in
favor of `logging` cleanly: the CLI configures the root logger, the library only emits.

### 7. `training/` stays outside the runtime package

The DSPy compile/optimize workflow is compile-time only: it consumes the JSONL traces
`core/tracing.py` emits and produces optimized programs. It depends on the runtime package, not
the reverse, so it lives as a sibling to `src/`, never imported by any stage.

---

## Convention carve-outs this layout assumes

The code-style conventions apply, with three carve-outs that are **first-class rules here, not
exceptions** — they're what the conventions look like once they meet this stack:

- **Framework classes are allowed.** "Prefer standalone functions; don't use classes as
  namespaces" holds — but DSPy `Signature`/`Module`, LangGraph nodes, Pydantic models, and the
  `TypedDict` state are framework *contracts*. Subclass as the framework requires.
- **Pydantic fields may use literal defaults.** `contents: list[str] = []` on a `BaseModel` is
  safe — Pydantic copies defaults per instance. The "no mutable default argument" rule is a
  plain-function / dataclass rule. (The dataclasses in `models.py` correctly use
  `field(default_factory=...)`; both idioms coexisting is expected.)
- **`None` for absence is fine; `None` for *failure* is not.** Returning `None` for a
  legitimately-absent optional (e.g. `_load_dspy_image(None)`) is correct. Signaling an *error*
  with `None`/`False`/a sentinel string is not — raise a `KmsError` subclass instead.

---

## Naming migration

The import package should be renamed `module` → `kms` to match the project and drop the generic
name. This touches every import, so it's the one bit of churn worth doing early and pointless to
do late. `PYTHONPATH=src` and `package = false` are unaffected.

---

## Migration order (low-risk first)

1. `module` → `kms` rename (mechanical, wide, best done in one commit).
2. Carve out `core/` and split `state.py` → `core/models.py` + `core/state.py`.
3. Move stages into `ingestion/`, `entity/`, `output/`; group the finders/attributors by stage.
4. Add `core/errors.py` and replace ad-hoc failure signaling with the exception hierarchy.
5. Split `cli.py` out of `pipeline.py`; introduce `logging`, retire `print()`.
6. Scaffold `graph/` (empty stubs) so the seam exists before the tier is built.

Each step preserves the backward-dependency invariant; nothing later imports something earlier.
