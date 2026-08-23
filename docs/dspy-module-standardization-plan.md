# DSPy Module Boundary Standardization Plan

## Goal

Standardize every DSPy-backed module around one predictable boundary:

```text
canonical application inputs
        ↓
Module.encode()
        ↓
exact DSPy signature fields
        ↓
LLM prediction
        ↓
Module.decode() + validation
        ↓
validated canonical output
```

The refactor should standardize input types, encoding ownership, decoding and
validation, deterministic application of model results, recording/replay, and
test coverage without flattening meaningful domain models into generic strings
or dictionaries.

## Implementation status

- [x] Slice 1: strict shared validators and decode-before-recording lifecycle.
- [x] Slice 2: window, span, splitter, partitioner, instruction, and fact
  extraction boundaries use typed inputs and validated positions.
- [x] Slice 3: boolean, confidence, role, governance, formatter, procedure,
  and entity-card gates reject permissive/coerced model values.
- [x] Slice 4: search candidates use canonical inputs; relevance decisions and
  decomposition plans are validated without silent invalid-index filtering.
- [x] Slice 5: enrichment and hub outputs validate cardinality, non-empty text,
  and declared structured result types.
- [x] Recording uses fresh run namespaces and loading rejects mixed schemas.
- [x] Live-smoke script now has an explicit destructive-DB guard and a separate
  replay-shape verification step.
- [x] Added direct module contract tests for governance, role partitioning,
  formatter/corrector routing, procedure writing, and structured hub outputs.
- [x] Focused and broader non-live verification: 402 passed, 3 skipped.
- [x] Authorized fresh-cache live ingestion completed with 59 nodes and 28
  triplets; the generated run replayed and validated as 22 uniform datasets.
- [x] Live replay verification uses `Example.toDict()` and rejects legacy
  `WindowNode` lists while preserving typed `ContentParts` values.

The repeated term/description cardinality loop in entity and predicate
 enrichment is intentionally kept local: these are parallel domain contracts,
not a new generic abstraction.

## Repository conventions

Every DSPy-backed module should:

```python
class SomeModule(module.Module):
    signature = SomeSignature
    record_name = 'some_module'
    use_chain_of_thought = False  # only when needed

    def encode(self, ...) -> dict:
        """Convert canonical inputs to DSPy signature fields."""
        ...

    def decode(self, prediction, **inputs):
        """Validate and convert the DSPy prediction."""
        ...
```

- Callers pass canonical application/domain types only.
- `encode()` is the only representation boundary into DSPy.
- `decode()` is the only prediction boundary back into application code.
- `ContentParts` is a DSPy adapter and normally belongs in signatures,
  `encode()`, and DSPy demonstrations—not in orchestration callers.
- Modules subclass `kms.core.module.Module`; do not hand-roll predictor setup,
  `aforward()`, `forward()`, or LM assignment.
- List-valued predictions use `module.as_list()` before validation.
- Malformed model output fails immediately; it is never clamped, sorted,
  deduplicated, filtered, repaired, or reinterpreted.
- Deterministic orchestration applies validated decisions to canonical state.
- Durable UUIDs and transient local positions remain separate contracts.

## Phase 0 — Freeze the baseline

- Keep this work separate from unrelated graph, identity, and ingestion edits.
- Run the existing focused test suite before each refactor slice.
- Inventory all `Module` subclasses, their canonical inputs, encoded inputs,
  decoded outputs, and existing validation.
- Record deliberate exceptions rather than forcing unrelated abstractions.

## Phase 1 — Define the contract

Document and enforce these rules:

### Canonical inputs

A caller may pass meaningful domain types such as:

- `content.Content`;
- `list[walker.WindowNode]`;
- `list[str]`;
- `str`;
- typed construction input models.

The caller should not format prompts, construct DSPy adapters, or decide how
images are represented in the LLM payload.

### Multimodal inputs

Use `content.Content` as the canonical multimodal representation:

- `Content` is the application/domain type;
- `ContentParts` is the DSPy signature adapter;
- `content.labeled_content_parts(nodes)` is used for position-emitting node
  windows;
- empty structured context is represented as empty `Content`, not as `""`,
  `[]`, or `None` interchangeably.

### Position semantics

Every positional signature must explicitly document:

- zero-based or one-based indexing;
- local or global scope;
- valid range;
- ordering requirements;
- duplicate behavior;
- empty-result behavior.

LLM-facing positions are transient local positions. They must not be confused
with node UUIDs or graph identities.

## Phase 2 — Audit and normalize inputs

Inventory and normalize modules by category.

### Labeled node-window modules

These should accept canonical node views and create `ContentParts` in
`encode()`:

- `Splitter`;
- `PedagogicalComponentFinder`;
- `InstructionRouter`;
- `InstructionGrower`;
- `StatementPartitioner`;
- `ProcedurePartitioner`;
- `_FactExtractor`.

Canonical input:

```python
list[walker.WindowNode]
```

Encoded input:

```python
content.labeled_content_parts(nodes)
```

### Typed multimodal modules

These should accept `content.Content` from callers and wrap it in
`encode()`:

- `RoleTyper`;
- `EntityEnricher`;
- `PredicateEnricher`;
- `StatementEnricher`;
- `ProcedureWriter` source procedure;
- `GovernanceJudge`;
- `SearchJudge`;
- `DecomposeJudge`;
- `QueryDecomposer`;
- entity and predicate hub adjudicators.

### Text-only modules

Keep text-only canonical inputs where text is the actual domain contract:

- formatter router/editor;
- procedure-need router;
- procedure writer statement;
- triplet decomposer;
- hub synthesis modules;
- entity-card modules;
- name-hub modules.

Their primary audit concerns are strict output validation and consistent
naming, not artificial conversion to `Content`.

### Image/OCR modules

OCR correction modules may accept an application-level image path and load the
model-facing `dspy.Image` inside `encode()`. The corrector remains full
resolution; other pipeline/search images use the configured downsampling
policy.

### Seam modules

`SeamNodeDTO` is a narrow seam-specific text DTO. Keep it as a deliberate
exception unless there is a clear canonical replacement. Do not introduce a
generic abstraction solely to eliminate this one domain-specific type.

## Phase 3 — Add shared validation helpers

Add narrowly scoped generic helpers alongside `as_list()` in
`kms.core.module`, or in a small `kms.core.contracts` module if the base module
becomes too crowded:

```python
def require_bool(value: object, field_name: str) -> bool:
    ...


def require_number(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    ...


def require_positions(
    value: object,
    *,
    field_name: str,
    upper_bound: int,
    unique: bool = True,
    ordered: bool = False,
) -> list[int]:
    ...
```

Helpers must:

- reject booleans where integers are expected;
- reject strings such as `"false"` for boolean fields;
- reject negative and out-of-range positions;
- reject duplicates where prohibited;
- reject unordered output where document order is required;
- never sort, clamp, deduplicate, or reinterpret.

`as_list()` remains only the DSPy cardinality adapter:

```text
None   → []
item   → [item]
tuple  → list
list   → list
```

## Phase 4 — Move validation to decode boundaries

### Position and span modules

Validate partitioners, splitters, and finders at the module boundary:

- correct item/model type;
- integer local positions;
- bounds within the supplied input;
- uniqueness where required;
- document order where required;
- non-reversed spans;
- non-overlap where required;
- valid empty-result behavior.

Existing downstream checks may remain as defense in depth, but malformed
predictions should be rejected before orchestration applies them.

### Boolean modules

Replace permissive coercion such as:

```python
return bool(prediction.needs_procedure)
```

with strict validation that accepts actual booleans only. Apply this to
procedure routing, formatter routing, search decomposition, entity-card
routing, governance, and role decisions.

### Governance

Validate that:

- `governs` is an actual boolean;
- `confidence` is numeric but not boolean;
- `confidence` is within `0.0` and `1.0`.

Do not clamp confidence.

### Search decisions

Validate each relevance decision's type, candidate index bounds, uniqueness,
and boolean relevance value.

Prefer a canonical search-judge input that preserves the exact candidate count,
such as `list[SearchResult]`, while `encode()` builds the labeled multimodal
content. Do not infer candidate count from `Content.parts`, since one candidate
may contain multiple text and image parts.

### Query decomposition

Validate that plans:

- contain integer in-range part indices;
- use contiguous groups when required;
- do not overlap;
- cover every input part exactly once;
- respect the maximum plan count;
- satisfy label requirements.

Remove silent invalid-index filtering from orchestration.

### Enrichment outputs

Validate structured enrichment results for item type, expected cardinality,
input-term correspondence, ordering, and required non-empty descriptions.

## Phase 5 — Record only valid predictions

`Module.aforward()` currently records before decoding. Change the lifecycle to:

```python
prediction = await self._call_predictor(kwargs)
output = self.decode(prediction, **inputs)

if self._recorder:
    self._recorder.record(...)

return output
```

This ensures malformed predictions do not enter replay datasets.

The recorder should continue storing encoded signature-form inputs, including
field-expanded `ContentParts` and content-addressed image sidecars—not raw
caller inputs.

## Phase 6 — Add contract-focused tests

### Shared helper tests

Cover strict booleans, numeric ranges, integer positions, bounds, duplicates,
ordering, and `as_list()` behavior.

### Encode tests

For each high-risk module, assert the exact encoded shape and multimodal part
ordering:

1. `Splitter`;
2. `PedagogicalComponentFinder`;
3. `InstructionRouter` and `InstructionGrower`;
4. statement/procedure partitioners;
5. `RoleTyper`;
6. `GovernanceJudge`;
7. `SearchJudge`;
8. `QueryDecomposer`;
9. enrichment modules.

### Decode tests

For each module, cover:

- valid output;
- `None` list output;
- singleton list output;
- malformed output type;
- invalid positions;
- duplicates;
- out-of-range values;
- non-boolean gates;
- invalid confidence;
- incomplete or overlapping search plans.

### Recording/replay tests

Verify that:

- encoded `ContentParts` serialize as `{content: {parts: ...}}`;
- images remain sidecars;
- empty contexts remain structured;
- failed decode results are not recorded;
- a clean run loads without schema mixing.

## Phase 7 — Validate with a clean live smoke run

After unit-level refactoring:

1. use a fresh timestamped output namespace;
2. clear only stale replay records when necessary;
3. run the live smoke pipeline;
4. load the resulting datasets;
5. verify one uniform current signature shape per dataset;
6. inspect representative records for structured `ContentParts`, ordered
   text/image parts, structured empty contexts, and no raw legacy node lists.

Recording/replay failures must remain independent of completed ingestion and
graph projection.

## Implementation slices

### Slice 1 — Base lifecycle and helpers

- add strict generic validators;
- add helper tests;
- decode before recording;
- verify invalid predictions are not recorded.

### Slice 2 — Window and positional modules

- partitioners;
- splitter;
- pedagogical finder;
- instruction finder;
- fact extractor.

### Slice 3 — Boolean and numeric judges

- procedure router;
- formatter routers;
- governance;
- role typer;
- search decomposition.

### Slice 4 — Search contracts

- canonicalize search-judge candidate inputs;
- validate candidate decisions;
- validate query decomposition plans;
- remove silent invalid-index filtering.

### Slice 5 — Enrichment and hub modules

- validate cardinality and structured outputs;
- standardize text/content annotations;
- preserve explicit domain-specific implementations instead of introducing
  generic hub abstractions.

### Slice 6 — Full verification

- run focused unit tests;
- run the broader unit suite;
- run a fresh live smoke test;
- verify replay loading and output uniformity.

## Acceptance criteria

The standardization is complete when:

- every DSPy module follows the same class and lifecycle pattern;
- callers pass canonical types only;
- `ContentParts` construction is isolated to signatures, `encode()`, and
  demonstrations;
- list outputs use `as_list()` followed by explicit validation;
- boolean outputs reject non-booleans;
- positional outputs reject malformed values immediately;
- malformed predictions are not recorded;
- no loader compatibility hacks are required;
- a fresh live run produces one uniform replay format.
