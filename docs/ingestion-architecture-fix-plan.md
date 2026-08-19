# Ingestion Architecture Fix Plan

## Purpose

Replace the current partial ingestion refactor with the direct typed architecture.
Do not add compatibility layers, heuristic repairs, or migration shims.

## Core rule

```text
LLM decides meaning.
Code owns explicit data boundaries.
Malformed LLM output fails immediately.
```

Remove:

- clamping;
- sorting model spans;
- gap filling;
- overlap repair;
- positional reinterpretation;
- source inference by majority vote;
- compatibility wrappers;
- fallback paths.

If an LLM response violates its contract, the stage fails with a diagnostic.
Fix the prompt, schema, model, or architecture rather than adding another
repair rule.

---

## 1. Reset fact provenance

### Target

`src/kms/construction/triplet_extractor.py`

### Contract

The LLM returns fact text only. It does not return node IDs or positions.

```python
class _FactInput(BaseModel):
    text: str
```

Fact extraction is anchored to one deterministic source node:

```text
anchor node + context_before + context_after → LLM fact extraction
```

Python assigns the fact's provenance:

```python
{'text': fact.text, 'node_ids': [anchor_node.id]}
```

Neighboring nodes are context only. They may help resolve a split sentence,
pronoun, definition, or reference, but they do not become evidence ownership.
Every triplet produced from the fact inherits exactly that one anchor ID.

Images may be present in context. If an image-backed node is the semantic
source of a fact, it may be the anchor and remains valid provenance.

### Remove

Delete all positional/stable-ID interpretation and normalization logic,
including support for zero-based positions, one-based positions, or multiple
LLM-generated identifier formats.

### Tests

Add coverage proving:

- every fact has exactly one deterministic anchor node ID;
- context nodes are not added to provenance;
- the fact signature has no provenance output field;
- image-bearing anchors remain valid provenance.

Delete tests for the retired positional output contract.

---

## 2. Retire the extractor

The extractor is retired. Do not repair it or introduce an extractor v2.

### Actions

- Remove `src/kms/construction/extractor.py`.
- Remove its workflow node and edges.
- Remove active imports and factories.
- Remove tests that exercise the retired extractor.
- Remove obsolete extractor-only live scripts and configuration.
- Remove obsolete gold extractor data if no active tool consumes it.

The active path is:

```text
OCR provider response
    → canonical Source / Document / Node models
    → correction/formatting
    → remaining structural stages
```

---

## 3. Remove model-output repair behavior

### Finder spans

`src/kms/core/walker.py` must not clamp, sort, silently bank, fill gaps, or
repair malformed model spans. Invalid spans fail at the boundary with a
diagnostic.

Do not introduce a second reconciliation subsystem.

### Finder ownership

Instruction and pedagogical ownership must not silently suppress content. A
malformed ownership result fails before it mutates the candidate stream.

### Splitter

`src/kms/construction/splitter.py` must not clamp model positions, redirect
invalid positions, repair duplicate positions, or rewrite source fragments to
make them fit. It consumes one explicit contract and fails on invalid output.

### General audit

Audit active ingestion code for model-output repair patterns such as:

```text
min / max used to clamp model values
sorted used to repair model ordering
set used to hide duplicates
normalize / repair / fallback helpers
inferred default positions
```

Remove only behavior that changes or guesses the semantic meaning of LLM
output. Ordinary deterministic ordering of already-valid results is fine.

---

## 4. Make procedure creation source-local

`ProcedureCreatorNode` must not scan or rebuild procedures globally during
normal ingestion.

Target flow:

```text
current source state
    → typed procedure input bundle
    → procedure creator
    → procedure enrichment
    → final projector
```

Procedure creation must not:

- scan all graph procedures;
- create or update procedures for unrelated sources;
- use global graph state as hidden input;
- perform maintenance behavior during ingestion.

Maintenance rebuilds remain explicit maintenance operations.

Add a test proving that ingesting source A cannot create or modify source B
procedures.

---

## 5. Fully wire the typed bundle architecture

### State

Carry explicit typed channels for:

- source/document bundle;
- statement enrichment inputs;
- procedure enrichment inputs;
- canonical knowledge;
- final projection inputs.

Remove legacy dictionary-shaped channels where typed bundles replace them.

### Workflow

The intended workflow is:

```text
OCR
  → correction/formatting
  → active structural stages
  → triplet extraction
  → entity/predicate enrichment inputs
  → entity/predicate enrichment
  → source-local hub inputs
  → source-local hub construction
  → triplet hub construction
  → statement/procedure composition
  → statement/procedure enrichment
  → statement/procedure hub construction
  → final projector
```

### Enrichment boundary

Enrichment nodes receive fully assembled inputs:

```text
target UUID
typed multimodal Content
canonical knowledge
```

They may call the LLM, generate descriptions, create embeddings, and return
enrichment results.

They may not compose graph context, call `compose_statement`, call
`compose_procedure`, read source-scoped Neo4j knowledge, or discover their own
inputs. Those operations belong upstream.

---

## 6. Delete generic legacy hub paths

Migrate all active callers away from:

- `learning_hub_builder`;
- generic learning-hub persistence;
- deleted `graph/persister.py`;
- generic learning-hub graph modules;
- compatibility imports retained only for the old architecture.

Use domain-specific ownership:

```text
entity_hubs
predicate_hubs
triplet_hubs
statement_hubs
procedure_hubs
```

Each domain owns its input bundle, builder, graph persistence contract,
configuration, and tests. Only genuinely domain-neutral mechanics such as
vector search or clustering remain shared.

After migration, delete obsolete modules instead of preserving shims.

---

## 7. Correct source ownership

Source-local hubs have an explicit source from `HubBuildBundle.source`.
Every member used to construct one must belong to that source. If a wrong
source enters the bundle, fail immediately. Do not infer source ownership by
majority vote.

Meta hubs may intentionally span multiple sources. Their cross-source
membership must be explicit; do not fabricate a source owner from the most
common member.

### Images

Images are intentional participants in the source/document mix.

Ensure that:

- image-bearing nodes remain in canonical documents;
- multimodal content preserves image ordering and identity;
- image nodes are not dropped merely because they lack text;
- source ownership applies to image-backed evidence as well as text-backed
evidence;
- typed bundles carry image content through enrichment and composition;
- source-local hub construction does not silently exclude image-derived
members.

This is a semantic inclusion rule, not a positional shortcut.

---

## 8. Unify document ownership

Use one canonical representation:

```text
models.Source
  → models.Document
    → models.Node
```

Adapt provider responses into these models at the OCR boundary. Remove the
parallel OCR-specific document representation from active downstream
contracts. Runner and state must carry the canonical source/document/node
structure rather than two competing representations.

---

## 9. Verification before live ingestion

Before clearing Aura again, add focused tests for:

1. fact anchor provenance;
2. image-bearing fact/context handling;
3. finder malformed-output failure;
4. splitter malformed-output failure;
5. source-local procedure creation;
6. typed enrichment bundle flow;
7. domain-specific hub/projector imports;
8. source ownership rejection;
9. intentional cross-source meta-hub membership;
10. canonical document flow.

Then run the complete non-live suite and import/compile checks.

Do not run another destructive live smoke test until:

- no retired modules are imported;
- no generic legacy persistence path is active;
- no active ingestion stage clamps or repairs LLM output;
- typed bundles traverse the workflow;
- procedure creation is source-local.

Only then clear Aura and run one clean end-to-end live test.

## Final target

```text
OCR/provider data
    → canonical typed Source/Document/Node models
    → LLM semantic decisions
    → deterministic source/anchor ownership
    → typed enrichment bundles
    → domain-specific hub builders
    → final graph projector
```

No ambiguous model-generated IDs.
No repair loops.
No hidden graph reads inside enrichment.
No global procedure creation.
No generic legacy hub layer.
Images remain part of the intended multimodal source mix.
