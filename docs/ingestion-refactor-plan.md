# Ingestion Refactor Plan

**Status:** Planning
**Scope:** Single-document construction only
**Out of scope:** Cross-source/meta rebuilds, application search, graph schema redesign

## Goal

Treat document construction as a self-contained in-memory computation. Neo4j
should be a durable projection at the end of construction, not a coordination
store used between construction stages.

The target flow is:

```text
PDF
  -> OCR and page artifacts
  -> correction, formatting, and structural reconstruction
  -> in-memory source analysis
  -> in-memory source-local knowledge construction
  -> complete construction bundle
  -> one durable Neo4j projection
```

The meta layer remains separate:

```text
persisted source-local graph
  -> explicit meta maintenance command
  -> graph queries across sources
  -> meta hubs
```

## Small-model execution checklist

Follow these tasks in order. Complete one checkbox at a time. Do not start a
later phase until its acceptance checks pass. Keep each change small enough to
review with `git diff`.

### Rules for every task

- [ ] Read `docs/STYLE.md` before editing Python.
- [ ] Use module-qualified local imports and 80-character lines.
- [ ] Do not add NetworkX or another general-purpose graph library.
- [ ] Do not move or modify the meta rebuild workflow in this refactor.
- [ ] Do not run live OCR, model, or Aura tests unless explicitly requested.
- [ ] Run focused tests after each task, then `git diff --check`.
- [ ] Do not preserve obsolete compatibility wrappers; update active callers.
- [ ] Keep source content separate from derived descriptions, embeddings, hubs,
      and generated steps.

### Phase 0 — establish the seam

- [x] Add a typed `ConstructionBundle` module for one document.
  Include request metadata, source nodes, statements, procedures, triplets,
  derived knowledge, indexes, and diagnostics. Do not add graph behavior.
- [x] Add bundle validation helpers for missing IDs, duplicate IDs, invalid
  memberships, invalid evidence references, and source-scope violations.
- [ ] Add tests that construct a small bundle and validate every relationship.
- [x] Add a conversion from the current early pipeline state into the bundle.
  Keep this conversion pure and do not query Neo4j.
- [x] Make the test suite pass before moving any later stage.

### Phase 1 — add FAISS without changing behavior

- [x] Add FAISS as the vector-index dependency.
- [x] Implement an application-owned `VectorIndex` wrapper around
  `IndexFlatIP`.
- [x] Normalize vectors before insertion and query.
- [x] Return stable record keys and scores; never expose FAISS row numbers to
  construction stages.
- [x] Define deterministic tie-breaking using stable record keys.
- [x] Add tests for empty indexes, dimension mismatch, top-k, normalization,
  duplicate keys, and deterministic ties.
- [ ] Do not change production construction to use FAISS yet.

### Phase 2 — replace graph composition with pure functions

- [x] Implement in-memory `compose_statement(bundle, statement_id)`.
- [x] Implement in-memory `compose_procedure(bundle, procedure_id)`.
- [x] Implement in-memory `knowledge_for_statement(bundle, statement_id)`.
- [x] Implement in-memory `knowledge_for_procedure(bundle, procedure_id)`.
- [x] Preserve document order, images, member IDs, triplet evidence, and steps.
- [x] Add fixture-based tests comparing the pure functions with expected
  composed content.
- [x] Add pure bundle-based statement and procedure enrichment input assembly.
- [x] Add explicit in-memory enrichment input loader nodes.
- [x] Refactor enrichment nodes to consume typed inputs and keep persistence
  as a separate boundary.
- [x] Remove enrichment-node source-local Neo4j reads.
- [x] Add source-local procedure materialization input loading.
- [x] Refactor procedure creation to consume typed inputs instead of graph
  composition reads.
- [x] Ensure procedure creation cannot enumerate unrelated sources.

### Phase 3 — move source-local retrieval in memory

- [x] Use the application-owned exact FAISS index for source-local entity
  and predicate hub candidate retrieval. Keep the index local to each hub
  assignment stage until shared bundle ownership is required.
- [x] Add explicit source-local filtering in the index or candidate loader.
- [x] Replace source-local Neo4j vector searches in entity and predicate hub
  construction with bundle indexes.
- [ ] Preserve the existing embedding thresholds and LLM adjudication rules.
- [ ] Represent memberships with dictionaries and sets, not a graph object.
- [ ] Build TripletHub membership using intersection of subject-, predicate-,
  and object-hub fact sets.
- [x] Add tests proving that unrelated source records are never candidates.
- [ ] Add tests proving TripletHub intersections preserve exact relation order.

### Phase 4 — make all source-local construction in memory

- [x] Add typed EntityHub/PredicateHub component records from enrichment
  outputs and expose exact ordered TripletHub membership helpers.
- [x] Refactor entity hub construction to return typed bundle updates.
- [x] Refactor predicate hub construction to return typed bundle updates.
- [x] Add source-local TripletHub construction from typed exact ordered
  memberships; retain graph-backed maintenance rebuilds.
- [x] Pass canonical EntityHub/PredicateHub context and assignments through
  state into source-local TripletHub synthesis.
- [x] Split StatementHub and ProcedureHub construction and graph ownership
  into parallel domain-specific modules; remove the generic learning-hub
  builder and state channel.
- [x] Refactor statement and procedure enrichment to return typed descriptions
  and embeddings in workflow state for downstream hub construction.
- [x] Refactor generated procedures and steps to remain source-provenanced.
- [x] Remove Neo4j session parameters from source-local construction functions.
- [ ] Add a fake-model end-to-end test that runs construction with Neo4j
  unavailable.
- [ ] Assert that the resulting bundle passes validation.

### Phase 5 — simplify LangGraph state

- [x] Make the bundle the only canonical source-analysis payload in LangGraph
  for enrichment, procedure, and hub-input construction.
- [ ] Keep request data, diagnostics, and persistence status outside the
  domain records.
- [ ] Remove obsolete parallel result channels after their collectors are
  replaced by explicit artifact updates.
- [ ] Ensure every stage declares its input artifact and output artifact.
- [ ] Ensure stable IDs and local finder positions are never interchanged.
- [ ] Update workflow tests for the new stage boundaries.

### Phase 6 — add the final Neo4j projector

- [x] Add a final construction projector for accumulated statement/procedure
  enrichment and source-local statement/procedure hub artifacts.
- [x] Implement one final projector as the construction-time graph write
  entry point.
- [x] Move raw, derived, and generated procedure records behind the final
  projector boundary.
- [x] Project durable source records after construction completes, preserving
  all provenance and membership relationships.
- [ ] Make projection source-scoped and retryable.
- [ ] Add document version or run identity before implementing replacement.
- [ ] Reconcile stale nodes, memberships, and derived records on re-ingestion.
- [ ] Add projector tests with a fake repository/session.
- [ ] Run the configured Neo4j integration test only after unit tests pass and
  only with explicit authorization.

### Phase 7 — finish and remove obsolete paths

- [x] Remove construction-time calls to graph composition, graph vector
  search, and graph enrichment queries for enrichment, procedure, and hub input
  assembly.
- [x] Remove obsolete construction-time writer calls.
- [ ] Keep graph query/writer code needed by application search and meta
  maintenance.
- [x] Update this plan with decisions made for the completed refactor slice;
  versioning, reconciliation, and retry semantics remain explicit follow-up
  work.
- [ ] Run the full non-live test suite.
- [ ] Run formatting, linting, type checks, and `git diff --check`.
- [ ] Review the final diff for accidental NetworkX, global-source reads,
  positional-ID lookups, or meta-workflow changes.

### Definition of done

- [ ] A complete single-document construction run succeeds without Neo4j.
- [ ] Construction uses typed in-memory records, dictionaries/sets, and FAISS
  indexes only; no NetworkX is required.
- [ ] Embedding and reranking providers remain replaceable service boundaries.
- [ ] Construction cannot read or modify unrelated sources.
- [ ] The complete bundle can be validated and persisted after construction.
- [ ] Neo4j is used by construction only through the final projector.
- [ ] Meta rebuild remains a separate graph-backed maintenance workflow.
- [ ] All focused and non-live regression tests pass.

## Architectural decisions

### FAISS for in-memory vector indexes

Use FAISS for construction-time vector retrieval where a vector index is
needed. Start with exact inner-product search over normalized vectors, which
is equivalent to cosine similarity and is deterministic. Keep the index behind
an application-owned interface so construction does not depend directly on
FAISS types.

Why FAISS:

- mature, fast, and purpose-built for vector search;
- supports exact and approximate indexes;
- can use compact numeric arrays rather than Python float objects;
- gives us a straightforward upgrade path from a flat exact index to HNSW or
  another ANN index if corpus size requires it.

Initial policy:

- normalize vectors before indexing;
- use an exact flat index first (`IndexFlatIP`);
- use source-local indexes during document construction;
- do not introduce ANN behavior until measured corpus sizes justify it;
- preserve deterministic tie-breaking outside FAISS using stable record IDs.

FAISS is an index, not the source of truth. The typed construction bundle owns
records and metadata; FAISS stores/searches vectors and returns record keys.

### No NetworkX initially

Do not add NetworkX to the ingestion path yet. The current construction
operations need typed records, membership maps, reverse indexes, and vector
retrieval more than they need a generic graph object.

Use explicit in-memory structures such as:

```text
records_by_id
nodes_by_id
members_by_block
components_by_hub
source_to_records
triplets_by_hub_tuple
relationships_by_type
```

A graph-library adapter can be added later for visualization or algorithms if
we identify a concrete traversal requirement. It must remain a projection of
the typed bundle, not the authoritative domain model.

### External model providers remain external

The following remain service boundaries during construction:

- Mistral OCR;
- local LLM calls through the configured model router;
- Voyage embedding generation;
- Nemotron reranking, where document retrieval needs it.

Their outputs are held in memory and used by later stages. Moving data and
indexes in memory does not imply replacing these providers with local models.

### Meta construction stays separate

`rebuild-meta` remains a separate graph-backed maintenance workflow. It may
query all persisted source-local hubs and build cross-source hubs later. This
refactor must not pull meta construction into document ingestion.

## Target data model

Introduce a typed construction bundle as the authoritative in-memory object.
The exact class names are open, but the shape should be similar to:

```text
ConstructionBundle
  request
  document
  pages
  nodes
  instructions
  pedagogical_units
  statements
  procedures
  steps
  triplets
  entity_components
  predicate_components
  entity_hubs
  predicate_hubs
  triplet_hubs
  statement_descriptions
  procedure_descriptions
  indexes
  diagnostics
```

Separate the bundle into logical areas rather than one flat state dictionary:

```text
request       input paths, selected pages, source metadata, run ID
document      raw/corrected/formatted page artifacts and provenance
structure     pages, nodes, spans, source coordinates
analysis      instructions, pedagogical units, statements, procedures, triplets
knowledge     components, hubs, descriptions, embeddings, steps
indexes       FAISS indexes and typed lookup maps
persistence   projection status and counts
diagnostics   warnings, validation failures, timings, model metadata
```

### Source versus derived data

Source-backed data:

- raw OCR response and page artifacts;
- corrected/formatted page content;
- structural nodes and source spans;
- instructions and pedagogical memberships;
- source-backed statements, procedures, and triplets.

Derived data:

- embeddings;
- entity/predicate hubs;
- triplet hubs;
- descriptions;
- generated procedures and steps;
- statement/procedure learning hubs.

Every derived item should retain provenance IDs back to source records where
possible. Generated text must not silently become a second source of truth.

### Stable IDs and positions

Every node and relationship must distinguish:

```text
stable ID       durable identity and provenance
order position  current traversal position
source span     location in the artifact that produced the record
```

A stable ID must not be treated as a list index. Finder windows may use local
positions, but their outputs must be resolved back to stable IDs immediately.

## Target construction stages

### Stage 1: Acquisition and structural processing

Keep the existing conceptual order, but make every stage consume and return
explicit artifacts:

```text
OCR
  -> block correction
  -> formatting
  -> extraction
  -> seam merger
  -> exercise splitter
  -> instruction finder
  -> pedagogical component finder
  -> statement/procedure builder
```

The raw OCR artifact is immutable. Correction and formatting create explicit
versions or patches rather than making the current representation ambiguous.

### Stage 2: In-memory semantic construction

Replace graph reads with bundle operations:

```text
triplet extraction
  -> entity/predicate enrichment
  -> entity component embeddings
  -> predicate component embeddings
  -> FAISS source-local indexes
  -> entity hub assignment
  -> predicate hub assignment
  -> triplet hub construction
  -> statement/procedure composition
  -> procedure creation and step splitting
  -> statement/procedure enrichment
  -> statement/procedure learning hubs
```

The stage outputs should be typed updates to the bundle. They should not write
to Neo4j or query Neo4j.

### Stage 3: Persistence projection

Create an explicit projector that receives a complete bundle:

```text
Neo4jProjector.persist(bundle)
```

The projector owns:

- durable document/page/node records;
- source-backed statements, procedures, instructions, and triplets;
- provenance and membership relationships;
- generated steps;
- source-local hubs and derived descriptions;
- embeddings and vector-index properties;
- cleanup/reconciliation for the current document version.

Construction code should not call graph writers directly.

## Search and ranking design

Define storage-neutral interfaces for the construction path.

### Embedding provider

The existing remote embedding client should implement an interface equivalent
to:

```python
class EmbeddingProvider(Protocol):
    async def embed(contents) -> list[list[float]]:
        ...
```

### Vector index

Add an application-owned FAISS wrapper:

```python
class VectorIndex(Protocol):
    def add(self, keys, vectors) -> None:
        ...

    def search(self, query, top_k) -> list[VectorMatch]:
        ...
```

`VectorMatch` should contain the stable record key and similarity score. The
wrapper should own normalization, index configuration, and deterministic
post-sorting.

### Candidate records

Do not make retrieval results Neo4j records. Use a storage-neutral candidate:

```text
candidate_id
text
image reference
embedding score
rerank score
metadata
```

Reranking and LLM relevance judging can operate on these candidates regardless
of whether they came from FAISS or a future persisted search backend.

### In-memory composition

Replace graph-dependent helpers with pure bundle functions:

```text
compose_statement(bundle, statement_id)
compose_procedure(bundle, procedure_id)
knowledge_for_statement(bundle, statement_id)
knowledge_for_procedure(bundle, procedure_id)
```

These functions must preserve source order, image associations, memberships,
triplet evidence, and generated steps.

## Refactor phases

### Phase 0: Freeze the target contracts

Write tests and types for:

- stable IDs versus local positions;
- source versus derived fields;
- document/page/node ownership;
- span-to-member resolution;
- source-local versus cross-source scope;
- projector input/output;
- complete bundle validation.

Acceptance criterion: the contracts are explicit enough that no construction
stage needs to infer whether an integer is an ID or a position.

### Phase 1: Introduce the bundle and indexes

Add the typed `ConstructionBundle`, lookup maps, relationship indexes, and
FAISS wrapper. Initially populate the bundle from the current early-stage
outputs.

Add unit tests for:

- vector normalization and cosine-equivalent scores;
- stable key to FAISS row mapping;
- deterministic tie-breaking;
- source filtering;
- empty indexes and missing keys;
- bundle relationship lookups.

### Phase 2: Move composition and enrichment in memory

Implement pure in-memory equivalents for graph composition and canonical
knowledge assembly. Refactor statement enrichment, procedure enrichment, and
procedure creation to consume the bundle.

Remove their same-run Neo4j reads and writes.

Acceptance criterion: a fake-model construction test can complete with no
Neo4j session factory.

### Phase 3: Move source-local hub construction in memory

Refactor entity, predicate, triplet, statement, and procedure hub construction
to operate on bundle records and FAISS indexes.

Keep the existing LLM adjudication and synthesis behavior. Change storage and
candidate retrieval, not the semantic decision contracts.

Acceptance criterion: source-local hub outputs match the expected typed
relationships and are reproducible for fixed model outputs.

### Phase 4: Make construction a pure analysis workflow

Rebuild the ingestion workflow around explicit stage inputs and outputs. The
workflow should produce a complete bundle and should not require Neo4j to run.

Neo4j configuration should be needed only if the caller requests projection.

Acceptance criterion: the complete construction pipeline runs in memory with
Neo4j disabled and fake source/assertion projectors.

### Phase 5: Add the final Neo4j projector

Project the completed bundle in one deliberate persistence boundary. Add
version/reconciliation behavior so re-ingestion can retire stale records and
relationships rather than relying on `MERGE` alone.

Acceptance criterion: projection is idempotent, source-scoped, and does not
leave stale records after a changed segmentation.

### Phase 6: Keep meta maintenance graph-backed

Leave `rebuild-meta` as a separate workflow. It may continue to query Neo4j,
load source-local hubs, and persist cross-source meta hubs. Do not make it a
requirement of document construction.

## Questions to answer before implementation

These are design questions, not blockers for writing the plan. Resolve them
before the corresponding phase begins.

### Bundle and ownership

1. What should the authoritative top-level type be called: `DocumentBundle`,
   `ConstructionBundle`, or `KnowledgeBundle`?
2. Should page artifacts and semantic records live in one bundle or in separate
   `DocumentArtifact` and `KnowledgeBundle` objects?
3. Which records are source-backed overlays, and which are purely generated?
4. Should excluded OCR furniture remain as nodes with a disposition, or be
   retained only in the raw page artifact?
5. What exact provenance fields must every derived record carry?

### Identity and versions

6. What is the stable node identity strategy across re-ingestion?
7. Do we need explicit `document_version` and `run_id` immediately?
8. Which IDs are exposed to the graph and application APIs, and which are only
   internal construction keys?
9. How should a split node retain provenance to its original OCR block?

### FAISS and vectors

10. Which vector families need indexes during construction: entity,
    predicate, statement, procedure, or only source-local hubs?
11. Should indexes be one per vector family, one per source, or one combined
    index with metadata filtering?
12. Should we start with `IndexFlatIP` everywhere and measure before adding
    HNSW?
13. Where should embeddings be cached during retries or repeated runs?
14. Are multimodal image embeddings needed during construction, or only for
    later application retrieval?
15. What is the expected largest source-local vector count?

### Reranking and search

16. Which construction stages actually need reranking rather than embedding
    similarity plus an LLM adjudicator?
17. Should the existing search API be split into a storage-neutral retrieval
    layer and an application/Neo4j adapter?
18. Should construction-time relevance judging return annotations in the bundle
    or only filtered candidate lists?

### Semantic construction

19. Should generated procedures be created before source-backed procedures are
    persisted, or should all procedure creation happen in memory first?
20. Which canonical knowledge is allowed to influence source-local procedure
    generation?
21. Should source-local hub assignment consider only the current document, or
    preload existing source-local hubs for incremental ingestion?
22. What should happen when an existing source-local hub conflicts with a new
    document-local candidate?
23. Which current graph-derived hub algorithms must remain bit-for-bit or
    relationship-for-relationship compatible?

### Persistence

24. Is the final projection one transaction, or a resumable staged projection?
25. What is the source-version replacement policy for re-ingestion?
26. Should derived hubs be persisted in the same projection as durable source
    records, or rebuilt afterward from the durable projection?
27. What should happen if model construction succeeds but Neo4j projection
    fails?
28. Do we need an exportable bundle artifact for retrying projection without
    repeating model calls?

### Testing and observability

29. Which fixture becomes the canonical end-to-end in-memory construction
    fixture?
30. Which invariants must be checked after every stage?
31. Should every stage emit an artifact manifest and validation report?
32. How will we compare in-memory outputs with the current graph-backed outputs
    while the refactor is underway?
33. What limits should apply to document size, vector count, and model call
    concurrency?

## Required invariants

Every implementation phase should preserve these invariants:

- every retained source span is represented exactly once in document order;
- every derived record points back to source evidence where applicable;
- stable IDs are never used as list positions;
- instruction and pedagogical memberships refer to existing node IDs;
- statement/procedure members preserve source order;
- triplet evidence references valid source nodes;
- FAISS row IDs resolve to stable record IDs;
- vector scores are reproducible for fixed vectors;
- source-local construction cannot read unrelated sources;
- construction can run without Neo4j;
- projection can be retried from a completed bundle;
- meta rebuild is not required for ingestion success.

## Proposed first implementation slice

Start with a narrow vertical slice rather than rewriting every module at once:

```text
existing OCR through triplet extraction
  -> ConstructionBundle
  -> in-memory entity/predicate component indexes
  -> in-memory statement composition
  -> fake-model enrichment
  -> bundle validation
  -> no Neo4j
```

Use `ea2e_ch1_review.pdf` as the dense exercise fixture and the existing
calculus fixture as the shared-instruction fixture. Once this slice is stable,
move hub assignment and procedure construction into the same bundle, then add
the final projector.

## Decisions recorded for the completed slice

- `ConstructionBundle` is the authoritative single-document payload; document
  artifacts and derived knowledge remain separate fields within that bundle.
- Source-backed nodes, statements, procedures, and triplets retain their
  source evidence. Descriptions, embeddings, hubs, generated procedures, and
  steps are derived artifacts and are not alternate source content.
- Composition and knowledge assembly are pure bundle operations. The active
  enrichment, procedure-materialization, and hub-input paths do not call graph
  composition or graph knowledge queries.
- Construction-time graph access is limited to the final projector. Graph
  query helpers for composition and source-local enrichment work items are
  retired; application search and the separate maintenance workflows keep
  their graph-backed query paths.
- Source-local procedure and enrichment inputs are assembled from the bundle,
  so unrelated sources cannot enter those stages. Neo4j remains optional until
  final projection is requested.
- FAISS remains behind the application-owned vector-index wrapper. The current
  slice does not claim that every construction hub stage has migrated to bundle
  indexes yet.
- Document versioning, run identity, projection retry/reconciliation, and
  stale-record cleanup are intentionally deferred until the projector contract
  is stabilized.

## Decision summary

For the greenfield ingestion refactor:

- **Yes:** use FAISS for in-memory vector indexes;
- **Yes:** keep embeddings and candidate metadata in memory during construction;
- **Yes:** keep reranker/model providers behind interfaces;
- **No:** do not add NetworkX yet;
- **Yes:** use typed dictionaries, sets, and relationship indexes first;
- **Yes:** make construction independent of Neo4j;
- **Yes:** keep Neo4j as the final durable projection;
- **Yes:** keep meta rebuild as a separate graph-backed maintenance loop.
