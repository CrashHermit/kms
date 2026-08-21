# Node Order and Durable Identity Migration

## Purpose

Change the ingestion pipeline so the ordered node list is the source of truth for
node order. Do not use one integer field as both a node identity and a stream
position.

This document is an implementation specification. Follow it literally. Make the
smallest coherent change that satisfies the contracts below. Do not add a
compatibility layer after the migration is complete.

## Final design

A document contains an ordered list:

```text
nodes[0] -> nodes[1] -> nodes[2] -> ...
```

The list provides the current order and adjacency. There is no stored `next`
pointer and there is no stored mutable stream ID.

A node has durable identity only when it crosses the graph persistence boundary:

```python
Node
├── source content and provenance
└── uuid: durable graph identity
```

The current list position is derived, never stored on the node:

```python
for position, node in enumerate(nodes):
    ...
```

## Definitions

### Position

A position is the node's current zero-based index in an ordered `list[Node]`.

Positions are temporary. They may change after:

- seam merging;
- exercise splitting;
- insertion or deletion;
- document reordering;
- flattening.

Use positions for:

- walking the document;
- context windows;
- local LLM window positions;
- ordering statements and procedures during the current run;
- deriving adjacent `HEAD` and `NEXT` graph edges.

Never use a position as a durable graph identity.

### Durable node UUID

A node UUID identifies a source occurrence across pipeline stages and graph
writes. It must not be derived from the current flattened position.

Use the UUID for:

- Neo4j `Node.uuid`;
- graph edges that point to source nodes;
- persisted provenance;
- durable evidence references.

## Scope

This migration applies to:

- `src/kms/core/models.py`;
- `src/kms/core/identity.py`;
- `src/kms/core/walker.py`;
- construction modules that currently read or write `Node.id`;
- graph node and edge builders;
- bundle validation;
- tests covering flattening, splitting, windows, extraction, and graph writes.

Do not change unrelated LLM prompts, model routing, enrichment algorithms, or hub
clustering behavior.

Read `docs/STYLE.md` before editing Python. Follow the repository's existing
module-qualified import style and run Ruff after edits.

## Target data model

Change `models.Node` to contain a nullable durable UUID and no mutable stream ID:

```python
@dataclass(slots=True)
class Node:
    type: NodeType | None = None
    content: str | None = None
    index: int = 0
    uuid: str | None = None
    document_index: int | None = None
    image_path: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    governing_instruction_uuids: list[str] = field(default_factory=list)
```

Field meanings:

- `index`: provider-local block index. It is not the flattened position.
- `uuid`: durable source-occurrence identity. It is not a position.
- `document_index`: owning page/document index.
- current stream position: `enumerate(nodes)` only.

Remove `Node.id`. Do not replace it with `Node.position`; the position belongs to
the list, not to the node.

## Transient references

During construction, use positions into the current canonical node list.
Rename reference fields so their temporary nature is explicit:

```python
Instruction.member_positions: list[int]
Statement.member_positions: list[int]
Procedure.member_positions: list[int]
Triplet.evidence_positions: list[int]
```

Do not store direct `Node` objects inside these models. Do not use UUIDs for
transient window traversal.

The position references are valid only for the current finalized node list. Any
stage that changes the node list must rebuild later semantic references.

## Required implementation sequence

Implement the migration in this order. Keep the repository testable after each
step.

### Step 1: Change the canonical node model

1. Add `uuid: str | None` to `models.Node`.
2. Remove `id` from `models.Node`.
3. Update constructors and fixtures that pass `id=`.
4. Update `flatten_documents()` so it only flattens and assigns
   `document_index`; it must not assign a node ID.
5. Update splitter rebuilds so they do not renumber nodes.
6. Preserve UUIDs when a stage only changes content or order.

At this step, use `enumerate(nodes)` wherever code needs a position.

### Step 2: Add deterministic UUID assignment

Add an identity helper that assigns UUIDs after the source key is known.

For an original OCR/source node, derive the UUID from stable source provenance,
not flattened position. Use the source key, document index, and provider-local
node index. The formula must distinguish two nodes from the same document.

For example, the identity inputs may be:

```text
source key + document index + provider-local node index
```

If a provider-local index is not unique, include the provider occurrence data in
`Node.provenance` and use that data in the formula.

The helper must:

- reject an empty source key;
- assign a UUID only when one is missing;
- preserve an existing UUID;
- produce the same UUID when called again with the same source provenance;
- never use the flattened list position.

Assign UUIDs after OCR has set `source.key` and before graph projection. It is
acceptable for UUIDs to remain `None` during early construction tests, but the
final projection boundary must fail if any node is missing a UUID.

### Step 3: Define transformation lineage

Every structural transformation must follow these rules.

#### Correction and formatting

These change content only. Preserve the node UUID.

#### Reordering

Move the same node objects or preserve their UUIDs. Reordering changes list
positions but not UUIDs.

#### Seam merge

When two nodes are merged, preserve the UUID of the surviving node. Add the
removed node's source information to the surviving node's provenance if the
existing seam implementation supports provenance accumulation. Do not derive a
new UUID from the new position.

#### Exercise split

A split produces multiple child nodes. Derive child UUIDs from the original
UUID and the stable child ordinal, for example:

```text
parent UUID + split-child ordinal
```

Store lineage in provenance if needed. Do not assign child UUIDs from their new
list positions.

#### Flattening

Flattening only creates the ordered list. It must never change an existing node
UUID.

### Step 4: Migrate walker APIs

The walker must operate on list positions.

Use this pattern:

```python
for position, node in enumerate(nodes):
    ...
```

Window behavior:

- `WindowNode.position` is the local position inside the returned window.
- A window must not require a global node ID.
- Convert a model's local window position to a stream position using the
  caller's window start.
- Use stream positions to select nodes from the canonical list.
- Mark target nodes by stream position or local position, never by `Node.id`.

Replace helpers such as `position_for_id()` with position validation or direct
list indexing. Do not add a new node-ID lookup table for the transient pipeline.

### Step 5: Migrate finders and builders

Update each module that currently reads `node.id`:

- `instruction_finder.py`;
- `pedagogical_component_finder.py`;
- `statement_procedure_builder.py`;
- `governance_walker.py`;
- `splitter.py`;
- `triplet_extractor.py`;
- `composition.py`;
- enrichment helpers that consume evidence references.

Required behavior:

- finder spans contain local window positions while inside a window;
- callers convert those positions to current stream positions;
- emitted instruction, statement, and procedure references are position lists;
- triplet extraction emits evidence positions;
- all position values are validated against `len(nodes)`;
- no module reads or writes `node.id`.

Update state and bundle annotations to use the renamed position fields.

### Step 6: Validate the bundle by position

Change `validate_bundle()` so it validates:

- every position is an integer;
- every position is within `0 <= position < len(bundle.nodes)`;
- positions are unique within each member/evidence list where required;
- every required node UUID is present before persistence;
- node UUIDs are unique when present;
- statement, procedure, and instruction UUID contracts remain unchanged.

Do not require UUIDs during early construction stages. Require them at the final
projection boundary with `require_identities=True`.

### Step 7: Resolve positions to UUIDs at graph projection

The graph boundary is the only place where transient positions become durable
node references.

Build the final ordered node list and assign/validate node UUIDs before writing.
For every reference:

```python
node = bundle.nodes[position]
node_uuid = node.uuid
```

Resolve:

- statement member positions;
- procedure member positions;
- instruction member positions;
- triplet evidence positions;
- entity and predicate provenance positions.

Graph edge builders must receive resolved UUIDs or nodes. They must not call a
UUID formula with a list position.

### Step 8: Persist node order separately from node identity

Node rows must contain separate properties:

```python
{
    'uuid': node.uuid,
    'position': position,
    'document_index': node.document_index,
    'type': node.type,
    'content': node.content,
}
```

Merge Neo4j nodes by `uuid`.

Build the ordered chain from the final list:

```python
chain = [node.uuid for node in bundle.nodes]
```

Build `HEAD` and `NEXT` from adjacent UUIDs in that list. The list determines
current order; UUIDs identify the vertices.

Do not derive `Node.uuid` from `position`.

## What must not be done

- Do not keep `Node.id` as a hidden alias.
- Do not add `Node.position` as another mutable identity field.
- Do not store `next_node` pointers on nodes.
- Do not use list positions to generate UUIDs.
- Do not store direct node objects inside Statements, Procedures, Instructions,
  or Triplets.
- Do not build a second parallel legacy path after the migration.
- Do not modify unrelated prompts or graph algorithms.

## Tests to add or update

### Model and flattening

- A node can carry a UUID independently of its list position.
- Flattening returns nodes in order and does not assign IDs.
- Flattening preserves UUIDs across repeated calls.
- Reordering nodes changes their positions but not UUIDs.
- UUID assignment is deterministic from source provenance.
- Two nodes with different source provenance receive different UUIDs.

### Transformations

- Correction and formatting preserve UUIDs.
- Seam merge preserves the surviving UUID.
- Exercise split creates deterministic child UUIDs from parent lineage.
- Split child UUIDs do not depend on output list position.

### Walker and construction

- Windows use local positions.
- Finder spans resolve to current list positions.
- Governance marks statement positions correctly.
- Missing or out-of-range positions fail fast.
- Triplet evidence references resolve to the expected node.

### Graph

- Node rows merge using `node.uuid`.
- Node rows persist current `position` separately.
- `HEAD` and `NEXT` use adjacent list order.
- Statement, procedure, instruction, and evidence edges use resolved UUIDs.
- Final projection rejects missing or duplicate node UUIDs.

## Acceptance criteria

The migration is complete only when all of the following are true:

1. `models.Node` has `uuid` and has no `id` field.
2. No production source code reads or writes `node.id`.
3. No flattening or splitting function assigns node IDs.
4. The ordered `list[Node]` is the only in-memory source of node order.
5. Transient semantic references are explicitly named as positions.
6. UUIDs are never derived from flattened positions.
7. Graph nodes merge by durable UUID.
8. Graph `position` is separate from graph `uuid`.
9. `HEAD` and `NEXT` are derived from adjacent list entries.
10. The full test suite passes.
11. Ruff format and Ruff check pass for all changed Python files.

If a change cannot satisfy one of these criteria without touching an unrelated
subsystem, stop and report the blocking dependency instead of inventing a
compatibility alias.
