# Plan: Build the Concept Layer

## Context

The concept layer is the only connective tissue between blocks (block-to-block relations are gone). It induces open, multi-granularity concept phrases (`linear algebra`, `group theory`, `abstract algebra`) for every `:Entity`, `:Procedure`, and `:Act`, and persists them as `:Concept` nodes with `:INSTANCE_OF` edges (phrase on the edge). This is AutoSchemaKG's schema induction (§B.2) adapted to KMS.

Currently **dark** — `graph/concepts.py` has identity primitives only (`concept_uuid`, `concept_batch`), and `schema.py` already declares the `:Concept` uniqueness constraint. Nothing writes through it.

## Design

### Where does it run in the pipeline?

```
… → instruction_distributor → conceptualizer → entity_persister → concept_persister → END
                                        │                          │
                                  (generates phrases,       (writes :Concept nodes +
                                   builds adjacency          :INSTANCE_OF edges)
                                   from state for context)
```

The conceptualizer is an **entity-stage** LangGraph node (has access to fully-formed entities, procedures, acts). The concept persister is a **graph-stage** node (writes to Neo4j). This cleanly separates "LLM work" from "persistence work," matching the existing pattern (entity stages are persistence-agnostic; persisters are I/O).

### State channel

New channel in `state.py`:

```python
concept_mappings: list[
    models.ConceptMapping
]  # induced phrases per graph element
```

A `ConceptMapping` is a new dataclass in `models.py`:

```python
@dataclass(slots=True)
class ConceptMapping:
    """One element → its induced concept phrases."""
    kind: str          # "Entity" | "Procedure" | "Act"
    source: str        # book identity
    entity_id: int     # entity id (for computing UUIDs of all three kinds)
    procedure_index: int | None = None  # set for Procedure/Act
    step_index: int | None = None       # set for Act only
    phrases: list[str] = field(default_factory=list)
```

The `entity_id`/`procedure_index`/`step_index` tuple maps directly to the existing `entity_uuid`, `procedure_uuid`, `act_uuid` in the graph tier — so the persister can construct the source element's vertex key without importing the entity-stage models.

---

## Files to create

### 1. `src/kms/entity/conceptualizer.py` — the LLM stage

**DSPy Signature (`Conceptualize`):** Adapt AutoSchemaKG's prompts (their Figs 5/6/7) for three kinds:

- **Entity conceptualization** (with context from member nodes + its procedures' content):
  ```
  Given an ENTITY and its CONTEXT, generate 3-5 abstract concept phrases
  at varying levels of specificity → general.
  ```
- **Procedure conceptualization** (with context from parent entity + its acts):
  Same shape, event-style prompt.
- **Act conceptualization** (with context from parent procedure + grandparent entity):
  Same shape, event-style prompt.

**DSPy Module (`Module`):** One `dspy.ChainOfThought(Conceptualize)` — the prompt accepts a `kind` field so one module handles all three. (Or three separate predict calls if one prompt proves too noisy — start with one.)

**Entry point `conceptualize(entities, nodes_by_id)` (async):**
1. For each entity: build context string from its member nodes + its procedures' labels/contents. Call LLM → get phrases. Emit `ConceptMapping(kind="Entity", ...)`.
2. For each procedure within each entity: build context from parent entity's type + its own acts. Call LLM → get phrases. Emit `ConceptMapping(kind="Procedure", ...)`.
3. For each act within each procedure: build context from parent procedure's content + grandparent entity. Call LLM → get phrases. Emit `ConceptMapping(kind="Act", ...)`.
4. Return list of `ConceptMapping`.

Context construction (matching AutoSchemaKG §B.2.2):
- **Entity**: sample up to N_ctx neighbors — its first few member nodes' content + its procedures' first lines. Concatenate as `"members: ... | procedures: ..."`
- **Procedure**: parent entity's `type` and `label` + first act's text → `"entity: {type} {label} | steps: ..."`
- **Act**: parent procedure's first 200 chars + grandparent entity's type → `"procedure: ... | entity: ..."`

**LangGraph node (`ConceptualizerNode`):**
```python
class ConceptualizerNode:
    async def run(self, state: state.State) -> dict:
        mappings = await conceptualize(
            state.get('entities', []),
            {n.id: n for n in state.get('nodes', [])},
            module=self.module,
        )
        return {'concept_mappings': mappings}
```

### 2. `src/kms/core/models.py` — add `ConceptMapping` dataclass

```python
@dataclass(slots=True)
class ConceptMapping:
    kind: str
    source: str
    entity_id: int
    procedure_index: int | None = None
    step_index: int | None = None
    phrases: list[str] = field(default_factory=list)
```

### 3. `src/kms/core/state.py` — add `concept_mappings` channel

```python
concept_mappings: list[
    models.ConceptMapping
]  # induced phrases, one per element, written by conceptualizer
```

Plain list (no reducer) — written once by the sequential conceptualizer node.

---

## Files to modify

### 4. `src/kms/graph/concepts.py` — add edge builders

Currently has: `CONCEPT_LABEL`, `normalize_concept`, `concept_uuid`, `concept_properties`, `concept_batch`.

Add:

```python
INST_OF_LABEL = 'INSTANCE_OF'

def inst_of_pairs(
    mappings: list[models.ConceptMapping], source: str
) -> list[dict]:
    """Build :INSTANCE_OF edge pairs from concept mappings.
    
    Each mapping produces one edge per phrase:
      (element_node)-[:INSTANCE_OF {phrase}]->(:Concept)
    
    The element node is Entity, Procedure, or Act (identified by its kind+uuids).
    """
    pairs = []
    for m in mappings:
        if not m.phrases:
            continue
        element_uuid = _element_uuid(m, source)
        for phrase in m.phrases:
            pairs.append({
                'element': element_uuid,
                'element_label': m.kind,  # "Entity", "Procedure", "Act"
                'concept': concept_uuid(phrase),
                'phrase': phrase,
            })
    return pairs

def _element_uuid(m: models.ConceptMapping, source: str) -> str:
    """Resolve a ConceptMapping to its target vertex uuid."""
    from kms.graph.entities import entity_uuid
    from kms.graph.procedures import act_uuid, procedure_uuid
    
    if m.kind == 'Entity':
        return entity_uuid(source, m.entity_id)
    elif m.kind == 'Procedure':
        return procedure_uuid(source, m.entity_id, m.procedure_index)
    elif m.kind == 'Act':
        return act_uuid(source, m.entity_id, m.procedure_index, m.step_index)
    raise ValueError(f'Unknown kind: {m.kind}')
```

### 5. `src/kms/graph/writer.py` — add `persist_conceptualization`

```python
async def persist_conceptualization(
    mappings: list[models.ConceptMapping], source: str
) -> None:
    """Upsert :Concept nodes and :INSTANCE_OF edges from induced mappings."""
    if not mappings:
        return
    
    # Collect unique concept names → batch MERGE :Concept nodes
    all_phrases = list({p for m in mappings for p in (m.phrases or [])})
    concept_rows = concepts.concept_batch(all_phrases)
    
    # Build INSTANCE_OF edge pairs
    pairs = concepts.inst_of_pairs(mappings, source)
    
    async with driver().session(database=database()) as session:
        # MERGE :Concept nodes (global, no source prefix)
        if concept_rows:
            await session.run(
                f'UNWIND $rows AS row MERGE (c:{concepts.CONCEPT_LABEL} {{uuid: row.uuid}}) SET c += row',
                rows=concept_rows,
            )
        
        # MERGE :INSTANCE_OF edges with phrase property
        # Each pair targets a different label (Entity/Procedure/Act), so we need
        # a generic MATCH that resolves by uuid across all three labels
        if pairs:
            await session.run(
                f'UNWIND $pairs AS pair '
                f'MATCH (e {{uuid: pair.element}}), (c:{concepts.CONCEPT_LABEL} {{uuid: pair.concept}}) '
                f'MERGE (e)-[r:{concepts.INST_OF_LABEL}]->(c) '
                f'SET r.phrase = pair.phrase',
                pairs=pairs,
            )
```

Note: `MATCH (e {uuid: pair.element})` without a label scans all nodes, but with a uuid uniqueness constraint this is a point lookup — fast, and avoids needing to dispatch by element label in Cypher.

### 6. `src/kms/graph/persister.py` — add `ConceptPersisterNode`

```python
class ConceptPersisterNode:
    """Persist the induced concept phrases as :Concept nodes and :INSTANCE_OF edges."""

    async def run(self, state: state.State) -> dict:
        source = state.get('source')
        if not db.is_configured() or not source:
            return {}
        await schema.ensure_schema()  # idempotent, safe
        await writer.persist_conceptualization(
            state.get('concept_mappings', []), source
        )
        return {}
```

### 7. `src/kms/pipeline.py` — wire the new stages

```python
# In build_graph():
conceptualizer = ConceptualizerNode()
concept_persister = ConceptPersisterNode()

graph.add_node('conceptualizer', conceptualizer.run)
graph.add_node('concept_persister', concept_persister.run)

# After instruction_distributor, before entity_persister:
# (conceptualizer needs entities fully attributed; entity_persister also needs concepts)
graph.add_edge('instruction_distributor', 'conceptualizer')
# conceptualizer → entity_persister → concept_persister → END
graph.add_edge('conceptualizer', 'entity_persister')
graph.add_edge('entity_persister', 'concept_persister')
graph.add_edge('concept_persister', END)
```

This order means:
- `conceptualizer` sees fully attributed entities (procedures, acts, instructions all done)
- `entity_persister` writes entities/procedures/acts first (so the `MATCH (e {uuid:...})` in concept persistence has vertices to attach to)
- `concept_persister` writes concepts and INSTANCE_OF edges last

### 8. `src/kms/graph/procedures.py` — small export

Export `act_uuid` and `procedure_uuid` from the module top-level so `concepts.py` can import them without circular dependencies. (Currently they're importable, just double-check the import paths are clean — `concepts.py` already imports nothing from `entities.py` or `procedures.py`, so adding these imports should be safe.)

---

## What about fusion?

Fusion (cross-book concept dedup) is **not** part of this plan. As `CONCEPT-LAYER.md` states:

> *"Fusion makes concept identity discovered (embedding + judge) rather than deterministic — a real change to the persistence contract for this one layer, and the only part of the system needing a corpus-wide execution mode."*

Fusion is a separate post-processing pass that:
1. Reads existing `:Concept` nodes from Neo4j
2. Embeds them and runs a conservative LLM judge to decide merges
3. Rewrites `:INSTANCE_OF` edges to point at surviving hubs

It runs **after** multiple books are in Neo4j, not inside the per-book pipeline. Nothing in this plan blocks it — the `phrase` property on every `:INSTANCE_OF` edge preserves what was originally extracted, so a bad merge can be undone by re-pointing (as the design doc requires).

---

## Reuse summary

| Pattern | Used from | For |
|---|---|---|
| DSPy Signature + Module + aforward | `entity/statement_extractor.py` | LLM conceptualization |
| Context construction from neighbors | AutoSchemaKG §B.2.2, adapted | Entity context for LLM |
| Entity-stage LangGraph node pattern | `entity/pedagogical_component_finder.py` etc. | `ConceptualizerNode` |
| Graph mapping functions | `graph/entities.py`, `graph/procedures.py` | `concepts.py` edge builders |
| Writer + session.run pattern | `graph/writer.py` | `persist_conceptualization` |
| Persister gating on `is_configured()` | `graph/persister.py` | `ConceptPersisterNode` |
| Concept identity + batch dedup | `graph/concepts.py` (existing) | Reused as-is |
| Schema constraint | `graph/schema.py` (already declared) | Reused as-is |

---

## Steps

- [ ] Add `ConceptMapping` dataclass to `core/models.py`
- [ ] Add `concept_mappings` channel to `core/state.py`
- [ ] Create `entity/conceptualizer.py` with DSPy Signature, Module, and conceptualization logic
- [ ] Expand `graph/concepts.py` with edge builders (`inst_of_pairs`, `_element_uuid`)
- [ ] Add `persist_conceptualization` to `graph/writer.py`
- [ ] Add `ConceptPersisterNode` to `graph/persister.py`
- [ ] Wire `conceptualizer` and `concept_persister` into `pipeline.py`
- [ ] Update imports/exports as needed (e.g., `procedure_uuid`/`act_uuid` export from `graph/procedures.py`)

---

## Verification

1. **Unit test the conceptualizer DSPy module** — mock the LM, verify the Signature accepts the expected fields and returns phrase lists (follow pattern from `tests/` entity tests).
2. **Unit test context construction** — verify neighbor sampling doesn't blow up on edge cases (entity with no members, procedure with no acts, etc.).
3. **Unit test `inst_of_pairs`** — verify the correct uuids and edge properties are generated for each kind.
4. **Unit test `persist_conceptualization`** — verify the Cypher is well-formed (can use the existing Neo4j opt-in test pattern).
5. **Run the full pipeline on a fixture PDF** with Neo4j disabled — verify the conceptualizer runs and produces `concept_mappings` in state (no crash).
6. **Run the full pipeline with Neo4j enabled** (`KMS_NEO4J_IT=1`) — verify `:Concept` nodes and `:INSTANCE_OF` edges appear in the database with the right properties.
