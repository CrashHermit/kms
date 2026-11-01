# Teaching, knowledge, and practice roadmap

Status: durable teaching sessions, local-first evidence retrieval, staged source
publication, append-only learner observations, bounded teaching graphs, and
per-response knowledge admission are implemented. Practice, a verified
interactive UI, and live provider-quality evaluation remain. External research
is disabled by default.

## Direction

Start a teaching session from a learning goal, not from a source. A source,
exercise, paper, URL, or selected passage may be supplied, but is not required.
When evidence is needed, the teacher searches owned sources and canonical
propositions first, then can explicitly expand a local outcome through external
research. Retrieval is never a compulsory pre-teaching pass.

Keep three responsibilities separate:

- **Source evidence:** searchable original blocks, images, and optional
  pedagogical annotations. Importing a source does not need to populate the
  canonical knowledge graph.
- **Canonical knowledge:** precise propositions, reasoning chains, identity
  resolution, and provenance across admitted material.
- **Learner evidence:** teaching sessions, questions, responses, practice attempts,
  hints, and feedback. A proposition's existence does not demonstrate mastery.

Use `teaching`, `practice`, and `knowledge` as domain package names. Keep `Source`
for artifacts representing particular ingested material. Existing `learning`
services continue to own decks, cards, and FSRS review.

## Ordered roadmap

### 1. Durable teaching sessions

Implemented. Typed session, turn, assessment, plan, and question contracts persist
history and current session state through database-only application operations.
Sessions start with a goal and no required source relationship. Waiting for a
learner does not retain an invocation, database session, or model resource.
Detailed design: [Teaching sessions and bounded teaching workflows](TEACHING_SESSIONS.md).

Step-1 completion evidence: an actual application/database exercise can start,
persist, reload, respond to, and end a session without inference resources. This
criterion covers the persistence foundation; teacher behavior is tracked under
step 4.

### 2. Local retrieval and evidence reading

Implemented. User-scoped natural-language retrieval uses existing embedding and
reranking clients. Separate operations search and read source blocks/images,
propositions/reasoning steps, and source statements/procedures. Session-workspace
positions are kept separate from durable UUIDs. Discovery previews are not
inspection receipts: only read, active evidence can be cited or used as direct
source support.

In the single-user deployment, unfiltered source-block, statement, procedure,
and proposition retrieval uses Neo4j vector indexes for approximate candidates,
then applies ownership checks and reranking. Searches restricted by source or
exercise type use exact owner-scoped cosine ranking before limiting candidates,
so post-filtering an approximate index cannot discard the best eligible hits.
Indexed retrieval may return fewer candidates when other users' records share
the same database; it does not provide equivalent multi-user recall.

Local retrieval APIs are available independently and in the bounded teaching
interaction. A local acquisition produces ranked source-block and proposition
candidates separately. The teaching phase may inspect those results and then
respond, or explicitly expand that same outcome through external research.

Completion evidence: retrieve and inspect each material type for a learning goal,
with no access to another user's sources and only inspected items cited.

### 3. External discovery and source admission

Implemented. Optional Tavily-backed research discovers and reads web pages,
images, and selected PDF pages. The shared `SourceProcessingGraph` finalizes
stable processed identities and durable file paths, then `SourceAdmissionGraph`
returns newly captured material as a staged snapshot rather than publishing it
immediately. The interaction can inspect staged blocks/images before commit.

At the end of a successful teacher turn, the session transaction publishes
staged sources cited by the response or used as direct knowledge support.
Inspected but unused captures remain private in the durable session workspace.
Failed or stale commits do not overwrite the prior workspace or publish the
source or turn. No new `ExternalEvidence` nodes or citation relationships are
created.

Legacy `CITES_EXTERNAL` records are still migrated explicitly per user with
`migrate_research_sources`. After every owner is migrated, the separate
`finalize_research_source_migration` operation verifies that no legacy citation
remains, removes orphan legacy evidence, and drops the old uniqueness constraint.

Completion evidence: retrieve and inspect a research source, reload its local
source-block citations, and prove global finalization refuses while legacy
citations remain.

### 4. Teaching inference, tool orchestration, and interactive UI

The provider-neutral DSPy prober, planner, and teacher use typed instructional
operations in a bounded interaction subgraph. A fixed, acyclic parent graph
retrieves learner history once, runs the interaction once, prepares observations
and knowledge, and commits one teacher response. Each tool action invokes a
one-action evidence graph; research and source preparation retain independent
bounded graphs. No outer restart/while loop or persistent graph process is used.

The workspace distinguishes local discovery from inspected receipts and active
evidence. Each successful turn atomically saves the workspace, including
unpublished staged captures, for the next invocation. Original image bytes
remain in source files. Historical learner observations remain private to their
owner and are separate from canonical subject knowledge.

Still to implement: a verified interactive UI calling the existing start, load,
submit, teacher-turn, citation-read, and end operations. Inspect actual
citations, images, explanations, and identity decisions with configured providers
before claiming teaching quality.

Completion evidence: a source-grounded session survives stop/resume while
awaiting an answer, continues appropriately from that answer, and leaves
reusable proposition and evidence provenance. Deterministic tests do not
establish provider quality.

### 5. Canonical knowledge admission

Implemented at the teaching boundary. After a final response is selected, any
response marked `instruction` can be converted into self-contained propositions
and joint-premise reasoning steps, regardless of whether the persisted phase was
probing, planning, or teaching. A `coordination` response skips extraction.
Support is selected only from the final active inspected source/image packet and
explicitly read propositions.

Identity resolution searches the user's existing propositions, then resolves
equivalent current-draft claims before persistence. It reuses one canonical
proposition while preserving per-turn assertions and alternative reasoning
chains. Extraction does not run once per probe/tool and does not create a
session-wide synthetic source.

Assertions link the teaching turn, proposition, direct source-block support, and
derived reasoning steps. Reasoning steps link premise propositions, conclusion
propositions, and direct source-block support. Source and knowledge revisions,
staged source publication, learner observations, and the teacher turn share the
same transaction. Identity matching is not truth verification; original source
evidence remains unchanged, and importing a source does not exhaustively extract
propositions.

Completion evidence: separate materials support one canonical claim without
destroying either evidence record; distinct technical claims remain distinct;
joint premises, alternative arguments, source ownership, and atomic stale-write
rejection survive real-driver verification.

### 6. Adaptive practice

Build a separate persistent practice workflow over admitted propositions and
reasoning chains. Prefer suitable existing source exercises when available;
generate adaptations or new problems when needed. Support recall,
discrimination, and composition.

Commit the solution before posing the problem. Retain attempts, graduated hints,
feedback, and retest decisions. Do not infer mastery from lesson navigation or
silently translate practice outcomes into FSRS ratings. Initially, practice does
not admit new subject propositions.

Completion evidence: a learner can attempt, request hints, conclude, resume, and
receive a fresh-surface retest while the original committed solution and attempt
history remain unchanged.

### 7. Retire exhaustive source proposition extraction

Completed. Source ingestion no longer creates source propositions, proposition
embeddings, proposition hubs, or proposition-specific semantic-stage machinery.
Source Semantic retains source blocks, entities, statements, procedures, and
their supported local indexes. Canonical proposition admission remains a
teaching/knowledge-boundary responsibility rather than exhaustive ingestion.

Source-learning cards now consume source-block material directly. Atomic requests
retain exact owning block provenance, coherent requests group material by source
page only for bounded inference, and persisted cards use direct
`SUPPORTED_BY` edges to `SourceBlock`. Existing card and review workflows remain
separate from canonical knowledge admission; no incidental user-graph migration
was performed.

Completion evidence: source ingestion works without source proposition
generation, source-learning cards retain direct source evidence, and no obsolete
source-claim consumer or compatibility shim remains.

## Architecture constraints

- Keep the existing six-layer ownership: TUI interaction, application operations
  and resource lifetimes, composition wiring, service transitions, repository
  persistence, and core domain/transport contracts.
- Use the existing DSPy predictor factory, explicit examples, profile resolution,
  token budgeting, shared clients, and configured concurrency. Do not add a
  second provider/authentication boundary or executor.
- Repositories alone own Cypher and database sessions. Never hold a transaction
  open while waiting for model inference, network tools, or a human response.
- The database owns authoritative teaching history. LangGraph working state is
  rebuilt from it, not an independently authoritative checkpoint conversation.
- Inputs and model outputs are valid and perfect. Do not add output repair,
  fallback content, missing-reference recovery, retries, or defensive coverage
  gates. Real operational failures and concurrent writes are different concerns
  and must preserve the persisted session's integrity.
- Plan files describe intended behavior. They do not supersede documentation of
  the current implementation in [Source semantics](SOURCE_SEMANTICS.md).

## References

Behavioral reference: LearnHarness at revision
[`dc4c398d5167865b84b0ccb3ec809eb88bf8f65f`](https://github.com/amosblomqvist/learnharness/tree/dc4c398d5167865b84b0ccb3ec809eb88bf8f65f).
Use implementation rather than assuming its development guide describes current
practice. Review its AGPL-3.0 license before copying code or prompt text.

Existing KMS2 boundaries:
[application](../src/kms2/application.py),
[service ownership](../src/kms2/service/__init__.py),
[service composition](../src/kms2/composition/services.py), and
[Python style](STYLE.md).
