# Teaching, knowledge, and practice roadmap

Status: roadmap step 1 is implemented; later teaching, retrieval, knowledge, and
practice workflows remain design. Existing source pipelines and card review remain
unchanged until their explicit cutovers.

## Direction

Start a teaching session from a learning goal, not from a source. A source,
exercise, paper, URL, or selected passage may be supplied, but is not required.
The teacher can retrieve local or external material whenever the interaction
needs evidence; retrieval is not a compulsory pre-teaching pass.

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

Define typed session, turn, assessment, plan, and question contracts; persist
history and current session state; expose database-only application operations.
Sessions start with a goal and no required source relationship. Waiting for a
learner must not retain an invocation, database session, or model resource.

Detailed design: [Teaching session foundation](TEACHING_SESSIONS.md).

Completion evidence: an actual application/database exercise can start, persist,
reload, respond to, and end a session without starting inference resources. The
full teacher is not claimed to work at this checkpoint.

### 2. Local retrieval and evidence reading

Add user-scoped natural-language search across eligible source material, using
existing embedding and reranking clients. Search must work from a query string,
not require an existing block as the query. Provide separate reading operations
for full passages, neighboring blocks, and original image assets.

Selecting material can prioritize or explicitly constrain a session's evidence;
it is not an implicit requirement or a mandatory restriction to one source.
Preserve that distinction in the retrieval contracts.

Completion evidence: retrieve and inspect relevant text and image evidence for a
learning goal, with no proposition admission and no access to another user's
sources.

### 3. External discovery, reading, and evidence retention

Provide web search and page/document reading. Search snippets are discovery
material, not automatically sufficient support for an explanation. Retain the
URL, retrieval time, and relevant content actually used so citations can be
inspected later. Do not require full-site ingestion merely to use one passage.

Connect teaching turns to the evidence they use. Local references reuse existing
blocks/assets; external evidence is retained without pretending it was already
processed through the source pipeline. Retrieval alone writes no propositions.

Completion evidence: discover external material, read it, retain the used
passage, and reload a turn's citations after restarting the application.

### 4. Teaching inference, tool orchestration, and interactive UI

Add probing, planning, and teaching modules with explicit demonstrations. Follow
predict, explain, check, and repair where appropriate; do not force artificial
prediction questions or add a separate boolean gate for every decision.

A bounded orchestration loop executes typed library/web search and reading
requests and supplies results back to inference. Modules remain provider-neutral;
nodes coordinate tools and graph phases. Plan and assessment can change as new
learner evidence arrives. A pending learner interaction ends the current graph
invocation; the next message starts another invocation from persisted state.

Provide UI operations to start from a goal, read cited explanations, answer,
resume, and end a session. The teacher need not search on every turn, but must not
present model recall as if it were inspected source evidence.

Completion evidence: a real source-grounded session exercises both library and
external retrieval paths, survives a stop/resume while awaiting an answer, and
continues appropriately from that answer. Inspect actual citations, images, and
explanations; deterministic tests alone cannot establish teaching quality.

### 5. Canonical knowledge admission

Extract substantive propositions and joint-premise reasoning chains from
learner-facing teaching explanations or explicitly selected material. Do not
admit learner mistakes, search results, or private model thinking as subject
facts. This operation is distinct from importing a document into the library.

Resolve proposition identity against the user's existing knowledge space before
creating new canonical nodes. Equivalent claims reuse identity and accumulate
provenance; related claims with different conditions, quantifiers, or domains
remain distinct. Alternative arguments for one conclusion remain separate chains.
Identity matching is not truth verification.

Keep original evidence intact. Distinguish explicit source claims, explanations,
and derived conclusions. Link provenance structurally rather than copying
`source_uuid` onto every object. Serialize identity resolution plus admission
where necessary to prevent competing duplicate creation, without serializing all
independent extraction work.

Completion evidence: two independent materials can support one proposition
without destroying either evidence record; genuinely different technical claims
remain separate; joint premises and alternative arguments survive admission.

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

### 7. Retire source proposition extraction cleanly

The intended destination removes `SourceProposition` extraction, embeddings,
hubs, and associated source-stage machinery. Canonical knowledge admission
becomes the proposition-producing boundary instead of exhaustive ingestion.

Current source card generation consumes source proposition evidence. Migrate
that consumer and every affected caller, result contract, setting, test, and
document before deleting the old paths. Preserve existing cards and review
history; do not reuse the source-card regeneration cleanup for teaching sessions
or canonical knowledge. This dependency prevents deleting the extractor first.

Completion evidence: source ingestion works without proposition generation, card
workflows retain their supported behavior, and no obsolete proposition-stage
consumer or compatibility shim remains. Do not clear or migrate the user's
Neo4j data as an incidental part of this work.

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
