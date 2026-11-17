# Teaching, adaptive planning, and FSRS scheduling

## Context

Build three independently callable components: grounded teaching generation, a unified session planner choosing assessment/teaching/finish, and FSRS scheduling with due-review selection. Assessments are atomic: exactly one proposition, argument, or individual task solution is graded and scheduled per task. Use the established local component pattern and ordered native multimodal Material; assume perfect domain/model inputs and outputs, with no repair, retries, compatibility aliases, or migrations. Do not wire the permanent coordinator or UI in this change.

## Confirmed starting points

- `src/kms/learning/session/models.py` currently defines `ProbeTarget(kind='probe', understanding, explanation, assessment_uuids)`, `ProbeFinished(kind='finish', summary, assessment_uuids)`, `ProbeDecisionRecord`, `SessionStep = Task | ProbeDecisionRecord`, and task-only session history/context. `SessionMessage` is still the goal-clarification conversation contract.
- `src/kms/learning/task/models.py` already owns `Task`, `Solution`, `GroundingReferences`, `KnowledgeReferences`, native-material `FreeAnswer`/`UnknownTaskAnswer`, `TaskResponse`, and `TaskAssessment`. `TaskGrade` separates correctness, reasoning interpretation, and the review rating `again|hard|good|easy`.
- Shared learning example images now live in `src/kms/learning/examples/`; the prior plan at `local://learning-components-plan.md` describes the completed, superseded multiple-choice probe architecture and is reference only.

- User confirmed the three-component scope: teaching generation, expanded session planner, and FSRS scheduling/review selection; the permanent coordinator is deferred.
- User clarified that assessments should be atomic. Replace task target lists with one typed, caller-supplied target; supporting knowledge/evidence can still contain many objects. The scheduling plan must not introduce multi-target grade distribution.
- Read-only scouts confirmed that task generation currently owns `ImageReference`, `MaterialDraft`, and `_material`; teaching will become the second output consumer, requiring a shared learning-owned material-draft boundary rather than a duplicate.
- Read-only scouts confirmed that session persistence currently dispatches only task/decision step kinds and that `append_decision` neither finalizes native media nor writes grounding. Teaching needs its own atomic append, plus explicit history-kind hydration.
- No FSRS implementation or dependency exists in the inspected source, tests, `pyproject.toml`, or `uv.lock`. Task grading already persists multiple assessments for one response, so scheduling must select one assessment per actual attempt rather than treating every regrade as another review.

- The user explicitly included individual task solutions as schedulable targets. A solution review revisits that specific problem and reference answer, not an inferred reusable procedure.
- PyPI and the upstream tags identify `fsrs` 6.3.3 as a published pure-Python release requiring Python >=3.10. The upstream Card supports `to_json`/`from_json`; Scheduler supports explicit UTC `review_datetime` and `enable_fuzzing=False`. Its default Card constructor sleeps to create an integer ID, so supply a deterministic ID rather than using that default.
- `src/kms/knowledge/preparation/graph.py::KnowledgePreparationGraph.prepare(user_uuid: str, understanding: str) -> PreparationResult` already searches owned knowledge, judges sufficiency, reasons/commits, and extracts from owned source hits until sufficient or exhausted. It returns context and the final judgment; it does not acquire external sources. Reuse this boundary when the permanent learning coordinator is implemented, rather than adding another preparation workflow inside teaching or session planning.

## Approach

Follow the responsibility split demonstrated by `source/processing/formatting/{node,inference,composition}.py`: nodes coordinate work and keep operation-specific transformations local; inference contains model calls, not workflow persistence; composition constructs injected dependencies. Do not introduce a parallel service layer or force algorithm-only memory operations through DSPy. Unlike `SourceProcessingGraph.run`, which returns an in-memory snapshot, learning nodes must publish their interactive step/review before returning; keep this intentional lifecycle difference. The permanent graph will only connect these nodes, not absorb their behavior.

### 1. Make the assessment target atomic and backend-owned

Create `src/kms/learning/references.py` for shared Pydantic reference contracts (`extra='forbid'`):

- `PropositionTarget(kind: Literal['proposition']='proposition', uuid: str)`.
- `ArgumentTarget(kind: Literal['argument']='argument', uuid: str)`.
- `SolutionTarget(kind: Literal['solution']='solution', uuid: str)`.
- `LearningTarget`: their discriminated union on `kind`.
- Move existing `GroundingReferences` here unchanged: supporting proposition/argument/unit UUID lists remain plural and set-like.

Remove `KnowledgeReferences` and replace `Task.targets` with required `Task.target: LearningTarget` in `learning/task/models.py`. Keep `Solution` as the hidden, task-specific reference solution. Update every exact occurrence returned by `grep -R -n -E 'KnowledgeReferences|GroundingReferences|\\.targets|targets=' src/kms/learning tests/test_kms_task*.py tests/test_kms_learning_session*.py`: distinguish Task targets from session/decomposed goal `targets`, which remain lists of understanding strings. Remove foreign reference re-exports from `learning/task/__init__.py`; import shared references from their canonical owner.

In `task/generation/models.py`, add local `NewSolutionTarget(kind: Literal['new_solution']='new_solution')` and `TaskTargetRequest = Annotated[PropositionTarget | ArgumentTarget | SolutionTarget | NewSolutionTarget, Field(discriminator='kind')]`. Add `TaskGenerationRequest.target: TaskTargetRequest = Field(default_factory=NewSolutionTarget)` and `solution_task: Task | None = None`; the latter is the owned original problem and solution when target.kind is `solution`, otherwise None. Existing session, understanding, knowledge, and evidence fields stay. This explicit default preserves source-only task creation: its newly committed reference solution is the one atomic target, never a source unit.

Request preconditions: an existing Proposition/Argument target and its closed argument endpoints are present in knowledge; an existing Solution target matches solution_task.solution.uuid. NewSolutionTarget requests have solution_task=None and enough supplied knowledge/evidence for that one task. Preserve these perfect-input contracts directly, without downstream repair or duplicate runtime validators.

Remove targets from `TaskDraft`; the model does not choose or invent the assessment identity. Construct the new Solution first in the node, then copy the caller's existing target or create `SolutionTarget(uuid=new_solution.uuid)` for `new_solution`. For an existing solution target, the request supplies its original task; the generator must preserve that problem's conditions and answer while asking for its solution or explanation. Every generated task still owns its own committed reference Solution; subsequent tasks reviewing an existing Solution retain that existing target UUID instead of creating a new memory identity.

Update TaskGenerationSignature and its demonstrations to assess only that one target: a proposition's meaning/truth conditions, an argument's premise-to-conclusion justification, or the specified solution to the same problem. Additional knowledge supports the question rather than adding independently graded objectives. TaskGradingSignature continues to use the actual task's hidden solution/criteria and interprets correctness/rating for that single target; it does not distribute a grade across supporting knowledge.

Change task persistence/readback to exactly one `ASSESSES` relationship. Match Proposition/Argument through `OWNS_PROPOSITION`/`OWNS_ARGUMENT`; match Solution through the user's `OWNS_SESSION` → `HAS_STEP` → `HAS_SOLUTION` path, including completed sessions. Create the new hidden solution before matching the assessed endpoint so a task can assess its own new Solution atomically. Exclude `target`, grounding, and solution associations from payload JSON; hydrate target kind/UUID from the actual endpoint. Preserve strict owned-result checks and rollback, without adding target-count repair or malformed-output validators.

Update conceptual/diagram demonstrations to explicit proposition targets and the procedural demonstration to `NewSolutionTarget`. Source-only test fixtures build their Solution before creating a Task targeting it. Add an actual Argument target case; supporting propositions do not become extra assessment targets.

### 2. Share only the material-output boundary

Create `src/kms/learning/materials.py`; move `ImageReference` and `MaterialDraft` from task generation and move `_material` from its node as `resolve_material(draft: MaterialDraft, image_assets: Sequence[ImageAsset]) -> Material`. Preserve ordered text/image occurrences and direct image-position resolution. No equivalent shared output resolver currently exists; `inference/generation.py` remains the only native-input projector.

Update task generation's models/node/examples and native/material tests to canonical imports, removing the old declarations rather than leaving aliases. Both teaching and task nodes construct catalogs with `collect_image_assets({'request': request})`, identical to the actual inference projection. Reuse `learning/examples/image_assets.py::learning_image`; do not create another image backend or generate pixels.

### 3. Add the teaching-generation component and durable teaching steps

Create `learning/teaching/models.py` with `Teaching(Vertex)` fields `kind: Literal['teaching']='teaching'`, `understanding: str`, `objective: str`, `target: LearningTarget`, `material: Material`, `grounding: GroundingReferences`, and `assessment_uuids: list[str]`. Define `PresentedTeaching(teaching_uuid: str, material: Material)` with extra forbidden. These are domain records, separate from stage requests to avoid a session-model import cycle.

Create `learning/teaching/generation/{__init__,models,inference,examples,node,settings,composition}.py`. Requests contain `session: SessionContext`, `understanding: str`, `objective: str`, `target: LearningTarget`, `knowledge: KnowledgeContext`, `evidence: list[SourceUnitWindow]=[]`, `assessment_uuids: list[str]=[]`, and `solution_task: Task|None=None` under the same existing-solution precondition as task generation. `TeachingDraft` contains `material: MaterialDraft` and `grounding: GroundingReferences`. `TeachingState` carries user_uuid/session_uuid/request.

`TeachingGenerationSignature` takes request plus native content_images and returns `teaching: TeachingDraft`. `TeachingGenerationModule(predictor)` exposes typed forward/aforward and only invokes the injected predictor. Teach one grounded connection, example, or repair for the supplied objective; start from demonstrated understanding, preserve uncertainty, and do not claim the learner understood merely because an explanation was presented. Address cited evidence rather than inventing a misconception. For solution targets, explain that specific problem's valid solution. Use native images by exact catalog indexes, grounded only in supplied knowledge/source material. Do not generate an assessment task or search for sources inside this component.

`TeachingGenerationNode(module, repository).run(state) -> dict[str, Teaching]` reads the owned session before inference, resolves output material, copies target/objective/understanding/assessment UUIDs from the request, assigns backend identity/timestamps, and awaits `repository.append_teaching` before returning `{'teaching': persisted}`.

Add `LearningSessionRepository.append_teaching(user_uuid: str, session_uuid: str, teaching: Teaching) -> Teaching`. Finalize immutable native artifacts before the transaction; reuse the session lock and `_append_step` with `Teaching` plus `LearningSessionStep` labels. Atomically attach `TEACHES` to the single owned target, `GROUNDED_IN` to supplied owned propositions/arguments/units, and `BASED_ON` to actual task assessments from this same session. Missing/foreign references produce strict-result rollback with no moved tail or partial step. Exclude those associations and lifecycle fields from payload JSON.

Extend SessionStep and READ_SESSION/hydrate_session with explicit task/teaching/decision branches, including target references, teaching grounding, and assessment citations. Preserve stored chronology. `select_session_context` remains explicit: selecting a teaching/decision step does not silently pull in other tasks; callers must select the cited task steps when they want their responses/assessments included.

Define `TeachingGenerationSettings` with NoRetryVisionInferenceSettings, input_token_budget=16384 (gt=0), safety_margin_tokens=256 (ge=0), and extra forbidden. Add `Settings.teaching_generation` with its default factory. `generation/composition.py::build_node(settings: Settings, database: DatabaseClient, *, predictors: PredictorFactory, tokenizers: LanguageModelTokenizers) -> TeachingGenerationNode` injects the session repository and passes the same TeachingGenerationSignature/examples to budget construction and predictor creation. Demonstrations cover a conceptual connection, native-diagram repair, and a worked solution using fixed IDs/UTC timestamps. Application `generate_teaching(settings: Settings, user_uuid: str, session_uuid: str, request: TeachingGenerationRequest) -> PresentedTeaching` uses the established inference-only resource path (`embeddings_required=False`) and returns only persisted learner-facing material.

### 4. Replace probe planning with session-owned learning planning

Move the current `learning/probe/planning/` package to `learning/session/planning/`, keeping its models/inference/examples/node/settings/composition structure. Rename `ProbePlanningRequest/Signature/Module/Node/Settings` to `LearningPlanningRequest/Signature/Module/Node/Settings`, `probe_planning_examples` to `learning_planning_examples`, settings `probe_planning` to `learning_planning` (`KMS_LEARNING_PLANNING__...`), and application `plan_probe_step` to `plan_learning_step`. Rename the direct test to `tests/test_kms_learning_planning.py`. Update exactly the references returned by `rg -n 'ProbePlanning|ProbeDecision|ProbeTarget|ProbeFinished|probe_planning|plan_probe_step' src tests`; no aliases or retained old package.

In session/models.py replace the old probe decision contracts with:

- `AssessmentTarget`: kind `Literal['assess']='assess'`, understanding str, explanation str, assessment_uuids list[str], target LearningTarget|None=None.
- `TeachingTarget`: kind `Literal['teach']='teach'`, understanding str, objective str, explanation str, assessment_uuids list[str], target LearningTarget.
- `LearningFinished`: kind `Literal['finish']='finish'`, summary str, assessment_uuids list[str].
- `LearningDecision`: discriminated union of those three on kind.
- `LearningDecisionRecord(Vertex)`: kind `Literal['decision']='decision'`, decision LearningDecision.
- `SessionStep`: discriminated Task|Teaching|LearningDecisionRecord. Retain SessionMessage, LearningSession, task responses/assessments, and initial session target strings.

`LearningPlanningRequest` contains session SessionContext and optional target LearningTarget=None. A supplied target anchors a selected goal/review item; absent target means the next understanding still needs knowledge preparation outside the planner. An assessment decision copies a supplied or already-observed target when applicable; None is allowed only when selecting a new understanding without an existing knowledge identity. A teaching decision always references an existing supplied/observed target. Do not invent target UUIDs.

The signature returns `decision: LearningDecision`; the thin module and `LearningPlanningNode.run` retain read-owned-session → infer → append-decision → return-persisted ordering. `PlanningState` still carries user_uuid/session_uuid/request. Application `plan_learning_step(settings: Settings, user_uuid: str, session_uuid: str, request: LearningPlanningRequest) -> LearningDecisionRecord` uses only its own preflight/settings and inference-only resources.

Planning instructions/examples implement the unified policy: empty history starts with one atomic assessment; incorrect/unknown attempts lead to targeted teaching, as does explicitly unsound reasoning about the target. Correct answers lacking unnecessary explanation are uncertainty, not automatic failure. After teaching, assess that same target before advancing or finishing. Use the observations to choose a missing connection rather than re-teaching all source material. Correct, sufficiently supported performance can move to another relevant understanding or finish with a bounded, cited summary. No phase modes, fixed turn limits, hint/difficulty framework, deterministic right/wrong routing, or FSRS call inside the inference module.

Change `append_decision` to the new record/`LearningDecision` graph label. Store cited assessments through same-session BASED_ON. For assess/teach decisions with a target, store exactly one TARGETS edge to the owned Proposition, Argument, or Solution; omit the edge for an unresolved assessment target and for finish. Exclude target and assessment UUIDs from JSON; hydrate them from actual relationships. A finish decision still closes the session atomically with its timeline append. Keep schema identity on LearningSessionStep; no extra per-step constraint is necessary.

LearningPlanningSettings has the same no-retry vision input/safety defaults as teaching. Examples must cover initial assessment, wrong/unknown → teaching, correct-but-unsound → teaching, teaching → same-target assessment, supported progress, and finish with explicit uncertainty. These are prompt-guided decisions, not local output-repair branches.

### 5. Add FSRS-backed memory reviews and due selection

Add the exact runtime dependency `fsrs==6.3.3` to pyproject.toml and resolve uv.lock during execution; do not install optimizer extras. This version's published APIs were inspected at `https://github.com/open-spaced-repetition/py-fsrs/tree/v6.3.3` and `https://pypi.org/pypi/fsrs/6.3.3/json`. Its Python requirement is >=3.10; the project runs Python 3.14.

Create `learning/memory/{__init__,models,node,repository,queries,records,settings,composition}.py`; there is no existing memory/scheduling implementation. This is an algorithm/database component, not a fake DSPy module or a new service framework.

Domain contracts in models.py (extra forbidden; durable records inherit Vertex):

- `MemoryState`: target LearningTarget, card_json str, due_at datetime, review_count int. The opaque card JSON is FSRS's canonical full state, not duplicated KMS difficulty/stability fields.
- `MemoryReview`: memory_uuid str, assessment_uuid str, rating ReviewRating, reviewed_at datetime, sequence int. Those UUID associations hydrate from relationships; they are not duplicated properties/JSON.
- `MemoryHistory`: memory MemoryState, reviews list[MemoryReview].
- `ReviewTargetContext`: target LearningTarget, understanding str, knowledge KnowledgeContext, solution_task Task|None=None.
- `ReviewInput`: target LearningTarget, response_uuid str, assessment_uuid str, rating ReviewRating, reviewed_at datetime, memory MemoryState|None.
- `MemoryReviewState(TypedDict)`: user_uuid str, session_uuid str, assessment_uuid str.

Define stable backend identities in models.py: `memory_uuid(user_uuid: str, target: LearningTarget) -> str` is UUID5(NAMESPACE_URL, `'kms:learning-memory:' + json.dumps([user_uuid, target.kind, target.uuid], separators=(',', ':'))`); `review_uuid(user_uuid: str, response_uuid: str) -> str` uses the prefix `'kms:learning-review:'` and JSON [user_uuid, response_uuid]. Including kind distinguishes same UUID strings under different target labels.

MemorySettings contains only desired_retention float=0.9, gt=0, lt=1; add `Settings.memory` default factory. Composition builds `fsrs.Scheduler(desired_retention=settings.memory.desired_retention, enable_fuzzing=False)`, preserving the pinned library's default parameters/learning steps/relearning steps/maximum interval. No optimizer, custom scheduler abstraction, difficulty model, duration estimate, or extra policy settings.

Graph storage:

- `(User)-[:HAS_MEMORY]->(LearningMemory)-[:FOR_TARGET]->(Proposition|Argument|Solution)`.
- `(LearningMemory)-[:HAS_REVIEW]->(MemoryReview)-[:BASED_ON]->(TaskAssessment)`.
- LearningMemory stores lifecycle fields, card_json, native due_at, and review_count. Target/owner identity comes from relationships; UUID5 enforces one learner/typed-target pair without duplicate owner_uuid/target_uuid properties.
- MemoryReview stores lifecycle fields, rating, native reviewed_at, and sequence; its deterministic UUID enforces one scheduling event per learner response, not per regrade.
- Add unique constraints `learning_memory_uuid` and `memory_review_uuid`, plus range index `learning_memory_due` on LearningMemory.due_at to SCHEMA_STATEMENTS. No vector indexes, database reset, or migration.
- All new relationships receive the established created_at/updated_at metadata.

`MemoryRepository(session_factory)` owns all queries and transactions. Reuse KnowledgeRepository.serialized's owner-write-lock shape, using a dedicated short-lived `_memory_lock` User property set to randomUUID then removed inside the transaction. Require the owning User with `single(strict=True)` before yielding. Methods:

```python
@asynccontextmanager
async def serialized(self, user_uuid: str) -> AsyncIterator[AsyncTransaction]: ...
async def read_review_input(self, transaction: AsyncTransaction, user_uuid: str, session_uuid: str, assessment_uuid: str) -> ReviewInput: ...
async def save_review(self, transaction: AsyncTransaction, user_uuid: str, memory: MemoryState, review: MemoryReview) -> MemoryState: ...
async def read_memory(self, user_uuid: str, target: LearningTarget) -> MemoryHistory: ...
async def list_due(self, user_uuid: str, as_of: datetime) -> list[MemoryState]: ...
async def read_target(self, user_uuid: str, target: LearningTarget) -> ReviewTargetContext: ...
```

`read_review_input` follows the owned session → Task → TaskResponse → requested TaskAssessment path. Read the single ASSESSES target from the actual task and the rating from the stored TaskGrade; neither is caller supplied. Optional existing memory is read through HAS_MEMORY/FOR_TARGET. Reviewed time is the response's actual created_at, converted with astimezone(UTC), not grading time. This query does not read image files or call inference.

`MemoryReviewNode(scheduler: fsrs.Scheduler, repository: MemoryRepository).run(state: MemoryReviewState) -> dict[str, MemoryState]` holds that owner lock across read, the small pure FSRS update, and save. For a first review create Card(card_id=int(UUID(memory_uuid)), due=reviewed_at); explicit ID avoids the library's timestamp/sleep constructor. Otherwise deserialize Card.from_json(existing.card_json). Map ReviewRating via an explicit four-entry mapping to fsrs.Rating.Again/Hard/Good/Easy. Call review_card with review_datetime=reviewed_at. Keep only the updated card; MemoryReview records the source rating/time and assessment provenance.

Under the same owner lock, assign sequence=1 for first review or existing.review_count+1 otherwise; publish that value as both MemoryReview.sequence and MemoryState.review_count in the atomic save. This records actual FSRS application order even when attempt/creation timestamps tie. The counter is storage chronology, not another scheduling model; do not count all historical edges or infer application order from UUIDs.

If reviewed_at precedes card.last_review, raise ValueError('Review precedes the latest recorded attempt') before saving. This protects actual memory-history ordering against delayed/concurrent processing; it is not model-output validation or a timestamp fallback. Equal timestamps are allowed. Never rewrite the source response time, replay history automatically, or turn a regrade into a new attempt.

Construct MemoryState with the deterministic UUID, Card.to_json(), and card.due; preserve existing created_at and advance updated_at at publication. Construct MemoryReview with deterministic response-based UUID and the selected assessment. `save_review` commits memory/card/due, the review event and provenance together: use a full-pattern `MERGE (user)-[:HAS_MEMORY]->(memory:LearningMemory {uuid: $memory_uuid})` with bound owned user, then FOR_TARGET to the strictly matched owned target; CREATE the MemoryReview and its links. The full-pattern merge must not attach a foreign pre-existing memory. A repeated scheduling call for the same response—even with another assessment—propagates the review UUID constraint failure and rolls back the memory update. Keep all TaskAssessments as history; only the explicitly selected assessment is used once.

Allow scheduling an assessment from a completed owned session: it updates learner memory, not the session timeline. No LLM call or filesystem work occurs under this transaction. The owner lock serializes the same learner-target across different sessions, which the existing per-session lock cannot do.

`list_due` captures one as_of from its caller, returns records with due_at <= as_of, ordered by due_at, target.kind, target.uuid. Return [] for a valid owner with nothing due; require a strict owner row so an unknown owner is not disguised as an empty queue. Do not load full source/knowledge contexts for every queued item. A target gets memory state when its first graded attempt is applied; there is no silent enrollment of all ingested knowledge.

`read_target` resolves only the selected owner-scoped target, without embeddings: Proposition → its full model in KnowledgeContext.matched/propositions; Argument → that Argument and exactly its owned premise/conclusion Proposition endpoints (matched=[]); Solution → locate its owning session/task and reuse READ_OWNED_TASK plus hydrate_task to return that original Task as solution_task. For Solution, knowledge is an empty KnowledgeContext because the complete problem/reference solution is supplied as native material rather than invented propositions. Underlying source windows remain caller supplied. Understanding is proposition.statement, the argument conclusion's statement, or the original task.understanding respectively. Missing/foreign targets fail strictly; no substituted target or answer.

`read_memory` requires the owner-linked typed target, hydrates card/due/count and its reviews ordered by sequence, and returns MemoryHistory. Reviews remain linked to actual selected assessments. No invalidation/soft-delete policy is introduced; use the current perfect, owned-target contract.

### 6. Expose the three independent capabilities

Use application-owned database/inference lifetimes as already implemented. Teaching and planning preflight only their own inference settings and use `_application_resources(..., embeddings_required=False)`. Memory operations use `_database_resources` only—no ChatGPT, embeddings, tokenizers, or local-model startup.

Expose these application signatures:

```python
async def generate_teaching(settings: Settings, user_uuid: str, session_uuid: str, request: TeachingGenerationRequest) -> PresentedTeaching: ...
async def plan_learning_step(settings: Settings, user_uuid: str, session_uuid: str, request: LearningPlanningRequest) -> LearningDecisionRecord: ...
async def record_memory_review(settings: Settings, user_uuid: str, session_uuid: str, assessment_uuid: str) -> MemoryState: ...
async def read_memory(settings: Settings, user_uuid: str, target: LearningTarget) -> MemoryHistory: ...
async def list_due_reviews(settings: Settings, user_uuid: str, as_of: datetime | None = None) -> list[MemoryState]: ...
async def read_review_target(settings: Settings, user_uuid: str, target: LearningTarget) -> ReviewTargetContext: ...
```

Require timezone-aware as_of; set omission to datetime.now(UTC) once and normalize a supplied value with astimezone(UTC) at the application boundary. `record_memory_review` builds the review node and returns its persisted memory. `read_review_target` yields fields directly usable to assemble a task/teaching request for the selected due target, including the original problem for solution targets.

Keep generate_task, record_task_response and grade_task_response independently callable, updating only their atomic-target contracts. Grading does not implicitly schedule, teaching does not implicitly assess, planning does not implicitly retrieve, and scheduling does not implicitly teach. The live verification composes these operations temporarily; do not add a permanent coordinator, UI, acquisition loop, or three separate learning/practice engines.

## Verification

Planning is read-only; no tests, dependency changes, model calls, or database writes have been performed for this task. During execution, follow docs/STYLE.md and request LSP references before exported-symbol cutovers. The available Ruff LSP previously returned unsupported-method errors for references; if still unsupported, use the exact symbol searches above, not guessed callers.

### Focused behavioral coverage

- Update existing task generation/grading/native/session tests for atomic targets and canonical shared imports. Existing source-only generation must target its own newly committed Solution; explicit Proposition/Argument/Solution targets must survive model output and readback unchanged. Generating another assessment for an existing Solution creates a fresh hidden solution but keeps the original target. Supporting propositions do not produce additional ASSESSES or memory records.
- Add `tests/test_kms_teaching_generation.py` and `tests/test_kms_teaching_native.py`: a request with an earlier history image before the evidence image must resolve the emitted catalog index to that evidence image; repeated references preserve ordered occurrences. Inspect actual native projected input for image content and absence of filesystem paths in model-visible JSON. Exercise material finalization and failure propagation so no learner-facing result is returned before its durable append. Do not write tests that merely echo predictor output or inspect prompt wording.
- Extend `tests/test_kms_learning_session_integration.py`: real Task → Decision → Teaching → Decision → Task chronology, exact target and same-session assessment citations, bytes/path image hydration, target/grounding/citation foreign-owner rollback, and concurrent appends preserving one chain. Confirm finish remains atomic and a solution owned through a completed session can be targeted by a new session. Query actual relationships, not source strings.
- Replace the old probe test with `tests/test_kms_learning_planning.py`, retaining authorization/persistence-failure behavior. Native budget demonstrations must include Teaching and its source image, not just the former task/decision variants. The live composition below establishes that the revised prompt can choose teaching and then reassessment.
- Add `tests/test_kms_memory.py`: compare the actual review node's persisted Card JSON/due against the pinned library using fixed UTC attempts and explicit expected Rating values for Again/Hard/Good/Easy; cover state serialization across several chronological attempts. Check the response timestamp is used despite a later assessment timestamp. Fixed responses are actual distinct attempts, not fabricated regrades masquerading as reviews.
- Add `tests/test_kms_memory_integration.py`, using the existing KMS_NEO4J_IT and four KMS_DATABASE environment gates. For each of Proposition/Argument/Solution, two different tasks assessing the same target must update one memory; unrelated targets and same UUID strings under different labels remain distinct. A repeated response via either the same or another grade must fail and preserve the exact prior card/due/review set. An older attempt must raise the specified ValueError and preserve state. A foreign assessment/target must not create memory. Apply concurrent same-time attempts from separate sessions for the same owner/target and confirm both reviews survive; replay their actual stored order to match final FSRS state.
- For concurrent memory attempts, assert sequences 1..N and review_count=N, and replay in sequence order even with identical reviewed_at/created_at timestamps. Duplicate/stale-attempt rollback must preserve the counter as well as card/due/events.
- Due-selection coverage: fixed as_of with an exactly-due item, an overdue item, and a future item; include timestamp ties and verify due_at/kind/uuid order, other-user exclusion, and [] for an owned empty queue. Resolve selected argument context with all premise/conclusion endpoints, and resolve a selected solution's original task/native material without loading all queued target content.
- Extend existing application/settings/schema tests for published contracts and actual ownership/resource failure boundaries. Assert new schema identities/indexes through initialization, not test-specific magic strings or call-order echoes.

Run the narrow offline behavior set once after the cutovers:

```sh
uv run pytest tests/test_kms_task_generation.py tests/test_kms_task_native.py tests/test_kms_task_grading.py tests/test_kms_task_grading_native.py tests/test_kms_learning_session.py tests/test_kms_learning_planning.py tests/test_kms_teaching_generation.py tests/test_kms_teaching_native.py tests/test_kms_memory.py tests/test_kms_database_schema.py tests/test_kms_settings.py tests/test_kms_application.py
```

With the existing local Neo4j available and credentials supplied by the environment, run the scoped real persistence set once:

```sh
KMS_NEO4J_IT=1 uv run pytest tests/test_kms_learning_session_integration.py tests/test_kms_memory_integration.py
```

Use isolated UUID fixtures and clean only those fixtures. Do not clear Neo4j or mutate unrelated source/knowledge/session records.

### Temporary live component composition

Use an uncommitted, disposable harness and the existing chatgpt-luna subscription profile (gpt-6-luna, low reasoning), with cache disabled. No API-key provider fallback, retries, or permanent graph. In a fresh owned session:

1. Supply the existing function graph image (open point at (2,15), filled point at (2,5)) as a source unit window. Generate one source-backed task asking for f(2), using NewSolutionTarget. Read its committed task/reference solution to establish the target identity and native material.
2. Persist a high-confidence answer using the open point's value, grade the actual stored response, and explicitly apply that assessment through record_memory_review. Verify the observed incorrect/Again result and one memory targeting the task's Solution.
3. Select the relevant task history; request a learning decision anchored to that target. Expect targeted teaching based on the actual incorrect assessment. Generate teaching whose objective explains the filled/open-point distinction and includes the supplied diagram; persist the teaching on the same timeline.
4. Select the task and teaching steps together. Plan again; expect assessment of the same target, not a new understanding. Resolve that target via read_review_target and generate a second task using the original SolutionTarget and original solution_task. Its own hidden solution UUID must differ without changing its assessed target.
5. Persist a sound answer giving f(2)=5 from the filled point, grade it, and apply the grade. Confirm the same memory UUID now has two actual-attempt review events. Compare its card/due to direct FSRS replay of the observed ratings/timestamps.
6. Query due reviews at that recorded due_at and resolve the returned solution target back to the original problem/solution. Inspect persisted step order, response/assessment provenance, and native teaching material. Report actual learner-visible prompt/teaching excerpts, observed grades/ratings and due time; distinguish model-quality discrepancies from structural failures rather than repairing model output.

Remove the harness and scoped fixtures. If authentication or Neo4j cannot be reached, report that exact live verification limit and the checks actually completed; do not substitute mocks or another provider and call it a live pass.

## Critical implementation anchors

- `src/kms/learning/task/generation/node.py` — existing backend identity construction and exact shared material resolver to extract.
- `src/kms/learning/session/queries.py` — timeline transaction, graph-owned associations, authoritative task/solution read used by review selection.
- `src/kms/learning/session/records.py` — native Path hydration and explicit discriminated step/target reconstruction.
- `src/kms/inference/generation.py` — unchanged native input projection/catalog contract for both output consumers.
- `src/kms/knowledge/storage.py` — existing serialized owner-write transaction shape to reuse for memory updates.

## Assumptions and contingencies

- Scope is confirmed: teaching generation, unified session planning, and FSRS/review selection; permanent graph/UI composition is deferred.
- Atomic schedulable targets are confirmed: propositions, arguments, and individual task solutions. Source units remain evidence; supporting knowledge may be plural.
- Domain/model inputs and outputs satisfy their contracts. Preserve authorization, artifact integrity, capacity errors, and transactional failures; do not add model retries, output repair, silent context truncation, or compatibility behavior.
- A selected assessment schedules its actual response at most once; regrading does not create another attempt. Late out-of-order scheduling fails rather than corrupting elapsed-time state.
- FSRS starts with the first applied graded attempt. Explanation presentation alone does not establish mastery or create a review.
- This is a clean contract cutover. Existing multiple-target/probe-planning data is not migrated or read through aliases; live verification uses fresh isolated records, without deleting existing user data.


