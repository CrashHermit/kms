"""Application entry points for running KMS pipelines."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from kms.config.inference import StageInferenceSettings
from kms.config.settings import Settings
from kms.database import schema
from kms.database.client import DatabaseClient
from kms.inference.chatgpt import (
    ensure_chatgpt_authenticated,
)
from kms.inference.chatgpt import (
    login_chatgpt as _login_chatgpt,
)
from kms.inference.embedding.resources import (
    EmbeddingResources,
    embedding_resources,
)
from kms.inference.predictors import PredictorFactory
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.knowledge.composition import build_committer
from kms.knowledge.correctness.composition import build_correctness_judge
from kms.knowledge.correctness.models import (
    CorrectnessJudgment,
    CorrectnessRequest,
)
from kms.knowledge.extraction import composition as extraction
from kms.knowledge.extraction.models import UnitExtractionResult
from kms.knowledge.models import (
    Argument,
    KnowledgeContext,
    Proposition,
    ReasonedKnowledge,
    StoredKnowledge,
)
from kms.knowledge.preparation.composition import build_knowledge_preparation
from kms.knowledge.preparation.models import PreparationResult
from kms.knowledge.reasoning import composition as reasoning
from kms.knowledge.retrieval.composition import build_search
from kms.knowledge.sufficiency import composition as sufficiency
from kms.knowledge.sufficiency.models import SufficiencyJudgment
from kms.learning.memory import composition as memory_composition
from kms.learning.memory.models import (
    MemoryHistory,
    MemoryState,
    ReviewTargetContext,
)
from kms.learning.memory.repository import MemoryRepository
from kms.learning.probe.goal.models import ClarifiedGoal
from kms.learning.probe.targets.composition import build_target_decomposition
from kms.learning.references import LearningTarget
from kms.learning.session.composition import build_session_repository
from kms.learning.session.models import (
    LearningDecisionRecord,
    LearningSession,
    SessionHistory,
)
from kms.learning.session.planning import composition as learning_planning
from kms.learning.session.planning.models import LearningPlanningRequest
from kms.learning.task.generation import composition as task_generation
from kms.learning.task.generation.models import TaskGenerationRequest
from kms.learning.task.grading import composition as task_grading
from kms.learning.task.models import (
    PresentedTask,
    TaskAnswer,
    TaskAssessment,
    TaskResponse,
)
from kms.learning.teaching.generation import composition as teaching_generation
from kms.learning.teaching.generation.models import TeachingGenerationRequest
from kms.learning.teaching.models import PresentedTeaching
from kms.local_models.runtime import LocalModelRuntime
from kms.media.models import Material
from kms.retrieval.selection.composition import build_search_selector
from kms.source.capture.models import PDFCandidate
from kms.source.capture.resources import capture_resources
from kms.source.composition import build_source_ingestion_service
from kms.source.discovery.composition import build_source_discovery
from kms.source.discovery.models import SourceCandidate
from kms.source.discovery.resources import discovery_resources
from kms.source.models.inputs import CapturedPDFSource
from kms.source.models.snapshot import SourceSnapshot
from kms.source.models.source import Source
from kms.source.models.unit import SourceUnitKind
from kms.source.retrieval.composition import (
    build_source_reader,
    build_source_search_service,
)
from kms.source.retrieval.models import (
    SourceBlockRead,
    SourceImage,
    SourceSearchHit,
    SourceUnitRead,
    SourceUnitSearchHit,
)
from kms.source.storage.catalog import SourceCatalogRepository
from kms.train.recorder import Recorder
from kms.users.models import User
from kms.users.repository import UserRepository
from kms.users.service import UserService


@dataclass(frozen=True, slots=True)
class _ApplicationResources:
    """Resources shared by the stages of one application invocation."""

    local_models: LocalModelRuntime
    embeddings: EmbeddingResources | None
    tokenizers: LanguageModelTokenizers
    database: DatabaseClient
    recorder: Recorder | None


@asynccontextmanager
async def _application_resources(
    settings: Settings,
    *,
    embeddings_required: bool = True,
) -> AsyncIterator[_ApplicationResources]:
    """Own invocation resources and acquire retrieval only when requested."""
    database = DatabaseClient(settings.database)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = LanguageModelTokenizers(settings)
    try:
        async with AsyncExitStack() as stack:
            local_models = await stack.enter_async_context(
                LocalModelRuntime(settings.local_models)
            )
            embeddings = (
                await stack.enter_async_context(embedding_resources(settings))
                if embeddings_required
                else None
            )
            if embeddings is None:
                await schema.ensure_structural_schema(database.session)
            else:
                await schema.ensure_schema(
                    database.session,
                    embedding_dimension=embeddings.model.dimension,
                )
            yield _ApplicationResources(
                local_models=local_models,
                embeddings=embeddings,
                tokenizers=tokenizers,
                database=database,
                recorder=recorder,
            )
    finally:
        await database.close()


def login_chatgpt() -> None:
    """Start LiteLLM's explicit ChatGPT device login from the terminal."""
    _login_chatgpt()


async def _preflight_chatgpt(
    settings: Settings,
    *stage_settings: BaseModel,
) -> None:
    """Require existing ChatGPT credentials for selected inference stages."""
    for stage in stage_settings:
        if _contains_chatgpt_inference(stage, settings):
            await ensure_chatgpt_authenticated()
            return


def _contains_chatgpt_inference(
    value: object,
    settings: Settings,
) -> bool:
    """Resolve inference leaves within one explicitly relevant stage tree."""
    if isinstance(value, StageInferenceSettings):
        return settings.llm.resolve(value).backend == 'chatgpt'
    if isinstance(value, BaseModel):
        return any(
            _contains_chatgpt_inference(field, settings)
            for field in value.__dict__.values()
        )
    if isinstance(value, dict):
        return any(
            _contains_chatgpt_inference(field, settings)
            for field in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_chatgpt_inference(field, settings) for field in value
        )
    return False


@asynccontextmanager
async def _database_resources(
    settings: Settings,
) -> AsyncIterator[DatabaseClient]:
    """Own database resources for commands that do not need local models."""
    database = DatabaseClient(settings.database)
    try:
        await schema.ensure_structural_schema(database.session)
        yield database
    finally:
        await database.close()


async def list_users(settings: Settings) -> list[User]:
    """Return users for terminal onboarding and selection."""
    database = DatabaseClient(settings.database)
    try:
        return await UserService(UserRepository(database.session)).list_users()
    finally:
        await database.close()


async def create_user(settings: Settings, name: str) -> User:
    """Create a user after ensuring only structural schema."""
    database = DatabaseClient(settings.database)
    try:
        await schema.ensure_structural_schema(database.session)
        return await UserService(UserRepository(database.session)).create_user(
            name
        )
    finally:
        await database.close()


async def list_unowned_sources(settings: Settings) -> list[Source]:
    """Return legacy sources available for explicit adoption."""
    database = DatabaseClient(settings.database)
    try:
        return await SourceCatalogRepository(
            database.session
        ).list_unowned_sources()
    finally:
        await database.close()


async def adopt_source(
    settings: Settings,
    user_uuid: str,
    source_uuid: str,
) -> None:
    """Explicitly associate an unowned source with a user."""
    database = DatabaseClient(settings.database)
    try:
        await SourceCatalogRepository(database.session).adopt_source(
            user_uuid,
            source_uuid,
        )
    finally:
        await database.close()


async def list_sources(settings: Settings, user_uuid: str) -> list[Source]:
    """Return persisted sources owned by one user."""
    database = DatabaseClient(settings.database)
    try:
        return await SourceCatalogRepository(database.session).list_sources(
            user_uuid
        )
    finally:
        await database.close()


async def search_sources(
    settings: Settings,
    user_uuid: str,
    query: Material,
    *,
    source_uuids: list[str] | None = None,
) -> list[SourceSearchHit]:
    """Search a user's source blocks with the configured embedding."""

    if source_uuids == []:
        return []
    await _preflight_chatgpt(settings, settings.search_selection)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        selection = build_search_selector(
            settings, predictors=predictors, tokenizers=resources.tokenizers
        )
        service = build_source_search_service(
            resources.database,
            embedder=resources.embeddings.embedder,
            settings=settings.source_search,
            selection=selection,
        )
        return await service.search_sources(
            user_uuid, query, source_uuids=source_uuids
        )


async def read_source_blocks(
    settings: Settings, user_uuid: str, block_uuids: list[str]
) -> list[SourceBlockRead]:
    """Read authorized source blocks."""
    if not block_uuids:
        return []
    async with _database_resources(settings) as database:
        return await build_source_reader(database).read_blocks(
            user_uuid, block_uuids
        )


async def read_source_units(
    settings: Settings,
    user_uuid: str,
    unit_uuids: list[str],
    *,
    before_units: int = 0,
    after_units: int = 0,
) -> list[SourceUnitRead]:
    """Read requested units and bounded neighboring units."""
    if not unit_uuids:
        return []
    async with _database_resources(settings) as database:
        return await build_source_reader(database).read_units(
            user_uuid,
            unit_uuids,
            before_units=before_units,
            after_units=after_units,
        )


async def read_source_images(
    settings: Settings,
    user_uuid: str,
    block_uuids: list[str],
) -> list[SourceImage]:
    """Read verified image bytes owned by the requested source blocks."""
    if not block_uuids:
        return []
    async with _database_resources(settings) as database:
        service = build_source_reader(database)
        return await service.read_images(user_uuid, block_uuids)


async def search_source_units(
    settings: Settings,
    user_uuid: str,
    query: Material,
    *,
    source_uuids: list[str] | None = None,
    kind: SourceUnitKind | None = None,
) -> list[SourceUnitSearchHit]:
    """Search complete units in the user's current-schema sources."""
    if source_uuids == []:
        return []
    await _preflight_chatgpt(settings, settings.search_selection)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        selection = build_search_selector(
            settings, predictors=predictors, tokenizers=resources.tokenizers
        )
        service = build_source_search_service(
            resources.database,
            embedder=resources.embeddings.embedder,
            settings=settings.source_search,
            selection=selection,
        )
        return await service.search_source_units(
            user_uuid, query, source_uuids=source_uuids, kind=kind
        )


async def extract_unit_knowledge(
    settings: Settings,
    user_uuid: str,
    unit_uuid: str,
    understanding: str,
    *,
    existing_propositions: list[Proposition],
    existing_arguments: list[Argument],
) -> UnitExtractionResult | None:
    """Extract source-grounded knowledge candidates from an owned unit."""
    await _preflight_chatgpt(settings, settings.knowledge_extraction.inference)
    async with _application_resources(
        settings, embeddings_required=False
    ) as resources:
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        node = extraction.build_node(
            settings,
            resources.database,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        result = await node.run(
            {
                'user_uuid': user_uuid,
                'unit_uuid': unit_uuid,
                'understanding': understanding,
                'context': KnowledgeContext(
                    matched=[],
                    propositions=existing_propositions,
                    arguments=existing_arguments,
                ),
            }
        )
        return result['pending']


async def store_unit_knowledge(
    settings: Settings, user_uuid: str, result: UnitExtractionResult
) -> StoredKnowledge:
    """Audit and persist source-grounded candidates and source provenance."""
    await _preflight_chatgpt(settings, settings.knowledge_identity.inference)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        committer = build_committer(
            settings,
            resources.database,
            embeddings=resources.embeddings,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        return await committer.commit(user_uuid, result)


async def search_knowledge(
    settings: Settings, user_uuid: str, understanding: str
) -> KnowledgeContext:
    """Find relevant propositions and their connected arguments."""
    await _preflight_chatgpt(settings, settings.search_selection)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        selection = build_search_selector(
            settings, predictors=predictors, tokenizers=resources.tokenizers
        )
        search = build_search(
            settings,
            resources.database,
            embeddings=resources.embeddings,
            selection=selection,
        )
        return await search.search(user_uuid, Material.from_text(understanding))


async def prepare_knowledge(
    settings: Settings, user_uuid: str, understanding: str
) -> PreparationResult:
    """Prepare knowledge for one target from owned knowledge and sources."""
    await _preflight_chatgpt(
        settings,
        settings.search_selection,
        settings.knowledge_sufficiency,
        settings.knowledge_reasoning,
        settings.knowledge_identity,
        settings.knowledge_extraction,
    )
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        graph = build_knowledge_preparation(
            settings,
            resources.database,
            embeddings=resources.embeddings,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        return await graph.prepare(user_uuid, understanding)


async def decompose_learning_goal(
    settings: Settings, goal: ClarifiedGoal
) -> list[str]:
    """Decompose one clarified goal into independently searchable targets."""
    await _preflight_chatgpt(settings, settings.target_decomposition.inference)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = LanguageModelTokenizers(settings)
    async with LocalModelRuntime(settings.local_models) as local_models:
        predictors = PredictorFactory(
            settings, local_models, recorder, tokenizers=tokenizers
        )
        module = build_target_decomposition(settings, predictors=predictors)
        return await module.acall(goal=goal)


async def create_learning_session(
    settings: Settings, user_uuid: str, goal: ClarifiedGoal, targets: list[str]
) -> LearningSession:
    """Create an owned session for an already decomposed learning goal."""
    async with _database_resources(settings) as database:
        repository = build_session_repository(settings, database)
        return await repository.create_session(
            user_uuid, LearningSession(goal=goal.goal, targets=targets)
        )


async def read_learning_session(
    settings: Settings, user_uuid: str, session_uuid: str
) -> SessionHistory:
    """Read one owned session and its persisted interaction history."""
    async with _database_resources(settings) as database:
        return await build_session_repository(settings, database).read_session(
            user_uuid, session_uuid
        )


async def record_task_response(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    task_uuid: str,
    answer: TaskAnswer,
) -> TaskResponse:
    """Persist a learner's free response to an owned, committed task."""
    async with _database_resources(settings) as database:
        return await build_session_repository(
            settings, database
        ).record_task_response(
            user_uuid,
            session_uuid,
            TaskResponse(task_uuid=task_uuid, answer=answer),
        )


async def generate_task(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    request: TaskGenerationRequest,
) -> PresentedTask:
    """Commit a grounded task and hidden solution before exposing its prompt."""
    await _preflight_chatgpt(settings, settings.task_generation.inference)
    async with _application_resources(
        settings, embeddings_required=False
    ) as resources:
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        node = task_generation.build_node(
            settings,
            resources.database,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        result = await node.run(
            {
                'user_uuid': user_uuid,
                'session_uuid': session_uuid,
                'request': request,
            }
        )
        stored = result['task']
        return PresentedTask(task_uuid=stored.uuid, prompt=stored.prompt)


async def generate_teaching(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    request: TeachingGenerationRequest,
) -> PresentedTeaching:
    """Persist grounded teaching before exposing it to the learner."""
    await _preflight_chatgpt(settings, settings.teaching_generation.inference)
    async with _application_resources(
        settings, embeddings_required=False
    ) as resources:
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        node = teaching_generation.build_node(
            settings,
            resources.database,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        result = await node.run(
            {
                'user_uuid': user_uuid,
                'session_uuid': session_uuid,
                'request': request,
            }
        )
        teaching = result['teaching']
        return PresentedTeaching(
            teaching_uuid=teaching.uuid, material=teaching.material
        )


async def grade_task_response(
    settings: Settings, user_uuid: str, session_uuid: str, response_uuid: str
) -> TaskAssessment:
    """Grade recorded answer/work against its actual stored task and solution."""
    await _preflight_chatgpt(settings, settings.task_grading.inference)
    async with _application_resources(
        settings, embeddings_required=False
    ) as resources:
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        node = task_grading.build_node(
            settings,
            resources.database,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        result = await node.run(
            {
                'user_uuid': user_uuid,
                'session_uuid': session_uuid,
                'response_uuid': response_uuid,
            }
        )
        return result['assessment']


async def plan_learning_step(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    request: LearningPlanningRequest,
) -> LearningDecisionRecord:
    """Persist the next adaptive learning decision or bounded completion."""
    await _preflight_chatgpt(settings, settings.learning_planning.inference)
    async with _application_resources(
        settings, embeddings_required=False
    ) as resources:
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        node = learning_planning.build_node(
            settings,
            resources.database,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        result = await node.run(
            {
                'user_uuid': user_uuid,
                'session_uuid': session_uuid,
                'request': request,
            }
        )
        return result['decision']


async def record_memory_review(
    settings: Settings, user_uuid: str, session_uuid: str, assessment_uuid: str
) -> MemoryState:
    """Apply one recorded attempt to the learner's memory."""
    async with _database_resources(settings) as database:
        node = memory_composition.build_node(settings, database)
        result = await node.run(
            {
                'user_uuid': user_uuid,
                'session_uuid': session_uuid,
                'assessment_uuid': assessment_uuid,
            }
        )
        return result['memory']


async def read_memory(
    settings: Settings, user_uuid: str, target: LearningTarget
) -> MemoryHistory:
    """Read one owned target's memory and ordered reviews."""
    async with _database_resources(settings) as database:
        return await MemoryRepository(database.session).read_memory(
            user_uuid, target
        )


async def list_due_reviews(
    settings: Settings, user_uuid: str, as_of: datetime | None = None
) -> list[MemoryState]:
    """List the learner's due targets at one UTC instant."""
    if as_of is None:
        as_of = datetime.now(UTC)
    elif as_of.utcoffset() is None:
        raise ValueError('as_of must be timezone-aware')
    else:
        as_of = as_of.astimezone(UTC)
    async with _database_resources(settings) as database:
        return await MemoryRepository(database.session).list_due(
            user_uuid, as_of
        )


async def read_review_target(
    settings: Settings, user_uuid: str, target: LearningTarget
) -> ReviewTargetContext:
    """Resolve the owned target selected for review."""
    async with _database_resources(settings) as database:
        return await MemoryRepository(database.session).read_target(
            user_uuid, target
        )


async def judge_knowledge_sufficiency(
    settings: Settings,
    understanding: str,
    context: KnowledgeContext,
) -> SufficiencyJudgment:
    """Judge whether supplied knowledge addresses one atomic understanding."""
    await _preflight_chatgpt(settings, settings.knowledge_sufficiency.inference)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = LanguageModelTokenizers(settings)
    async with LocalModelRuntime(settings.local_models) as local_models:
        predictors = PredictorFactory(
            settings,
            local_models,
            recorder,
            tokenizers=tokenizers,
        )
        node = sufficiency.build_node(
            settings, predictors=predictors, tokenizers=tokenizers
        )
        result = await node.run(
            {'understanding': understanding, 'context': context}
        )
        return result['judgment']


async def judge_knowledge_correctness(
    settings: Settings, request: CorrectnessRequest
) -> CorrectnessJudgment:
    """Assess one proposition or argument against supplied source evidence."""
    await _preflight_chatgpt(settings, settings.knowledge_correctness.inference)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = LanguageModelTokenizers(settings)
    async with LocalModelRuntime(settings.local_models) as local_models:
        predictors = PredictorFactory(
            settings, local_models, recorder, tokenizers=tokenizers
        )
        judge = build_correctness_judge(
            settings, predictors=predictors, tokenizers=tokenizers
        )
        return await judge.acall(request=request)


async def reason_knowledge(
    settings: Settings, user_uuid: str, understanding: str
) -> ReasonedKnowledge:
    """Search existing knowledge, construct reasoning, and persist it."""
    await _preflight_chatgpt(
        settings,
        settings.search_selection,
        settings.knowledge_reasoning,
        settings.knowledge_identity,
    )
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        graph = reasoning.build_graph(
            settings,
            resources.database,
            embeddings=resources.embeddings,
            predictors=predictors,
            tokenizers=resources.tokenizers,
        )
        return await graph.reason(user_uuid, understanding)


async def ingest_source(
    settings: Settings,
    user_uuid: str,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceSnapshot:
    """Ingest one local PDF as a complete immutable owned source."""
    data = await asyncio.to_thread(Path(pdf_path).read_bytes)
    capture = CapturedPDFSource(
        data=data,
        title=Path(pdf_path).name,
        url=None,
        retrieved_at=datetime.now(UTC),
    )
    return await ingest_captured_source(
        settings, user_uuid, capture, pages=pages
    )


async def ingest_captured_source(
    settings: Settings,
    user_uuid: str,
    capture: CapturedPDFSource,
    *,
    pages: list[int] | None = None,
) -> SourceSnapshot:
    """Process and publish a captured PDF as an owned source."""
    async with _database_resources(settings) as database:
        await UserRepository(database.session).read_user(user_uuid)
    await _preflight_chatgpt(settings, settings.source_processing)
    async with _application_resources(settings) as resources:
        assert resources.embeddings is not None
        predictors = PredictorFactory(
            settings,
            resources.local_models,
            resources.recorder,
            tokenizers=resources.tokenizers,
        )
        ingestion = build_source_ingestion_service(
            settings,
            resources.database,
            predictors=predictors,
            embeddings=resources.embeddings,
            tokenizers=resources.tokenizers,
        )
        return await ingestion.ingest(user_uuid, capture, pages=pages)


async def discover_sources(
    settings: Settings, question: str
) -> list[SourceCandidate]:
    """Search and select useful links without capturing or ingesting them."""
    await _preflight_chatgpt(settings, settings.source_discovery.inference)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    async with AsyncExitStack() as stack:
        local_models = await stack.enter_async_context(
            LocalModelRuntime(settings.local_models)
        )
        web = await stack.enter_async_context(
            discovery_resources(settings.source_discovery)
        )
        predictors = PredictorFactory(
            settings,
            local_models,
            recorder,
            tokenizers=LanguageModelTokenizers(settings),
        )
        discovery = build_source_discovery(
            settings, web=web, predictors=predictors
        )
        return await discovery.discover(question)


async def capture_source(
    settings: Settings, url: str, *, title: str | None = None
) -> CapturedPDFSource | PDFCandidate:
    """Capture a web page as a PDF, or return a PDF link for manual handling."""
    async with capture_resources(settings.source_capture) as capture:
        return await capture.capture(url, title=title)
