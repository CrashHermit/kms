"""Application entry points for running KMS2 pipelines."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from kms2.composition.embedding import (
    build_embedding_client,
    embedding_dimension,
)
from kms2.composition.reranking import reranker_resources
from kms2.composition.research import (
    research_resources as open_research_resources,
)
from kms2.composition.research_source_migration import (
    build_research_source_migration_finalizer,
)
from kms2.composition.services import (
    build_knowledge_search_service,
    build_knowledge_service,
    build_learner_observation_search_service,
    build_learner_observation_service,
    build_learning_service,
    build_research_source_migration_service,
    build_source_evidence_service,
    build_source_search_service,
    build_teaching_service,
    build_user_service,
)
from kms2.composition.source_learning import build_source_learning_graph
from kms2.composition.source_pedagogical import (
    build_source_pedagogical_graph,
)
from kms2.composition.source_processing import build_source_processing_graph
from kms2.composition.source_semantic import build_source_semantic_graph
from kms2.composition.teaching import build_teaching_graph
from kms2.config.inference import StageInferenceSettings
from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.core.model.knowledge.proposition import PropositionRecord
from kms2.core.model.knowledge.reasoning import ReasoningStepRecord
from kms2.core.model.knowledge.retrieval import PropositionSearchHit
from kms2.core.model.learning import (
    DeckCard,
    DueDeckCard,
    ReviewEvent,
    ReviewRating,
    ReviewReceipt,
)
from kms2.core.model.source import Source
from kms2.core.model.source_evidence import (
    SourceImage,
    SourcePassage,
    SourceProcedurePassage,
    SourceProcedureSearchHit,
    SourceSearchHit,
    SourceStatementPassage,
    SourceStatementSearchHit,
)
from kms2.core.model.source_processing.admission import (
    ResearchSourceMigrationResult,
)
from kms2.core.model.source_processing.input import PDFSourceInput
from kms2.core.model.teaching.evidence import (
    TeachingCitationMedia,
    TeachingCitationView,
    TeachingEvidenceReference,
)
from kms2.core.model.teaching.observation import LearnerObservationView
from kms2.core.model.teaching.session import (
    TeachingSession,
    TeachingSessionView,
    TeachingStatus,
)
from kms2.core.model.teaching.workflow import TeachingTurnRequest
from kms2.core.model.user import Deck, User
from kms2.core.reranking import RerankerClient
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.inference.chatgpt import (
    ensure_chatgpt_authenticated,
)
from kms2.inference.chatgpt import (
    login_chatgpt as _login_chatgpt,
)
from kms2.inference.token_counting import InferenceTokenizers
from kms2.local_models.runtime import LocalModelRuntime
from kms2.train.recorder import Recorder


@dataclass(frozen=True, slots=True)
class SourceProcessingStageResult:
    """Counts materialized by one complete source processing stage."""

    source: Source
    ocr_page_count: int
    corrected_page_count: int
    formatted_page_count: int
    text_seam_page_count: int
    image_seam_page_count: int
    image_description_page_count: int
    embedded_page_count: int


@dataclass(frozen=True, slots=True)
class SourcePedagogicalStageResult:
    """Counts persisted by one complete source pedagogical stage."""

    split_page_count: int
    instruction_count: int
    exercise_component_count: int
    pedagogical_component_count: int
    statement_count: int
    procedure_count: int
    reembedded_block_count: int


@dataclass(frozen=True, slots=True)
class SourceSemanticStageResult:
    """Counts persisted by one complete source semantic stage."""

    source_statement_hub_count: int
    source_procedure_hub_count: int
    source_entity_count: int
    source_entity_description_count: int
    source_entity_hub_count: int


@dataclass(frozen=True, slots=True)
class SourceLearningStageResult:
    """Counts persisted by one complete source-learning stage."""

    atomic_flashcard_count: int
    coherent_flashcard_count: int


@dataclass(frozen=True, slots=True)
class SourcePipelineResult:
    """Summary of source processing, pedagogical, and semantic stages."""

    source_processing_stage: SourceProcessingStageResult
    source_pedagogical_stage: SourcePedagogicalStageResult
    source_semantic_stage: SourceSemanticStageResult


@dataclass(frozen=True, slots=True)
class _ApplicationResources:
    """Resources shared by the stages of one application invocation."""

    local_models: LocalModelRuntime
    embedding_client: EmbeddingClient | None
    tokenizers: InferenceTokenizers
    database: DatabaseClient
    recorder: Recorder | None
    reranker_client: RerankerClient | None


@asynccontextmanager
async def _application_resources(
    settings: Settings,
    *,
    embeddings_required: bool = True,
    reranking_required: bool = False,
) -> AsyncIterator[_ApplicationResources]:
    """Own invocation resources and acquire the reranker only when requested."""
    database = DatabaseClient(settings.database)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = InferenceTokenizers(settings)
    try:
        if embeddings_required:
            await schema.ensure_schema(
                database.session,
                embedding_dimension=embedding_dimension(settings),
            )
        else:
            await schema.ensure_structural_schema(database.session)
        async with LocalModelRuntime(
            settings.local_models,
            embedding_token_counter=tokenizers.embedding,
        ) as local_models:
            embedding_client = (
                build_embedding_client(
                    settings, local_models, tokenizers.selected_embedding
                )
                if embeddings_required
                else None
            )
            async with AsyncExitStack() as stack:
                reranker_client = (
                    await stack.enter_async_context(
                        reranker_resources(
                            settings,
                            local_models,
                            tokenizers.reranker,
                        )
                    )
                    if reranking_required
                    else None
                )
                yield _ApplicationResources(
                    local_models=local_models,
                    embedding_client=embedding_client,
                    tokenizers=tokenizers,
                    database=database,
                    recorder=recorder,
                    reranker_client=reranker_client,
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


def _source_processing_stage_result(
    final_state: dict[str, object],
) -> SourceProcessingStageResult:
    """Convert the final source processing graph state to its public result."""
    return SourceProcessingStageResult(
        source=final_state['source'],
        ocr_page_count=len(final_state['ocr_pages']),
        corrected_page_count=len(final_state['corrected_pages']),
        formatted_page_count=len(final_state['formatted_pages']),
        text_seam_page_count=len(final_state['text_seam_pages']),
        image_seam_page_count=len(final_state['image_seam_pages']),
        image_description_page_count=len(final_state['image_described_pages']),
        embedded_page_count=len(final_state['embedded_pages']),
    )


def _source_pedagogical_stage_result(
    final_state: dict[str, object],
) -> SourcePedagogicalStageResult:
    """Convert the final source pedagogical graph state to its result."""
    return SourcePedagogicalStageResult(
        split_page_count=len(final_state['split_pages']),
        instruction_count=len(final_state['instructions']),
        exercise_component_count=len(final_state['exercise_components']),
        pedagogical_component_count=len(final_state['pedagogical_components']),
        statement_count=len(final_state['statements']),
        procedure_count=len(final_state['procedures']),
        reembedded_block_count=final_state['reembedded_block_count'],
    )


def _source_semantic_stage_result(
    final_state: dict[str, object],
) -> SourceSemanticStageResult:
    """Convert the final source semantic graph state to its public result."""
    return SourceSemanticStageResult(
        source_statement_hub_count=final_state['source_statement_hub_count'],
        source_procedure_hub_count=final_state['source_procedure_hub_count'],
        source_entity_count=len(final_state['source_entities']),
        source_entity_description_count=final_state[
            'source_entity_description_persisted_count'
        ],
        source_entity_hub_count=final_state['source_entity_hub_count'],
    )


def _source_learning_stage_result(
    final_state: dict[str, object],
) -> SourceLearningStageResult:
    """Convert the final source-learning graph state to its public result."""
    return SourceLearningStageResult(
        atomic_flashcard_count=final_state['atomic_flashcard_count'],
        coherent_flashcard_count=final_state['coherent_flashcard_count'],
    )


async def list_users(settings: Settings) -> list[User]:
    """Return users for terminal onboarding and selection."""
    database = DatabaseClient(settings.database)
    try:
        return await build_user_service(database).list_users()
    finally:
        await database.close()


async def create_user(settings: Settings, name: str) -> User:
    """Create a user after ensuring only structural schema."""
    database = DatabaseClient(settings.database)
    try:
        await schema.ensure_structural_schema(database.session)
        return await build_user_service(database).create_user(name)
    finally:
        await database.close()


async def list_unowned_sources(settings: Settings) -> list[Source]:
    """Return legacy sources available for explicit adoption."""
    database = DatabaseClient(settings.database)
    try:
        return await build_user_service(database).list_unowned_sources()
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
        await build_user_service(database).adopt_source(
            user_uuid,
            source_uuid,
        )
    finally:
        await database.close()


async def list_sources(settings: Settings, user_uuid: str) -> list[Source]:
    """Return persisted sources owned by one user."""
    database = DatabaseClient(settings.database)
    try:
        return await build_user_service(database).list_sources(user_uuid)
    finally:
        await database.close()


async def search_sources(
    settings: Settings,
    user_uuid: str,
    query: str,
    *,
    source_uuids: list[str] | None = None,
) -> list[SourceSearchHit]:
    """Search a user's source blocks with the configured embedding and reranker."""

    if source_uuids == []:
        return []

    async with _application_resources(
        settings,
        reranking_required=True,
    ) as resources:
        assert resources.embedding_client is not None
        assert resources.reranker_client is not None
        service = build_source_search_service(
            resources.database,
            embedding_client=resources.embedding_client,
            reranker_client=resources.reranker_client,
            settings=settings.source_search,
        )
        return await service.search_sources(
            user_uuid,
            query,
            source_uuids=source_uuids,
        )


async def read_source_passages(
    settings: Settings,
    user_uuid: str,
    block_uuids: list[str],
    *,
    expand_memberships: bool = True,
    before: int = 0,
    after: int = 0,
) -> list[SourcePassage]:
    """Read authorized source passages and their direct evidence groups."""
    async with _database_resources(settings) as database:
        service = build_source_evidence_service(database)
        return await service.read_passages(
            user_uuid,
            block_uuids,
            expand_memberships=expand_memberships,
            before=before,
            after=after,
        )


async def read_source_images(
    settings: Settings,
    user_uuid: str,
    asset_uuids: list[str],
) -> list[SourceImage]:
    """Read original image bytes attached to a user's source blocks."""
    async with _database_resources(settings) as database:
        service = build_source_evidence_service(database)
        return await service.read_images(user_uuid, asset_uuids)


async def search_source_statements(
    settings: Settings,
    user_uuid: str,
    query: str,
    *,
    source_uuids: list[str] | None = None,
    is_exercise: bool | None = None,
) -> list[SourceStatementSearchHit]:
    """Search reconstructed statements in the user's eligible sources."""
    if source_uuids == []:
        return []
    async with _application_resources(
        settings, reranking_required=True
    ) as resources:
        assert resources.embedding_client is not None
        assert resources.reranker_client is not None
        service = build_source_search_service(
            resources.database,
            embedding_client=resources.embedding_client,
            reranker_client=resources.reranker_client,
            settings=settings.source_search,
        )
        return await service.search_statements(
            user_uuid,
            query,
            source_uuids=source_uuids,
            is_exercise=is_exercise,
        )


async def search_source_procedures(
    settings: Settings,
    user_uuid: str,
    query: str,
    *,
    source_uuids: list[str] | None = None,
) -> list[SourceProcedureSearchHit]:
    """Search reconstructed procedures without mixing material-type scores."""
    if source_uuids == []:
        return []
    async with _application_resources(
        settings, reranking_required=True
    ) as resources:
        assert resources.embedding_client is not None
        assert resources.reranker_client is not None
        service = build_source_search_service(
            resources.database,
            embedding_client=resources.embedding_client,
            reranker_client=resources.reranker_client,
            settings=settings.source_search,
        )
        return await service.search_procedures(
            user_uuid, query, source_uuids=source_uuids
        )


async def read_source_statements(
    settings: Settings, user_uuid: str, statement_uuids: list[str]
) -> list[SourceStatementPassage]:
    """Read statements with their original member evidence and instructions."""
    async with _database_resources(settings) as database:
        return await build_source_evidence_service(database).read_statements(
            user_uuid, statement_uuids
        )


async def read_source_procedures(
    settings: Settings, user_uuid: str, procedure_uuids: list[str]
) -> list[SourceProcedurePassage]:
    """Read procedures with their ordinary source passages."""
    async with _database_resources(settings) as database:
        return await build_source_evidence_service(database).read_procedures(
            user_uuid, procedure_uuids
        )


async def search_propositions(
    settings: Settings, user_uuid: str, query: str
) -> list[PropositionSearchHit]:
    """Search canonical claims across the user's learned knowledge."""
    async with _application_resources(
        settings, reranking_required=True
    ) as resources:
        assert resources.embedding_client is not None
        assert resources.reranker_client is not None
        service = build_knowledge_search_service(
            resources.database,
            embedding_client=resources.embedding_client,
            reranker_client=resources.reranker_client,
            settings=settings.knowledge.search,
        )
        return await service.search_propositions(user_uuid, query)


async def read_propositions(
    settings: Settings, user_uuid: str, proposition_uuids: list[str]
) -> list[PropositionRecord]:
    """Read claims and their bounded per-turn support references."""
    async with _database_resources(settings) as database:
        return await build_knowledge_service(database).read_propositions(
            user_uuid, proposition_uuids
        )


async def search_learner_observations(
    settings: Settings,
    user_uuid: str,
    query: str,
) -> list[LearnerObservationView]:
    """Search user-owned learner evidence with embedding and reranking."""
    async with _application_resources(
        settings,
        reranking_required=True,
    ) as resources:
        assert resources.embedding_client is not None
        assert resources.reranker_client is not None
        search_service = build_learner_observation_search_service(
            resources.database,
            embedding_client=resources.embedding_client,
            reranker_client=resources.reranker_client,
            settings=settings.teaching.learner_history,
        )
        view_service = build_learner_observation_service(resources.database)
        hits = await search_service.search(user_uuid, query)
        return [view_service.view(hit.record) for hit in hits]


async def read_learner_observations(
    settings: Settings,
    user_uuid: str,
    observation_uuids: list[str],
) -> list[LearnerObservationView]:
    """Read learner-safe views of selected user-owned observations."""
    async with _database_resources(settings) as database:
        service = build_learner_observation_service(database)
        records = await service.read(user_uuid, observation_uuids)
        return [service.view(record) for record in records]


async def read_reasoning_steps(
    settings: Settings, user_uuid: str, step_uuids: list[str]
) -> list[ReasoningStepRecord]:
    """Read selected joint-premise arguments without recursive expansion."""
    async with _database_resources(settings) as database:
        return await build_knowledge_service(database).read_reasoning_steps(
            user_uuid, step_uuids
        )


async def migrate_research_sources(
    settings: Settings, user_uuid: str
) -> ResearchSourceMigrationResult:
    """Move this user's retained research citations into ordinary sources."""
    async with _application_resources(
        settings,
        embeddings_required=True,
        reranking_required=False,
    ) as resources:
        assert resources.embedding_client is not None
        service = build_research_source_migration_service(
            resources.database,
            embedding_client=resources.embedding_client,
            storage_root=settings.research.source_directory,
        )
        return await service.migrate(user_uuid)


async def finalize_research_source_migration(settings: Settings) -> None:
    """Drop the legacy evidence constraint after every owner is migrated."""
    async with _database_resources(settings) as database:
        finalizer = build_research_source_migration_finalizer(database)
        await finalizer.finalize()


async def _run_source_processing_stage(
    settings: Settings,
    resources: _ApplicationResources,
    pdf_path: str,
    pages: list[int] | None,
) -> SourceProcessingStageResult:
    """Run the source processing graph and return its stage summary."""
    await _preflight_chatgpt(settings, settings.source_processing)
    source = Source(key=Path(pdf_path).name)
    initial_state = {
        'input': PDFSourceInput(pdf_path=pdf_path, pages=pages),
        'source': source,
    }
    assert resources.embedding_client is not None
    graph = build_source_processing_graph(
        settings,
        resources.local_models,
        resources.database,
        embedding_client=resources.embedding_client,
        tokenizers=resources.tokenizers,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke(initial_state)
    return _source_processing_stage_result(final_state)


async def _run_source_pedagogical_stage(
    settings: Settings,
    resources: _ApplicationResources,
    source_uuid: str,
) -> SourcePedagogicalStageResult:
    """Run one complete source pedagogical graph."""
    await _preflight_chatgpt(settings, settings.source_pedagogical)
    assert resources.embedding_client is not None
    graph = build_source_pedagogical_graph(
        settings,
        resources.local_models,
        resources.database,
        embedding_client=resources.embedding_client,
        tokenizers=resources.tokenizers,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return _source_pedagogical_stage_result(final_state)


async def _run_source_semantic_stage(
    settings: Settings,
    resources: _ApplicationResources,
    source_uuid: str,
) -> SourceSemanticStageResult:
    """Run one complete source semantic graph in dependency order."""
    await _preflight_chatgpt(settings, settings.source_semantic)
    assert resources.embedding_client is not None
    assert resources.reranker_client is not None
    graph = build_source_semantic_graph(
        settings,
        resources.local_models,
        resources.database,
        embedding_client=resources.embedding_client,
        reranker_client=resources.reranker_client,
        tokenizers=resources.tokenizers,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return _source_semantic_stage_result(final_state)


async def _run_source_learning_stage(
    settings: Settings,
    resources: _ApplicationResources,
    source_uuid: str,
) -> SourceLearningStageResult:
    """Run one complete source-learning graph in dependency order."""
    await _preflight_chatgpt(settings, settings.source_learning)
    graph = build_source_learning_graph(
        settings,
        resources.local_models,
        resources.database,
        tokenizers=resources.tokenizers,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return _source_learning_stage_result(final_state)


async def ingest_source(
    settings: Settings,
    user_uuid: str,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceProcessingStageResult:
    """Process and attach a new source to its user."""
    async with _application_resources(
        settings, reranking_required=False
    ) as resources:
        result = await _run_source_processing_stage(
            settings,
            resources,
            pdf_path,
            pages,
        )
        await build_user_service(resources.database).attach_new_source(
            user_uuid,
            result.source.uuid,
        )
        return result


async def run_source_pedagogical_stage(
    settings: Settings,
    user_uuid: str,
    source_uuid: str,
) -> SourcePedagogicalStageResult:
    """Run source pedagogy only for a source owned by the user."""
    sources = await list_sources(settings, user_uuid)
    if source_uuid not in {source.uuid for source in sources}:
        raise ValueError(
            f'Source {source_uuid} is not owned by user {user_uuid}'
        )
    async with _application_resources(
        settings, reranking_required=False
    ) as resources:
        return await _run_source_pedagogical_stage(
            settings,
            resources,
            source_uuid,
        )


async def run_source_semantic_stage(
    settings: Settings,
    user_uuid: str,
    source_uuid: str,
) -> SourceSemanticStageResult:
    """Run source semantics only for a source owned by the user."""
    sources = await list_sources(settings, user_uuid)
    if source_uuid not in {source.uuid for source in sources}:
        raise ValueError(
            f'Source {source_uuid} is not owned by user {user_uuid}'
        )
    async with _application_resources(
        settings, reranking_required=True
    ) as resources:
        return await _run_source_semantic_stage(
            settings,
            resources,
            source_uuid,
        )


async def run_source_learning_stage(
    settings: Settings,
    user_uuid: str,
    source_uuid: str,
) -> SourceLearningStageResult:
    """Run source learning only for a source owned by the user."""
    sources = await list_sources(settings, user_uuid)
    if source_uuid not in {source.uuid for source in sources}:
        raise ValueError(
            f'Source {source_uuid} is not owned by user {user_uuid}'
        )
    async with _application_resources(
        settings,
        embeddings_required=False,
        reranking_required=False,
    ) as resources:
        return await _run_source_learning_stage(
            settings,
            resources,
            source_uuid,
        )


async def ingest_and_run_source_semantic_stage(
    settings: Settings,
    user_uuid: str,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourcePipelineResult:
    """Ingest, attach, then run pedagogy and semantics in one session."""
    async with _application_resources(
        settings, reranking_required=True
    ) as resources:
        source_processing_stage = await _run_source_processing_stage(
            settings,
            resources,
            pdf_path,
            pages,
        )
        await build_user_service(resources.database).attach_new_source(
            user_uuid,
            source_processing_stage.source.uuid,
        )
        source_pedagogical_stage = await _run_source_pedagogical_stage(
            settings,
            resources,
            source_processing_stage.source.uuid,
        )
        source_semantic_stage = await _run_source_semantic_stage(
            settings,
            resources,
            source_processing_stage.source.uuid,
        )

    return SourcePipelineResult(
        source_processing_stage=source_processing_stage,
        source_pedagogical_stage=source_pedagogical_stage,
        source_semantic_stage=source_semantic_stage,
    )


async def start_teaching_session(
    settings: Settings,
    user_uuid: str,
    goal: str,
) -> TeachingSessionView:
    """Start one goal-driven teaching session without inference resources."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(database).start_session(
            user_uuid,
            goal,
        )


async def list_teaching_sessions(
    settings: Settings,
    user_uuid: str,
) -> list[TeachingSession]:
    """List a user's teaching sessions without loading their transcripts."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(database).list_sessions(user_uuid)


async def load_teaching_session(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
) -> TeachingSessionView:
    """Load one goal-driven teaching session for display resumption."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(database).load_session(
            user_uuid,
            session_uuid,
        )


async def read_teaching_citations(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    references: list[TeachingEvidenceReference],
) -> list[TeachingCitationView]:
    """Read safe evidence views cited in one owned teaching session."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(database).read_teaching_citations(
            user_uuid, session_uuid, references
        )


async def read_teaching_citation_media(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    asset_uuid: str,
) -> TeachingCitationMedia:
    """Read an original image cited in one owned teaching session."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(
            database
        ).read_teaching_citation_media(user_uuid, session_uuid, asset_uuid)


async def submit_teaching_response(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    content: str,
    *,
    expected_revision: int,
) -> TeachingSessionView:
    """Append one learner response through the teaching service."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(database).submit_response(
            user_uuid,
            session_uuid,
            content,
            expected_revision=expected_revision,
        )


async def run_teaching_turn(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    *,
    expected_revision: int,
    source_uuids: list[str] | None = None,
) -> TeachingSessionView:
    """Run one bounded teacher turn for an awaiting-teacher session.

    The session is ownership-checked before inference resources are acquired.
    Source identifiers, when provided, restrict local search to that allowlist.
    """
    async with _database_resources(settings) as database:
        teaching = build_teaching_service(database)
        snapshot = await teaching.load_snapshot(user_uuid, session_uuid)
        if (
            snapshot.session.status != TeachingStatus.AWAITING_TEACHER
            or snapshot.session.revision != expected_revision
        ):
            raise ValueError(
                'Teaching session is not awaiting teacher at expected revision'
            )

    await _preflight_chatgpt(
        settings, settings.knowledge.extraction, settings.knowledge.identity
    )
    if settings.research.enabled:
        await _preflight_chatgpt(
            settings,
            settings.research.inference,
            settings.research.formatting.inference,
            settings.source_processing,
        )
    async with AsyncExitStack() as stack:
        research = (
            await stack.enter_async_context(open_research_resources(settings))
            if settings.research.enabled
            else None
        )
        resources = await stack.enter_async_context(
            _application_resources(
                settings,
                reranking_required=True,
            )
        )
        assert resources.embedding_client is not None
        assert resources.reranker_client is not None
        graph = build_teaching_graph(
            settings,
            resources.local_models,
            resources.database,
            embedding_client=resources.embedding_client,
            reranker_client=resources.reranker_client,
            tokenizers=resources.tokenizers,
            research_resources=research,
            recorder=resources.recorder,
        )
        return await graph.run(
            TeachingTurnRequest(
                user_uuid=user_uuid,
                session_uuid=session_uuid,
                expected_revision=expected_revision,
                snapshot=snapshot,
                source_uuids=(
                    None if source_uuids is None else list(source_uuids)
                ),
            )
        )


async def end_teaching_session(
    settings: Settings,
    user_uuid: str,
    session_uuid: str,
    *,
    expected_revision: int,
) -> TeachingSessionView:
    """End one teaching session while retaining its interaction history."""
    async with _database_resources(settings) as database:
        return await build_teaching_service(database).end_session(
            user_uuid,
            session_uuid,
            expected_revision=expected_revision,
        )


async def list_decks(settings: Settings, user_uuid: str) -> list[Deck]:
    """List decks owned by one user without starting local models."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).list_decks(user_uuid)


async def create_deck(
    settings: Settings,
    user_uuid: str,
    name: str,
) -> Deck:
    """Create one deck for a user."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).create_deck(
            user_uuid,
            name,
        )


async def list_assignable_cards(
    settings: Settings,
    user_uuid: str,
) -> list[DeckCard]:
    """List source cards the user may add to decks."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).list_assignable_cards(
            user_uuid
        )


async def add_card_to_deck(
    settings: Settings,
    user_uuid: str,
    deck_uuid: str,
    card_uuid: str,
) -> DeckCard:
    """Add one source card to a user's deck."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).add_card_to_deck(
            user_uuid,
            deck_uuid,
            card_uuid,
        )


async def remove_card_from_deck(
    settings: Settings,
    user_uuid: str,
    deck_uuid: str,
    card_uuid: str,
) -> None:
    """Remove one card from one deck."""
    async with _database_resources(settings) as database:
        await build_learning_service(database).remove_card_from_deck(
            user_uuid,
            deck_uuid,
            card_uuid,
        )


async def list_deck_cards(
    settings: Settings,
    user_uuid: str,
    deck_uuid: str,
) -> list[DeckCard]:
    """List cards currently in one deck."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).list_deck_cards(
            user_uuid,
            deck_uuid,
        )


async def list_due_deck_cards(
    settings: Settings,
    user_uuid: str,
    deck_uuid: str,
    now: datetime,
) -> list[DueDeckCard]:
    """List due cards in one deck at the supplied UTC time."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).list_due_deck_cards(
            user_uuid,
            deck_uuid,
            now,
        )


async def review_card(
    settings: Settings,
    user_uuid: str,
    deck_uuid: str,
    card_uuid: str,
    rating: ReviewRating,
    review_datetime: datetime,
    review_duration: int | None = None,
) -> ReviewReceipt:
    """Submit one card rating through the learning service."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).review_card(
            user_uuid,
            deck_uuid,
            card_uuid,
            rating,
            review_datetime,
            review_duration,
        )


async def list_review_events(
    settings: Settings,
    user_uuid: str,
    card_uuid: str,
) -> list[ReviewEvent]:
    """List immutable review history for one user/card association."""
    async with _database_resources(settings) as database:
        return await build_learning_service(database).list_review_events(
            user_uuid,
            card_uuid,
        )
