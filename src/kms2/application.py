"""Application entry points for running KMS2 pipelines."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kms2.composition.global_semantic import build_global_semantic_graph
from kms2.composition.services import build_learning_service, build_user_service
from kms2.composition.source_learning import build_source_learning_graph
from kms2.composition.source_processing import build_source_processing_graph
from kms2.composition.source_semantic import build_source_semantic_graph
from kms2.config.settings import Settings
from kms2.core.model.learning import (
    DeckCard,
    DueDeckCard,
    ReviewEvent,
    ReviewRating,
    ReviewReceipt,
)
from kms2.core.model.source import Source
from kms2.core.model.user import Deck, User
from kms2.database import schema
from kms2.database.client import DatabaseClient
from kms2.local_models.runtime import LocalModelRuntime
from kms2.local_models.token_counting import LocalTokenizers
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
    split_page_count: int
    instruction_count: int
    exercise_component_count: int
    pedagogical_component_count: int
    statement_count: int
    procedure_count: int
    embedded_page_count: int


@dataclass(frozen=True, slots=True)
class SourceSemanticStageResult:
    """Counts persisted by one complete source semantic stage."""

    source_fact_count: int
    triplet_count: int
    source_entity_description_count: int
    source_event_description_count: int
    source_predicate_description_count: int
    source_statement_description_count: int
    source_procedure_description_count: int
    source_entity_hub_count: int
    source_event_hub_count: int
    source_predicate_hub_count: int
    source_triplet_hub_count: int
    source_statement_hub_count: int
    source_procedure_hub_count: int


@dataclass(frozen=True, slots=True)
class SourceLearningStageResult:
    """Counts persisted by one complete source-learning stage."""

    entity_learning_fact_count: int
    entity_flashcard_count: int
    event_learning_fact_count: int
    event_flashcard_count: int
    predicate_learning_fact_count: int
    predicate_flashcard_count: int
    triplet_learning_fact_count: int
    triplet_flashcard_count: int


@dataclass(frozen=True, slots=True)
class GlobalSemanticStageResult:
    """Counts persisted by one complete global semantic stage."""

    global_entity_hub_count: int
    global_event_hub_count: int
    global_predicate_hub_count: int
    global_triplet_count: int
    global_triplet_hub_count: int
    global_statement_hub_count: int
    global_procedure_hub_count: int


@dataclass(frozen=True, slots=True)
class SourcePipelineResult:
    """Summary of source processing and source semantic stages."""

    source_processing_stage: SourceProcessingStageResult
    source_semantic_stage: SourceSemanticStageResult


@dataclass(frozen=True, slots=True)
class _ApplicationResources:
    """Resources shared by the stages of one application invocation."""

    local_models: LocalModelRuntime
    tokenizers: LocalTokenizers
    database: DatabaseClient
    recorder: Recorder | None


@asynccontextmanager
async def _application_resources(
    settings: Settings,
) -> AsyncIterator[_ApplicationResources]:
    """Own database, local-model, and optional recorder resources."""
    database = DatabaseClient(settings.database)
    recorder = (
        None
        if settings.training.examples_directory is None
        else Recorder(settings.training.examples_directory)
    )
    tokenizers = LocalTokenizers(settings.local_models)
    try:
        await schema.ensure_schema(
            database.session,
            embedding_dimension=settings.local_models.embedding.model.dimension,
        )
        async with LocalModelRuntime(
            settings.local_models,
            embedding_token_counter=tokenizers.embedding,
        ) as local_models:
            yield _ApplicationResources(
                local_models, tokenizers, database, recorder
            )
    finally:
        await database.close()


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
        split_page_count=len(final_state['split_pages']),
        instruction_count=len(final_state['instructions']),
        exercise_component_count=len(final_state['exercise_components']),
        pedagogical_component_count=len(final_state['pedagogical_components']),
        statement_count=len(final_state['statements']),
        procedure_count=len(final_state['procedures']),
        embedded_page_count=len(final_state['embedded_pages']),
    )


def _source_semantic_stage_result(
    final_state: dict[str, object],
) -> SourceSemanticStageResult:
    """Convert the final source semantic graph state to its public result."""
    return SourceSemanticStageResult(
        source_fact_count=len(final_state['source_facts']),
        triplet_count=len(final_state['triplet_occurrences']),
        source_entity_description_count=final_state[
            'source_entity_description_persisted_count'
        ],
        source_event_description_count=final_state[
            'source_event_description_persisted_count'
        ],
        source_predicate_description_count=final_state[
            'source_predicate_description_persisted_count'
        ],
        source_statement_description_count=final_state[
            'source_statement_description_persisted_count'
        ],
        source_procedure_description_count=final_state[
            'source_procedure_description_persisted_count'
        ],
        source_entity_hub_count=final_state['source_entity_hub_count'],
        source_event_hub_count=final_state['source_event_hub_count'],
        source_predicate_hub_count=final_state['source_predicate_hub_count'],
        source_triplet_hub_count=final_state['source_triplet_hub_count'],
        source_statement_hub_count=final_state['source_statement_hub_count'],
        source_procedure_hub_count=final_state['source_procedure_hub_count'],
    )


def _source_learning_stage_result(
    final_state: dict[str, object],
) -> SourceLearningStageResult:
    """Convert the final source-learning graph state to its public result."""
    return SourceLearningStageResult(
        entity_learning_fact_count=final_state['entity_learning_fact_count'],
        entity_flashcard_count=final_state['entity_flashcard_count'],
        event_learning_fact_count=final_state['event_learning_fact_count'],
        event_flashcard_count=final_state['event_flashcard_count'],
        predicate_learning_fact_count=final_state[
            'predicate_learning_fact_count'
        ],
        predicate_flashcard_count=final_state['predicate_flashcard_count'],
        triplet_learning_fact_count=final_state['triplet_learning_fact_count'],
        triplet_flashcard_count=final_state['triplet_flashcard_count'],
    )


def _global_semantic_stage_result(
    final_state: dict[str, object],
) -> GlobalSemanticStageResult:
    """Convert the final global semantic graph state to its public result."""
    return GlobalSemanticStageResult(
        global_entity_hub_count=final_state['global_entity_hub_count'],
        global_event_hub_count=final_state['global_event_hub_count'],
        global_predicate_hub_count=final_state['global_predicate_hub_count'],
        global_triplet_count=final_state['global_triplet_count'],
        global_triplet_hub_count=final_state['global_triplet_hub_count'],
        global_statement_hub_count=final_state['global_statement_hub_count'],
        global_procedure_hub_count=final_state['global_procedure_hub_count'],
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


async def _run_source_processing_stage(
    settings: Settings,
    resources: _ApplicationResources,
    pdf_path: str,
    pages: list[int] | None,
) -> SourceProcessingStageResult:
    """Run the source processing graph and return its stage summary."""
    source = Source(key=Path(pdf_path).name)
    initial_state = {
        'pdf_path': pdf_path,
        'pages': pages,
        'source': source,
    }
    graph = build_source_processing_graph(
        settings,
        resources.local_models,
        resources.database,
        tokenizers=resources.tokenizers,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke(initial_state)
    return _source_processing_stage_result(final_state)


async def _run_source_semantic_stage(
    settings: Settings,
    resources: _ApplicationResources,
    source_uuid: str,
) -> SourceSemanticStageResult:
    """Run one complete source semantic graph in dependency order."""
    graph = build_source_semantic_graph(
        settings,
        resources.local_models,
        resources.database,
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
    graph = build_source_learning_graph(
        settings,
        resources.local_models,
        resources.database,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke({'source_uuid': source_uuid})
    return _source_learning_stage_result(final_state)


async def _run_global_semantic_stage(
    settings: Settings,
    resources: _ApplicationResources,
) -> GlobalSemanticStageResult:
    """Run the global semantic graph over persisted source hubs."""
    graph = build_global_semantic_graph(
        settings,
        resources.local_models,
        resources.database,
        tokenizers=resources.tokenizers,
        recorder=resources.recorder,
    ).build_graph()
    final_state = await graph.ainvoke({})
    return _global_semantic_stage_result(final_state)


async def ingest_source(
    settings: Settings,
    user_uuid: str,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourceProcessingStageResult:
    """Process and attach a new source to its user."""
    async with _application_resources(settings) as resources:
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
    async with _application_resources(settings) as resources:
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
    async with _application_resources(settings) as resources:
        return await _run_source_learning_stage(
            settings,
            resources,
            source_uuid,
        )


async def run_global_semantic_stage(
    settings: Settings,
) -> GlobalSemanticStageResult:
    """Run global semantic consolidation over all persisted source hubs."""
    async with _application_resources(settings) as resources:
        return await _run_global_semantic_stage(settings, resources)


async def ingest_and_run_source_semantic_stage(
    settings: Settings,
    user_uuid: str,
    pdf_path: str,
    pages: list[int] | None = None,
) -> SourcePipelineResult:
    """Ingest, attach, then run semantics in one application session."""
    async with _application_resources(settings) as resources:
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
        source_semantic_stage = await _run_source_semantic_stage(
            settings,
            resources,
            source_processing_stage.source.uuid,
        )

    return SourcePipelineResult(
        source_processing_stage=source_processing_stage,
        source_semantic_stage=source_semantic_stage,
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
