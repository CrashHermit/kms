"""Interactive terminal entry point for KMS2 source processing."""

import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from kms2.application import (
    add_card_to_deck,
    adopt_source,
    create_deck,
    create_user,
    ingest_source,
    list_assignable_cards,
    list_deck_cards,
    list_decks,
    list_due_deck_cards,
    list_review_events,
    list_sources,
    list_unowned_sources,
    list_users,
    remove_card_from_deck,
    review_card,
    run_global_semantic_stage,
    run_source_learning_stage,
    run_source_semantic_stage,
)
from kms2.config.settings import Settings
from kms2.core.model.learning import DeckCard, ReviewRating
from kms2.core.model.source import Source
from kms2.core.model.user import Deck, User

PDF_DIRECTORY = Path('pdfs')
NEW_SOURCE_OPTION = 'Run Source Processing for a new PDF'
EXISTING_SOURCE_OPTION = 'Run Source Semantic for an existing source'
SOURCE_LEARNING_OPTION = 'Run Source Learning for an existing source'
ADOPT_SOURCE_OPTION = 'Adopt existing source'
GLOBAL_SEMANTIC_OPTION = 'Run Global Semantic'
CREATE_DECK_OPTION = 'Create deck'
ADD_CARD_OPTION = 'Add source card to deck'
REMOVE_CARD_OPTION = 'Remove card from deck'
REVIEW_CARDS_OPTION = 'Review due cards'
REVIEW_HISTORY_OPTION = 'Display card review history'
CREATE_USER_OPTION = 'Create user'

logger = logging.getLogger(__name__)


def _pdf_directory() -> Path:
    """Return the project-local directory used for source PDFs."""
    directory = Path.cwd() / PDF_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run() -> None:
    """Run the KMS2 TUI, exiting cleanly on Ctrl-C."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        _run_tui()
    except KeyboardInterrupt, asyncio.CancelledError:
        logger.info('Cancelled.')
        sys.exit(0)


def _run_tui() -> None:
    """Resolve a user, then prompt for one user-scoped operation."""
    settings = Settings()
    user = _select_user(settings)
    sources = _run_async(list_sources(settings, user.uuid))
    unowned_sources = _run_async(list_unowned_sources(settings))
    choices = [
        NEW_SOURCE_OPTION,
        CREATE_DECK_OPTION,
        ADD_CARD_OPTION,
        REMOVE_CARD_OPTION,
        REVIEW_CARDS_OPTION,
        REVIEW_HISTORY_OPTION,
    ]
    if sources:
        choices.extend([EXISTING_SOURCE_OPTION, SOURCE_LEARNING_OPTION])
    if unowned_sources:
        choices.append(ADOPT_SOURCE_OPTION)
    choices.append(GLOBAL_SEMANTIC_OPTION)
    action = inquirer.select(
        message='What would you like to do?',
        choices=choices,
    ).execute()
    if action == EXISTING_SOURCE_OPTION:
        _run_existing_source(settings, user, sources)
    elif action == SOURCE_LEARNING_OPTION:
        _run_existing_source_learning(settings, user, sources)
    elif action == ADOPT_SOURCE_OPTION:
        _adopt_existing_source(settings, user, unowned_sources)
    elif action == GLOBAL_SEMANTIC_OPTION:
        _run_global_semantic(settings)
    elif action == CREATE_DECK_OPTION:
        _create_user_deck(settings, user)
    elif action == ADD_CARD_OPTION:
        _add_card_to_user_deck(settings, user)
    elif action == REMOVE_CARD_OPTION:
        _remove_card_from_user_deck(settings, user)
    elif action == REVIEW_CARDS_OPTION:
        _review_due_cards(settings, user)
    elif action == REVIEW_HISTORY_OPTION:
        _display_review_history(settings, user)
    else:
        _run_new_source(settings, user)


def _select_user(settings: Settings) -> User:
    """Create the first user, reuse the sole user, or prompt among users."""
    users = _run_async(list_users(settings))
    if not users:
        return _create_user(settings)
    if len(users) == 1:
        return users[0]

    selection = inquirer.select(
        message='Select a user:',
        choices=[
            *[
                Choice(value=user, name=f'{user.name} ({user.uuid})')
                for user in users
            ],
            CREATE_USER_OPTION,
        ],
    ).execute()
    return (
        _create_user(settings) if selection == CREATE_USER_OPTION else selection
    )


def _create_user(settings: Settings) -> User:
    """Prompt for and persist a user display name."""
    name = inquirer.text(message='Name for the new user:').execute()
    return _run_async(create_user(settings, name))


def _run_async[T](awaitable: Awaitable[T]) -> T:
    """Run one application coroutine with graceful signal cancellation."""

    async def runner() -> T:
        task = asyncio.create_task(awaitable)
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, task.cancel)
            except NotImplementedError, RuntimeError:
                continue
            installed.append(signum)
        try:
            return await task
        finally:
            for signum in installed:
                loop.remove_signal_handler(signum)

    return asyncio.run(runner())


def _select_deck(settings: Settings, user: User, message: str) -> Deck:
    """Prompt for one deck owned by the selected user."""
    decks = _run_async(list_decks(settings, user.uuid))
    return inquirer.select(
        message=message,
        choices=[
            Choice(value=deck, name=f'{deck.name} ({deck.uuid})')
            for deck in decks
        ],
    ).execute()


def _select_card(
    cards: list[DeckCard],
    message: str,
) -> DeckCard:
    """Prompt for one card from a previously selected collection."""
    return inquirer.select(
        message=message,
        choices=[
            Choice(value=card, name=f'{card.question} ({card.card_uuid})')
            for card in cards
        ],
    ).execute()


def _create_user_deck(settings: Settings, user: User) -> None:
    """Prompt for and create one deck."""
    name = inquirer.text(message='Name for the new deck:').execute()
    deck = _run_async(create_deck(settings, user.uuid, name))
    logger.info('Created deck %s (%s).', deck.name, deck.uuid)


def _add_card_to_user_deck(settings: Settings, user: User) -> None:
    """Select an owned source card and add it to one deck."""
    deck = _select_deck(settings, user, 'Select the destination deck:')
    card = _select_card(
        _run_async(list_assignable_cards(settings, user.uuid)),
        'Select the source card:',
    )
    _run_async(add_card_to_deck(settings, user.uuid, deck.uuid, card.card_uuid))
    logger.info('Added card %s to deck %s.', card.card_uuid, deck.name)


def _remove_card_from_user_deck(settings: Settings, user: User) -> None:
    """Select and remove one card membership from one deck."""
    deck = _select_deck(settings, user, 'Select the deck:')
    card = _select_card(
        _run_async(list_deck_cards(settings, user.uuid, deck.uuid)),
        'Select the card to remove:',
    )
    _run_async(
        remove_card_from_deck(settings, user.uuid, deck.uuid, card.card_uuid)
    )
    logger.info('Removed card %s from deck %s.', card.card_uuid, deck.name)


def _review_due_cards(settings: Settings, user: User) -> None:
    """Review every card due in a selected deck."""
    deck = _select_deck(settings, user, 'Select the deck to review:')
    due_cards = _run_async(
        list_due_deck_cards(
            settings,
            user.uuid,
            deck.uuid,
            datetime.now(UTC),
        )
    )
    for card in due_cards:
        logger.info('Question: %s\\nAnswer: %s', card.question, card.answer)
        rating = inquirer.select(
            message='Rating:',
            choices=[
                Choice(value=choice, name=choice.value.title())
                for choice in ReviewRating
            ],
        ).execute()
        _run_async(
            review_card(
                settings,
                user.uuid,
                deck.uuid,
                card.card_uuid,
                rating,
                datetime.now(UTC),
            )
        )
    if not due_cards:
        logger.info('No cards are due in %s.', deck.name)


def _display_review_history(settings: Settings, user: User) -> None:
    """Display ordered review history for one card in one deck."""
    deck = _select_deck(settings, user, 'Select the deck:')
    card = _select_card(
        _run_async(list_deck_cards(settings, user.uuid, deck.uuid)),
        'Select the card:',
    )
    events = _run_async(list_review_events(settings, user.uuid, card.card_uuid))
    for event in events:
        logger.info(
            'Review %d: %s',
            event.review_index,
            event.fsrs_review_log_json,
        )


def _log_source_semantic_completion(source: Source, semantic) -> None:
    """Log all counts persisted by one source semantic stage."""
    logger.info(
        'Done: source %s (%s), Source Semantic persisted %d source fact(s), '
        '%d triplet(s); descriptions: %d entity, %d event, %d predicate, '
        '%d statement, %d procedure; hubs: %d entity, %d event, %d predicate, '
        '%d triplet, %d statement, %d procedure.',
        source.uuid,
        source.key,
        semantic.source_fact_count,
        semantic.triplet_count,
        semantic.source_entity_description_count,
        semantic.source_event_description_count,
        semantic.source_predicate_description_count,
        semantic.source_statement_description_count,
        semantic.source_procedure_description_count,
        semantic.source_entity_hub_count,
        semantic.source_event_hub_count,
        semantic.source_predicate_hub_count,
        semantic.source_triplet_hub_count,
        semantic.source_statement_hub_count,
        semantic.source_procedure_hub_count,
    )


def _run_new_source(settings: Settings, user: User) -> None:
    """Run Source Processing for a newly selected PDF."""
    pdf_path = inquirer.filepath(
        message='Select the PDF to process:',
        default=str(_pdf_directory()),
        only_files=True,
    ).execute()
    raw_pages = inquirer.text(
        message=(
            'Limit to pages (0-based, comma-separated, or leave empty for all):'
        ),
        default='',
    ).execute()
    pages = (
        None
        if not raw_pages
        else [int(page.strip()) for page in raw_pages.split(',')]
    )
    result = _run_async(ingest_source(settings, user.uuid, pdf_path, pages))
    logger.info(
        'Done: source %s (%s), Source Processing ingested %d page(s).',
        result.source.uuid,
        result.source.key,
        result.split_page_count,
    )


def _run_existing_source(
    settings: Settings,
    user: User,
    sources: list[Source],
) -> None:
    """Run Source Semantic for a source owned by the selected user."""
    source = inquirer.select(
        message='Select the source for Source Semantic:',
        choices=[
            Choice(
                value=source,
                name=f'{source.key} ({source.uuid})',
            )
            for source in sources
        ],
    ).execute()
    proceed = inquirer.confirm(
        message='Run Source Semantic on the selected source?',
        default=True,
    ).execute()

    if not proceed:
        logger.info('Cancelled.')
        return

    semantic = _run_async(
        run_source_semantic_stage(settings, user.uuid, source.uuid)
    )
    _log_source_semantic_completion(source, semantic)


def _log_source_learning_completion(source: Source, learning) -> None:
    """Log all counts persisted by one source-learning stage."""
    logger.info(
        'Done: source %s (%s), Source Learning persisted %d/%d entity '
        'learning fact(s)/card(s), %d/%d event, %d/%d predicate, and %d/%d '
        'triplet learning fact(s)/card(s).',
        source.uuid,
        source.key,
        learning.entity_learning_fact_count,
        learning.entity_flashcard_count,
        learning.event_learning_fact_count,
        learning.event_flashcard_count,
        learning.predicate_learning_fact_count,
        learning.predicate_flashcard_count,
        learning.triplet_learning_fact_count,
        learning.triplet_flashcard_count,
    )


def _run_existing_source_learning(
    settings: Settings,
    user: User,
    sources: list[Source],
) -> None:
    """Run Source Learning for a source owned by the selected user."""
    source = inquirer.select(
        message='Select the source for Source Learning:',
        choices=[
            Choice(
                value=source,
                name=f'{source.key} ({source.uuid})',
            )
            for source in sources
        ],
    ).execute()
    proceed = inquirer.confirm(
        message='Run Source Learning on the selected source?',
        default=True,
    ).execute()

    if not proceed:
        logger.info('Cancelled.')
        return

    learning = _run_async(
        run_source_learning_stage(settings, user.uuid, source.uuid)
    )
    _log_source_learning_completion(source, learning)


def _adopt_existing_source(
    settings: Settings,
    user: User,
    sources: list[Source],
) -> None:
    """Confirm explicit ownership of one legacy source."""
    source = inquirer.select(
        message='Select an unowned source to adopt:',
        choices=[
            Choice(value=source, name=f'{source.key} ({source.uuid})')
            for source in sources
        ],
    ).execute()
    proceed = inquirer.confirm(
        message=f'Adopt {source.key} for {user.name}?',
        default=False,
    ).execute()
    if proceed:
        _run_async(adopt_source(settings, user.uuid, source.uuid))
        logger.info('Adopted source %s (%s).', source.key, source.uuid)
    else:
        logger.info('Cancelled.')


def _run_global_semantic(settings: Settings) -> None:
    """Run Global Semantic over all persisted source hubs."""
    proceed = inquirer.confirm(
        message='Run Global Semantic over all persisted source hubs?',
        default=True,
    ).execute()
    if not proceed:
        logger.info('Cancelled.')
        return
    result = _run_async(run_global_semantic_stage(settings))
    logger.info(
        (
            'Done: Global Semantic persisted %d entity, %d event, %d predicate, '
            '%d statement, and %d procedure global hub(s).'
        ),
        result.global_entity_hub_count,
        result.global_event_hub_count,
        result.global_predicate_hub_count,
        result.global_statement_hub_count,
        result.global_procedure_hub_count,
    )


if __name__ == '__main__':
    run()
