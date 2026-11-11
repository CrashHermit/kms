"""Interactive terminal entry point for KMS source processing."""

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Awaitable
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from kms.application import (
    adopt_source,
    create_user,
    discover_sources,
    ingest_source,
    list_unowned_sources,
    list_users,
    login_chatgpt,
)
from kms.config.settings import Settings
from kms.source.models.source import Source
from kms.users.models import User

PDF_DIRECTORY = Path('pdfs')
NEW_SOURCE_OPTION = 'Ingest a PDF'
DISCOVERY_OPTION = 'Discover sources'
ADOPT_SOURCE_OPTION = 'Adopt existing source'
SELECT_LLM_PROFILE_OPTION = 'Select text inference profile'
CREATE_USER_OPTION = 'Create user'

logger = logging.getLogger(__name__)


def _pdf_directory() -> Path:
    """Return the project-local directory used for source PDFs."""
    directory = Path.cwd() / PDF_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run() -> None:
    """Run the KMS TUI or an explicit authentication/profile command."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(prog='kms')
    parser.add_argument(
        '--login-chatgpt',
        action='store_true',
        help='Authenticate LiteLLM with your ChatGPT subscription, then exit.',
    )
    parser.add_argument(
        '--llm-profile',
        help='Select a configured default text inference profile.',
    )
    arguments = parser.parse_args()
    try:
        if arguments.login_chatgpt:
            login_chatgpt()
            return
        settings = Settings()
        if arguments.llm_profile is not None:
            if arguments.llm_profile not in settings.llm.profiles:
                parser.error(f'Unknown LLM profile: {arguments.llm_profile}')
            settings.llm.default_text_profile = arguments.llm_profile
        _run_tui(settings)
    except KeyboardInterrupt, asyncio.CancelledError:
        logger.info('Cancelled.')
        sys.exit(0)


def _run_tui(settings: Settings | None = None) -> None:
    """Resolve a user, then prompt for one user-scoped operation."""
    settings = Settings() if settings is None else settings
    user = _select_user(settings)
    unowned_sources = _run_async(list_unowned_sources(settings))
    choices = [
        NEW_SOURCE_OPTION,
        DISCOVERY_OPTION,
        SELECT_LLM_PROFILE_OPTION,
    ]
    if unowned_sources:
        choices.append(ADOPT_SOURCE_OPTION)
    action = inquirer.select(
        message='What would you like to do?',
        choices=choices,
    ).execute()
    while action == SELECT_LLM_PROFILE_OPTION:
        _select_llm_profile(settings)
        action = inquirer.select(
            message='What would you like to do?',
            choices=choices,
        ).execute()
    if action == ADOPT_SOURCE_OPTION:
        _adopt_existing_source(settings, user, unowned_sources)
    elif action == DISCOVERY_OPTION:
        _run_discovery(settings)
    else:
        _run_new_source(settings, user)


def _select_llm_profile(settings: Settings) -> None:
    """Choose a default text profile for the current TUI invocation."""
    profile = inquirer.select(
        message='Select the default text inference profile:',
        choices=list(settings.llm.profiles),
        default=settings.llm.default_text_profile,
    ).execute()
    settings.llm.default_text_profile = profile


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
    logger.info('Done: source %s (%s).', result.source.uuid, result.source.key)


def _run_discovery(settings: Settings) -> None:
    """Search for source links without capturing or admitting them."""
    question = inquirer.text(message='Source discovery question:').execute()
    candidates = _run_async(discover_sources(settings, question))
    for candidate in candidates:
        logger.info(
            '%s\n%s\n%s', candidate.title, candidate.url, candidate.snippet
        )
    if not candidates:
        logger.info('No sources selected.')


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


if __name__ == '__main__':
    run()
