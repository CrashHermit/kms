"""Explicit authentication for LiteLLM's ChatGPT subscription provider."""

import asyncio
import os

from litellm.llms.chatgpt.authenticator import Authenticator


def login_chatgpt() -> None:
    """Run LiteLLM's interactive ChatGPT device login on the main thread."""
    authenticator = Authenticator()
    os.chmod(authenticator.token_dir, 0o700)
    authenticator.get_access_token()
    os.chmod(authenticator.auth_file, 0o600)


async def ensure_chatgpt_authenticated() -> None:
    """Check cached ChatGPT credentials without allowing worker-thread login."""
    await asyncio.to_thread(Authenticator().get_access_token)
