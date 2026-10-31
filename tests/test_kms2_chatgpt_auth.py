import asyncio
import threading

import pytest

from kms2.inference import chatgpt


def test_login_discards_token_and_restricts_native_auth_file_permissions(
    monkeypatch,
    tmp_path,
):
    returned = object()
    calls = []
    token_directory = tmp_path / 'litellm-chatgpt'
    token_directory.mkdir(mode=0o755)
    auth_file = token_directory / 'auth.json'

    class Authenticator:
        def __init__(self):
            self.token_dir = str(token_directory)
            self.auth_file = str(auth_file)

        def get_access_token(self):
            calls.append('login')
            auth_file.write_text('provider-owned credentials')
            return returned

    monkeypatch.setattr(chatgpt, 'Authenticator', Authenticator)

    assert chatgpt.login_chatgpt() is None
    assert calls == ['login']
    assert token_directory.stat().st_mode & 0o777 == 0o700
    assert auth_file.stat().st_mode & 0o777 == 0o600


def test_async_preflight_uses_worker_and_propagates_missing_auth(
    monkeypatch,
):
    caller_thread = threading.get_ident()
    calls = []

    class Authenticator:
        def get_access_token(self):
            calls.append(threading.get_ident())
            raise RuntimeError('ChatGPT device-code login needs a human')

    monkeypatch.setattr(chatgpt, 'Authenticator', Authenticator)

    with pytest.raises(RuntimeError, match='needs a human'):
        asyncio.run(chatgpt.ensure_chatgpt_authenticated())

    assert len(calls) == 1
    assert calls[0] != caller_thread
