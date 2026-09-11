import logging
from types import SimpleNamespace

from kms2 import tui
from kms2.application import SourceIngestionResult
from kms2.config import Settings
from kms2.core.model.source import Source


class _Prompt:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


def test_tui_runs_selected_pages_and_logs_completion(monkeypatch, caplog):
    prompts = iter(
        [
            _Prompt('book.pdf'),
            _Prompt('0,2'),
            _Prompt(True),
        ]
    )
    fake_inquirer = SimpleNamespace(
        filepath=lambda **kwargs: next(prompts),
        text=lambda **kwargs: next(prompts),
        confirm=lambda **kwargs: next(prompts),
    )
    calls = []

    async def fake_ingest(settings, pdf_path, pages):
        calls.append((settings, pdf_path, pages))
        return SourceIngestionResult(
            source=Source(uuid='source-1', key='book.pdf'),
            page_count=3,
        )

    monkeypatch.setattr(tui, 'inquirer', fake_inquirer)
    monkeypatch.setattr(tui, 'ingest_source', fake_ingest)
    caplog.set_level(logging.INFO, logger='kms2.tui')

    tui._run_tui()

    assert len(calls) == 1
    settings, pdf_path, pages = calls[0]
    assert isinstance(settings, Settings)
    assert pdf_path == 'book.pdf'
    assert pages == [0, 2]
    assert (
        'Done: source source-1 (book.pdf), 3 page(s) persisted.'
        in caplog.messages
    )


def test_tui_declined_confirmation_does_not_ingest(monkeypatch):
    prompts = iter(
        [
            _Prompt('book.pdf'),
            _Prompt(''),
            _Prompt(False),
        ]
    )
    fake_inquirer = SimpleNamespace(
        filepath=lambda **kwargs: next(prompts),
        text=lambda **kwargs: next(prompts),
        confirm=lambda **kwargs: next(prompts),
    )
    calls = []

    async def fake_ingest(settings, pdf_path, pages):
        calls.append((settings, pdf_path, pages))

    monkeypatch.setattr(tui, 'inquirer', fake_inquirer)
    monkeypatch.setattr(tui, 'ingest_source', fake_ingest)

    tui._run_tui()

    assert calls == []
