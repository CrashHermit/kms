import logging
from types import SimpleNamespace

from kms2 import tui
from kms2.application import SemanticStageResult
from kms2.config import Settings
from kms2.core.model import Source


class _Prompt:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


def _semantic_result() -> SemanticStageResult:
    return SemanticStageResult(
        raw_assertion_count=4,
        source_entity_description_count=3,
        source_event_description_count=2,
        source_predicate_description_count=1,
    )


def test_tui_ingests_new_source_without_semantic_prompt(monkeypatch, caplog):
    prompts = iter(
        [
            _Prompt('book.pdf'),
            _Prompt('0,2'),
        ]
    )
    select_calls = []
    confirm_calls = []

    def select(**kwargs):
        select_calls.append(kwargs)
        return next(prompts)

    def confirm(**kwargs):
        confirm_calls.append(kwargs)
        return next(prompts)

    fake_inquirer = SimpleNamespace(
        select=select,
        filepath=lambda **kwargs: next(prompts),
        text=lambda **kwargs: next(prompts),
        confirm=confirm,
    )
    calls = []

    async def fake_list_sources(settings):
        assert isinstance(settings, Settings)
        return []

    async def fake_ingest(settings, pdf_path, pages):
        calls.append((settings, pdf_path, pages))
        return SimpleNamespace(
            source=Source(uuid='source-1', key='book.pdf'),
            page_count=3,
        )

    monkeypatch.setattr(tui, 'inquirer', fake_inquirer)
    monkeypatch.setattr(tui, 'list_sources', fake_list_sources)
    monkeypatch.setattr(tui, 'ingest_source', fake_ingest)
    caplog.set_level(logging.INFO, logger='kms2.tui')

    tui._run_tui()

    assert select_calls == []
    assert len(calls) == 1
    settings, pdf_path, pages = calls[0]
    assert isinstance(settings, Settings)
    assert pdf_path == 'book.pdf'
    assert pages == [0, 2]
    assert confirm_calls == []
    assert (
        'Done: source source-1 (book.pdf), ingested 3 page(s).'
        in caplog.messages
    )


def test_tui_runs_complete_stage_from_existing_source(monkeypatch, caplog):
    source = Source(uuid='source-1', key='book.pdf')
    prompts = iter(
        [
            _Prompt(tui.EXISTING_SOURCE_OPTION),
            _Prompt(source),
            _Prompt(True),
        ]
    )
    select_calls = []
    confirm_calls = []

    def select(**kwargs):
        select_calls.append(kwargs)
        return next(prompts)

    def confirm(**kwargs):
        confirm_calls.append(kwargs)
        return next(prompts)

    fake_inquirer = SimpleNamespace(select=select, confirm=confirm)
    calls = []

    async def fake_list_sources(settings):
        return [source]

    async def fake_run(settings, source_uuid):
        calls.append((settings, source_uuid))
        return _semantic_result()

    monkeypatch.setattr(tui, 'inquirer', fake_inquirer)
    monkeypatch.setattr(tui, 'list_sources', fake_list_sources)
    monkeypatch.setattr(tui, 'run_semantic_stage', fake_run)
    caplog.set_level(logging.INFO, logger='kms2.tui')

    tui._run_tui()

    assert select_calls[0] == {
        'message': 'What would you like to do?',
        'choices': [
            tui.NEW_SOURCE_OPTION,
            tui.EXISTING_SOURCE_OPTION,
        ],
    }
    source_choice = select_calls[1]['choices'][0]
    assert source_choice.value is source
    assert source_choice.name == 'book.pdf (source-1)'
    assert confirm_calls == [
        {
            'message': 'Run the semantic stage on the selected source?',
            'default': True,
        }
    ]
    assert len(calls) == 1
    settings, source_uuid = calls[0]
    assert isinstance(settings, Settings)
    assert source_uuid == 'source-1'
    assert (
        'Done: source source-1 (book.pdf), semantic stage persisted 4 assertion(s), '
        '3 source entity description(s), 2 source event description(s), and '
        '1 source predicate description(s).' in caplog.messages
    )


def test_pdf_directory_is_created_under_working_directory(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)

    directory = tui._pdf_directory()

    assert directory == tmp_path / 'pdfs'
    assert directory.is_dir()
