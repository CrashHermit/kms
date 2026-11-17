import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest

from kms import application
from kms.config.llm import LocalLLMProfile
from kms.config.settings import Settings
from kms.core.embedding import EmbeddingModel
from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewArgument,
    NewProposition,
    NewPropositionReference,
)
from kms.knowledge.commit import KnowledgeCommitter
from kms.knowledge.extraction.models import UnitExtractionCandidates
from kms.knowledge.extraction.node import UnitKnowledgeExtractionNode
from kms.knowledge.identity.models import AuditResult
from kms.knowledge.models import KnowledgeContext, Proposition
from kms.knowledge.sufficiency.models import SufficientKnowledge
from kms.knowledge.sufficiency.node import KnowledgeSufficiencyNode
from kms.media.budget import ImagePayloadBudget
from kms.media.models import Material
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.models import SourceUnitRead
from kms.source.retrieval.read import SourceReader


class _Runtime:
    def __init__(self, settings):
        self.settings = settings
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True


_EMBEDDINGS = SimpleNamespace(
    embedder=object(),
    model=EmbeddingModel(
        model='voyage/voyage-multimodal-3.5',
        dimension=1024,
    ),
    budget=object(),
)


@asynccontextmanager
async def _embedding_resources(settings):
    yield _EMBEDDINGS


def test_application_chatgpt_login_delegates_to_provider(monkeypatch):
    calls = []
    monkeypatch.setattr(
        application,
        '_login_chatgpt',
        lambda: calls.append('login'),
    )

    assert application.login_chatgpt() is None
    assert calls == ['login']


def test_due_reviews_reject_naive_time_before_database_access():
    with pytest.raises(ValueError, match='timezone-aware'):
        asyncio.run(
            application.list_due_reviews(
                Settings(_env_file=None), 'owner', datetime(2025, 1, 1)
            )
        )


def test_chatgpt_preflight_resolves_only_relevant_stage_settings(
    monkeypatch,
):
    settings = Settings(
        _env_file=None, llm={'default_text_profile': 'qwen3.8-9b-distill-text'}
    )
    for stage in (
        settings.source_processing.content_correction,
        settings.source_processing.image_seam,
        settings.source_processing.image_filter,
    ):
        stage.inference.profile = 'gemma-vision-8k'
    settings.source_processing.formatting.inference.profile = (
        'qwen3.8-9b-distill-text'
    )
    settings.source_processing.procedure_finder.start_router.profile = (
        'chatgpt-luna'
    )
    calls = []

    async def ensure_authenticated():
        calls.append('preflight')

    monkeypatch.setattr(
        application,
        'ensure_chatgpt_authenticated',
        ensure_authenticated,
    )

    asyncio.run(
        application._preflight_chatgpt(
            settings, settings.source_processing.procedure_finder
        )
    )
    assert calls == ['preflight']

    calls.clear()
    asyncio.run(
        application._preflight_chatgpt(
            settings, settings.source_processing.formatting
        )
    )
    assert calls == []


def test_chatgpt_preflight_skips_local_stage(monkeypatch):
    settings = Settings(
        _env_file=None, llm={'default_text_profile': 'qwen3.8-9b-distill-text'}
    )
    # The named vision profile and application text default both run locally.
    settings.llm.profiles['chatgpt-luna'] = LocalLLMProfile(
        model_server_profile='gemma-vision-8k'
    )
    calls = []

    async def ensure_authenticated():
        calls.append('preflight')

    monkeypatch.setattr(
        application,
        'ensure_chatgpt_authenticated',
        ensure_authenticated,
    )

    asyncio.run(
        application._preflight_chatgpt(settings, settings.source_processing)
    )
    assert calls == []


class _Database:
    def __init__(self, settings):
        self.settings = settings
        self.closed = False
        self.schema_calls = []

    async def close(self):
        self.closed = True

    def session(self):
        return _SchemaSession(self.schema_calls)


class _SchemaSession:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def run(self, statement, **parameters):
        self.calls.append((statement, parameters))
        return self

    async def data(self):
        return []

    async def consume(self):
        pass


@pytest.fixture(autouse=True)
def _use_fake_resources(monkeypatch):
    monkeypatch.setattr(
        application, 'embedding_resources', _embedding_resources
    )


def test_create_user_ensures_structural_schema_and_closes_database(
    monkeypatch,
):
    settings = Settings()
    database = _Database(settings.database)
    schema_calls = []
    service_calls = []
    user = SimpleNamespace(name='Alex')

    async def ensure_structural_schema(session_factory):
        schema_calls.append(session_factory)

    class _UserRepositorySpy:
        def __init__(self, session_factory):
            service_calls.append(('init', session_factory))

        async def create_user(self, user):
            service_calls.append(('create', user.name))

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application.schema,
        'ensure_structural_schema',
        ensure_structural_schema,
    )
    monkeypatch.setattr(application, 'UserRepository', _UserRepositorySpy)

    created = asyncio.run(application.create_user(settings, 'Alex'))
    assert created.name == user.name
    assert schema_calls == [database.session]
    assert service_calls == [('init', database.session), ('create', 'Alex')]
    assert database.closed is True


def test_unknown_user_fails_before_provider_acquisition(monkeypatch):
    @asynccontextmanager
    async def database_resources(settings):
        yield SimpleNamespace(session=object())

    class MissingUser:
        def __init__(self, session):
            pass

        async def read_user(self, user_uuid):
            raise LookupError(user_uuid)

    def unexpected_resources(*args, **kwargs):
        raise AssertionError('provider resources acquired for missing user')

    monkeypatch.setattr(application, '_database_resources', database_resources)
    monkeypatch.setattr(application, 'UserRepository', MissingUser)
    monkeypatch.setattr(
        application, '_application_resources', unexpected_resources
    )
    with pytest.raises(LookupError, match='missing-user'):
        settings = Settings(_env_file=None)
        capture = application.CapturedPDFSource(
            data=b'pdf', retrieved_at=application.datetime.now(application.UTC)
        )
        asyncio.run(
            application.ingest_captured_source(
                settings, 'missing-user', capture
            )
        )


def test_list_sources_closes_database_and_returns_user_sources(monkeypatch):
    settings = Settings()
    database = _Database(settings.database)
    sources = [
        Source(uuid='source-1', key='book.pdf'),
        Source(uuid='source-2', key='notes.pdf'),
    ]
    repository_calls = []

    class _SourceRepository:
        def __init__(self, session_factory):
            repository_calls.append(session_factory)

        async def list_sources(self, user_uuid):
            assert user_uuid == 'user-1'
            return sources

    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)
    monkeypatch.setattr(
        application, 'SourceCatalogRepository', _SourceRepository
    )

    result = asyncio.run(application.list_sources(settings, 'user-1'))

    assert result == sources
    assert len(repository_calls) == 1
    assert database.closed is True


def test_schema_failure_closes_resources_and_database(monkeypatch):
    settings = Settings()
    runtime = _Runtime(settings.local_models)
    database = _Database(settings.database)

    async def fail_schema(session_factory, *, embedding_dimension):
        assert embedding_dimension == _EMBEDDINGS.model.dimension
        raise RuntimeError('schema failed')

    monkeypatch.setattr(application.schema, 'ensure_schema', fail_schema)
    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda settings: runtime
    )
    monkeypatch.setattr(application, 'DatabaseClient', lambda _: database)

    async def acquire():
        async with application._application_resources(settings):
            pass

    with pytest.raises(RuntimeError, match='schema failed'):
        asyncio.run(acquire())

    assert database.closed is True
    assert runtime.entered is True
    assert runtime.closed is True


def test_search_sources_with_empty_allowlist_skips_application_resources(
    monkeypatch,
) -> None:
    settings = Settings(_env_file=None)

    def fail_resources(*_args, **_kwargs):
        pytest.fail('empty source scope must not acquire resources')

    monkeypatch.setattr(application, '_application_resources', fail_resources)

    assert (
        asyncio.run(
            application.search_sources(
                settings,
                'user-1',
                Material.from_text('query'),
                source_uuids=[],
            )
        )
        == []
    )


def test_search_source_units_with_empty_allowlist_skips_application_resources(
    monkeypatch,
) -> None:
    settings = Settings(_env_file=None)

    def fail_resources(*_args, **_kwargs):
        pytest.fail('empty source scope must not acquire resources')

    monkeypatch.setattr(application, '_application_resources', fail_resources)

    assert (
        asyncio.run(
            application.search_source_units(
                settings,
                'user-1',
                Material.from_text('query'),
                source_uuids=[],
            )
        )
        == []
    )


def test_extract_then_store_commits_only_at_the_public_storage_boundary(
    monkeypatch,
):
    settings = Settings(_env_file=None)
    events = []
    acquisitions = []
    preflights = []
    premise = Proposition(
        uuid='premise', name='Premise', statement='The premise is true.'
    )
    candidates = UnitExtractionCandidates(
        new_propositions=[
            NewProposition(
                id='conclusion-local',
                name='Conclusion',
                statement='The conclusion follows.',
            )
        ],
        existing_proposition_uuids=[premise.uuid],
        new_arguments=[
            NewArgument(
                premises=[ExistingPropositionReference(uuid=premise.uuid)],
                conclusion=NewPropositionReference(id='conclusion-local'),
            )
        ],
        existing_argument_uuids=[],
    )

    class _SourceRepository:
        async def read_units(
            self, user_uuid, unit_uuids, *, before_units=0, after_units=0
        ):
            assert user_uuid == 'owner'
            assert unit_uuids == ['unit']
            assert before_units in (0, 1)
            assert after_units in (0, 1)
            unit = SourceUnit(
                uuid='unit',
                kind=SourceUnitKind.CONTENT,
                blocks=[
                    SourceBlock(
                        block_type=BlockType.PARAGRAPH,
                        content='The source passage.',
                    )
                ],
            )
            return [
                SourceUnitRead(
                    source=Source(uuid='source', key='source.pdf'),
                    unit=unit,
                )
            ]

    class _Extraction:
        async def acall(self, *, request):
            events.append('inference')
            assert request.understanding == 'Understand the conclusion'
            assert request.target.uuid == 'unit'
            assert request.existing_propositions == [premise]
            assert request.existing_arguments == []
            return candidates

    module = _Extraction()

    class _LengthCounter:
        def count_texts(self, texts):
            return [len(text) for text in texts]

    def build_extraction_node(settings, database, *, predictors, tokenizers):
        return UnitKnowledgeExtractionNode(
            SourceReader(_SourceRepository()),
            module,
            TokenBudget(MaterialTokenEstimator(_LengthCounter()), 100000),
            ImagePayloadBudget(4, 100000, 100000),
            before_tokens=1024,
            after_tokens=1024,
        )

    class _Repository:
        @asynccontextmanager
        async def serialized(self, user_uuid):
            assert user_uuid == 'owner'
            yield object()

        async def save(
            self,
            user_uuid,
            propositions,
            embeddings,
            arguments,
            *,
            source_uuid,
            target_unit_uuid,
            existing_proposition_uuids,
            existing_argument_uuids,
            embedding_model,
            transaction,
        ):
            events.append('save')
            assert user_uuid == 'owner'
            assert source_uuid == 'source'
            assert target_unit_uuid == 'unit'
            assert existing_proposition_uuids == ['premise']
            assert existing_argument_uuids == []
            assert embedding_model == 'test-model'
            assert transaction is not None
            assert len(propositions) == len(embeddings) == 1
            self.arguments = arguments
            return arguments

    class _Embedder:
        async def embed(self, request):
            events.append('embed')
            assert request.input_type == 'document'
            return [[1.0, 0.0] for _ in request.materials]

    class _Auditor:
        async def audit(self, user_uuid, new_propositions):
            events.append('audit')
            assert user_uuid == 'owner'
            [candidate] = new_propositions
            canonical = Proposition(
                uuid='canonical-conclusion',
                name=candidate.name,
                statement=candidate.statement,
            )
            return AuditResult(
                canonical={candidate.id: canonical},
                new_propositions=[canonical],
            )

    repository = _Repository()
    committer = KnowledgeCommitter(
        repository,
        _Embedder(),
        'test-model',
        auditor=_Auditor(),
    )

    @asynccontextmanager
    async def application_resources(settings, *, embeddings_required=True):
        acquisitions.append(embeddings_required)
        yield SimpleNamespace(
            local_models=object(),
            embeddings=_EMBEDDINGS if embeddings_required else None,
            tokenizers=object(),
            database=object(),
            recorder=None,
        )

    async def preflight(settings, *stages):
        preflights.append(stages)

    monkeypatch.setattr(application, '_preflight_chatgpt', preflight)

    monkeypatch.setattr(
        application, '_application_resources', application_resources
    )
    monkeypatch.setattr(
        application,
        'PredictorFactory',
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        application.extraction, 'build_node', build_extraction_node
    )
    monkeypatch.setattr(
        application, 'build_committer', lambda *args, **kwargs: committer
    )

    extracted = asyncio.run(
        application.extract_unit_knowledge(
            settings,
            'owner',
            'unit',
            'Understand the conclusion',
            existing_propositions=[premise],
            existing_arguments=[],
        )
    )

    assert extracted is not None
    assert extracted.source_uuid == 'source'
    assert extracted.target_unit_uuid == 'unit'
    assert extracted.candidates == candidates
    assert [item.id for item in extracted.candidates.new_propositions] == [
        'conclusion-local'
    ]
    assert events == ['inference']
    assert acquisitions == [False]
    assert preflights == [(settings.knowledge_extraction.inference,)]

    stored = asyncio.run(
        application.store_unit_knowledge(settings, 'owner', extracted)
    )

    [conclusion] = stored.propositions
    [argument] = stored.arguments
    assert conclusion.uuid == 'canonical-conclusion'
    assert argument.premise_uuids == [premise.uuid]
    assert argument.conclusion_uuid == conclusion.uuid
    assert events == ['inference', 'audit', 'embed', 'save']
    assert acquisitions == [False, True]
    assert preflights == [
        (settings.knowledge_extraction.inference,),
        (settings.knowledge_identity.inference,),
    ]


def test_sufficiency_uses_inference_resources_without_database_or_embeddings(
    monkeypatch, tmp_path
):
    settings = Settings(
        _env_file=None,
        training={'examples_directory': tmp_path},
    )
    runtime = _Runtime(settings.local_models)
    tokenizers = object()
    predictors = object()
    recorder = object()
    recorder_paths = []
    context = KnowledgeContext(
        matched=[],
        propositions=[
            Proposition(
                uuid='proposition',
                name='Claim',
                statement='A supplied claim.',
            )
        ],
        arguments=[],
    )
    judgment = SufficientKnowledge(
        explanation='The claim addresses the request.'
    )
    calls = []
    preflight_calls = []

    class _Module:
        async def acall(self, *, request):
            calls.append(request)
            return judgment

    def build_node(actual_settings, *, predictors, tokenizers):
        assert actual_settings is settings
        assert predictors is expected_predictors
        assert tokenizers is expected_tokenizers
        return KnowledgeSufficiencyNode(_Module())

    expected_predictors = predictors
    expected_tokenizers = tokenizers

    def make_predictor_factory(
        actual_settings, actual_runtime, actual_recorder, *, tokenizers
    ):
        assert actual_settings is settings
        assert actual_runtime is runtime
        assert actual_recorder is recorder
        assert tokenizers is expected_tokenizers
        return predictors

    async def preflight(settings, *stages):
        preflight_calls.append(stages)

    def unavailable(*_args, **_kwargs):
        pytest.fail(
            'inference-only sufficiency acquired database or embeddings'
        )

    monkeypatch.setattr(application, '_preflight_chatgpt', preflight)
    monkeypatch.setattr(
        application, 'LocalModelRuntime', lambda settings: runtime
    )
    monkeypatch.setattr(
        application, 'LanguageModelTokenizers', lambda settings: tokenizers
    )
    monkeypatch.setattr(application, 'PredictorFactory', make_predictor_factory)
    monkeypatch.setattr(application.sufficiency, 'build_node', build_node)
    monkeypatch.setattr(
        application,
        'Recorder',
        lambda directory: recorder_paths.append(directory) or recorder,
    )
    monkeypatch.setattr(application, '_application_resources', unavailable)
    monkeypatch.setattr(application, 'DatabaseClient', unavailable)
    monkeypatch.setattr(application, 'embedding_resources', unavailable)

    result = asyncio.run(
        application.judge_knowledge_sufficiency(
            settings, 'Understand the claim', context
        )
    )

    assert result == judgment
    assert len(calls) == 1
    assert calls[0].understanding == 'Understand the claim'
    assert calls[0].context == context
    assert preflight_calls == [(settings.knowledge_sufficiency.inference,)]
    assert runtime.entered is True
    assert runtime.closed is True
    assert recorder_paths == [tmp_path]
