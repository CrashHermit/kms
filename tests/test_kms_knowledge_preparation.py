"""Behavioral tests for target knowledge preparation."""

import asyncio
from contextlib import asynccontextmanager

from kms.core.tokenization.budget import TokenBudget
from kms.core.tokenization.material import MaterialTokenEstimator
from kms.knowledge.candidates import (
    ExistingPropositionReference,
    NewArgument,
    NewProposition,
    NewPropositionReference,
)
from kms.knowledge.commit import KnowledgeCommitNode, KnowledgeCommitter
from kms.knowledge.extraction.models import (
    UnitExtractionCandidates,
    UnitExtractionResult,
)
from kms.knowledge.extraction.node import UnitKnowledgeExtractionNode
from kms.knowledge.identity.audit import IdentityAuditor
from kms.knowledge.identity.models import IdentityDecision
from kms.knowledge.models import (
    Argument,
    KnowledgeContext,
    Proposition,
    StoredKnowledge,
)
from kms.knowledge.preparation.graph import KnowledgePreparationGraph
from kms.knowledge.preparation.node import SourceSearchNode
from kms.knowledge.reasoning.models import ReasoningCandidates
from kms.knowledge.reasoning.node import KnowledgeReasoningNode
from kms.knowledge.retrieval.node import KnowledgeSearchNode
from kms.knowledge.state import KnowledgeState
from kms.knowledge.sufficiency.models import (
    InsufficientKnowledge,
    SufficientKnowledge,
)
from kms.knowledge.sufficiency.node import KnowledgeSufficiencyNode
from kms.media.budget import ImagePayloadBudget
from kms.media.models import Material
from kms.source.models.source import BlockType, Source, SourceBlock
from kms.source.models.unit import SourceUnit, SourceUnitKind
from kms.source.retrieval.models import SourceUnitRead, SourceUnitSearchHit
from kms.source.retrieval.read import SourceReader


class _Counter:
    def count_texts(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


class _Embedder:
    async def embed(self, request):
        return [
            [1.0] if material.text == 'Existing premise' else [2.0]
            for material in request.materials
        ]


class _Repository:
    def __init__(self, existing: list[Proposition] | None = None) -> None:
        self.existing = existing or []
        self.propositions: list[Proposition] = []
        self.arguments: list[Argument] = []
        self.saves: list[dict] = []

    @asynccontextmanager
    async def serialized(self, _user_uuid):
        yield object()

    async def search(self, _user_uuid, embedding, *, limit, embedding_model):
        return self.existing if embedding == [1.0] else []

    async def save(
        self,
        _user_uuid,
        propositions,
        embeddings,
        arguments,
        *,
        source_uuid,
        target_unit_uuid,
        embedding_model,
        existing_proposition_uuids,
        existing_argument_uuids,
        transaction,
    ):
        self.saves.append(
            {
                'source_uuid': source_uuid,
                'target_unit_uuid': target_unit_uuid,
                'existing_proposition_uuids': existing_proposition_uuids,
            }
        )
        self.propositions.extend(propositions)
        self.arguments.extend(arguments)
        return arguments


class _Identity:
    async def acall(self, *, request):
        return IdentityDecision(
            match_uuid=next(
                (
                    match.uuid
                    for match in request.matches
                    if match.statement == request.candidate.statement
                ),
                None,
            )
        )


class _Search:
    def __init__(self, context: KnowledgeContext) -> None:
        self.context = context
        self.calls = []

    async def search(self, user_uuid: str, query: Material) -> KnowledgeContext:
        self.calls.append((user_uuid, query))
        return self.context


class _Source:
    def __init__(self, units: list[SourceUnit]) -> None:
        source = Source(uuid='source', key='source')
        self.hits = [
            SourceUnitSearchHit(source=source, unit=unit, similarity=1.0)
            for unit in units
        ]
        self.queries = []

    async def search_source_units(self, user_uuid: str, query: Material):
        self.queries.append((user_uuid, query))
        return self.hits


class _SourceRepository:
    def __init__(self, units: list[SourceUnit]) -> None:
        self.source = Source(uuid='source', key='source')
        self.units = units

    async def read_units(
        self, user_uuid, unit_uuids, *, before_units=0, after_units=0
    ):
        if user_uuid != 'owner':
            return []
        units = [unit for unit in self.units if unit.uuid in unit_uuids]
        return [SourceUnitRead(source=self.source, unit=unit) for unit in units]


class _Extraction:
    def __init__(self, candidates_by_unit=None) -> None:
        self.candidates_by_unit = candidates_by_unit or {}
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        if request.target.uuid in self.candidates_by_unit:
            return self.candidates_by_unit[request.target.uuid]
        candidate_id = f'claim-{request.target.uuid}'
        return UnitExtractionCandidates(
            new_propositions=[
                NewProposition(
                    id=candidate_id,
                    name=f'Claim {request.target.uuid}',
                    statement=f'Claim from {request.target.uuid}',
                )
            ],
            existing_proposition_uuids=[],
            new_arguments=[],
            existing_argument_uuids=[],
        )


class _Reasoning:
    def __init__(self, candidates: ReasoningCandidates | None = None) -> None:
        self.candidates = candidates or ReasoningCandidates(
            new_propositions=[], new_arguments=[]
        )
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        return self.candidates


class _Sufficiency:
    def __init__(self, decisions) -> None:
        self.decisions = iter(decisions)
        self.requests = []

    async def acall(self, *, request):
        self.requests.append(request)
        return next(self.decisions)


def _unit(uuid: str) -> SourceUnit:
    return SourceUnit(
        uuid=uuid,
        kind=SourceUnitKind.CONTENT,
        blocks=[
            SourceBlock(
                block_type=BlockType.PARAGRAPH,
                content=f'Source content for {uuid}',
            )
        ],
    )


def _budget() -> TokenBudget:
    return TokenBudget(MaterialTokenEstimator(_Counter()), 100_000)


def _components(
    *,
    units: list[SourceUnit] | None = None,
    context: KnowledgeContext | None = None,
    decisions=None,
    extraction=None,
    reasoning=None,
):
    units = units or []
    context = context or KnowledgeContext(
        matched=[], propositions=[], arguments=[]
    )
    repository = _Repository(
        [
            proposition
            for proposition in context.propositions
            if proposition.statement == 'Existing premise'
        ]
    )
    embedder = _Embedder()
    budget = _budget()
    auditor = IdentityAuditor(
        repository,
        embedder,
        'test-model',
        _Identity(),
        budget,
        candidate_limit=16,
    )
    committer = KnowledgeCommitter(
        repository, embedder, 'test-model', auditor=auditor
    )
    source = _Source(units)
    search = _Search(context)
    extraction_module = extraction or _Extraction()
    reasoning_module = reasoning or _Reasoning()
    sufficiency_module = _Sufficiency(decisions or [])
    reader = SourceReader(_SourceRepository(units))
    extraction_node = UnitKnowledgeExtractionNode(
        reader,
        extraction_module,
        budget,
        ImagePayloadBudget(4, 100_000, 100_000),
        before_tokens=1024,
        after_tokens=1024,
    )
    reasoning_node = KnowledgeReasoningNode(
        reasoning_module, budget, candidate_limit=64, argument_limit=64
    )
    graph = KnowledgePreparationGraph(
        KnowledgeSearchNode(search),
        SourceSearchNode(source),
        extraction_node,
        reasoning_node,
        KnowledgeSufficiencyNode(sufficiency_module),
        KnowledgeCommitNode(committer),
        source_limit=8,
    )
    return (
        graph,
        search,
        source,
        extraction_module,
        reasoning_module,
        repository,
    )


def _insufficient():
    return InsufficientKnowledge(
        explanation='Missing evidence', missing_knowledge='Need a claim'
    )


def test_existing_sufficient_knowledge_stops_without_creation():
    proposition = Proposition(
        uuid='existing', name='Claim', statement='Existing fact'
    )
    context = KnowledgeContext(
        matched=[proposition], propositions=[proposition], arguments=[]
    )
    graph, _, source, extraction, reasoning, repository = _components(
        units=[_unit('unit')],
        context=context,
        decisions=[SufficientKnowledge(explanation='Covered')],
    )
    result = asyncio.run(graph.prepare('owner', 'target'))
    assert result.judgment.kind == 'sufficient'
    assert result.context.propositions == [proposition]
    assert not source.queries
    assert not extraction.requests
    assert not reasoning.requests
    assert repository.saves == []


def test_reasoning_can_satisfy_before_source_search():
    premise = Proposition(
        uuid='premise', name='Premise', statement='Existing premise'
    )
    context = KnowledgeContext(
        matched=[premise], propositions=[premise], arguments=[]
    )
    reasoning = _Reasoning(
        ReasoningCandidates(
            new_propositions=[
                NewProposition(
                    id='derived', name='Conclusion', statement='Derived fact'
                )
            ],
            new_arguments=[
                NewArgument(
                    premises=[ExistingPropositionReference(uuid='premise')],
                    conclusion=NewPropositionReference(id='derived'),
                )
            ],
        )
    )
    graph, _, source, _, reasoning, _ = _components(
        units=[_unit('unit')],
        context=context,
        decisions=[
            _insufficient(),
            SufficientKnowledge(explanation='Covered'),
        ],
        reasoning=reasoning,
    )
    result = asyncio.run(graph.prepare('owner', 'target'))
    assert result.judgment.kind == 'sufficient'
    assert len(reasoning.requests) == 1
    assert not source.queries
    assert 'Derived fact' in {
        item.statement for item in result.context.propositions
    }


def test_source_units_are_committed_and_prior_results_reused_until_sufficient():
    units = [_unit('first'), _unit('second'), _unit('unused')]
    graph, _, source, extraction, reasoning, repository = _components(
        units=units,
        decisions=[
            _insufficient(),
            _insufficient(),
            _insufficient(),
            SufficientKnowledge(explanation='Covered'),
        ],
    )
    result = asyncio.run(graph.prepare('owner', 'target'))
    assert [query.text for _, query in source.queries] == ['target']
    assert [request.target.uuid for request in extraction.requests] == [
        'first',
        'second',
    ]
    assert [
        [item.statement for item in request.existing_propositions]
        for request in extraction.requests
    ] == [[], ['Claim from first']]
    assert [item.statement for item in result.context.propositions] == [
        'Claim from first',
        'Claim from second',
    ]
    assert [item['target_unit_uuid'] for item in repository.saves] == [
        'first',
        'second',
    ]
    assert [
        [item.statement for item in request.context.matched]
        for request in reasoning.requests
    ] == [
        ['Claim from first'],
        ['Claim from first', 'Claim from second'],
    ]
    assert result.judgment.kind == 'sufficient'


def test_no_source_returns_explicit_gap():
    graph, _, _, _, _, _ = _components(
        decisions=[_insufficient(), _insufficient()]
    )
    result = asyncio.run(graph.prepare('owner', 'target'))
    assert result.judgment.kind == 'insufficient'
    assert result.judgment.missing_knowledge == 'Need a claim'
    assert result.context.propositions == []


def test_exhausted_source_hits_preserve_last_gap_and_complete_context():
    units = [_unit(f'unit-{index}') for index in range(8)]
    graph, _, _, extraction, _, repository = _components(
        units=units, decisions=[_insufficient() for _ in range(10)]
    )
    result = asyncio.run(graph.prepare('owner', 'target'))
    assert result.judgment.kind == 'insufficient'
    assert result.judgment.missing_knowledge == 'Need a claim'
    assert [request.target.uuid for request in extraction.requests] == [
        unit.uuid for unit in units
    ]
    assert [item.statement for item in result.context.propositions] == [
        f'Claim from {unit.uuid}' for unit in units
    ]
    assert [item['target_unit_uuid'] for item in repository.saves] == [
        unit.uuid for unit in units
    ]


def test_candidates_enter_context_only_after_canonical_commit():
    premise = Proposition(
        uuid='canonical-premise',
        name='Premise',
        statement='Existing premise',
    )
    context = KnowledgeContext(
        matched=[premise], propositions=[premise], arguments=[]
    )
    candidates = UnitExtractionCandidates(
        new_propositions=[
            NewProposition(
                id='local-premise', name='Premise', statement='Existing premise'
            ),
            NewProposition(
                id='local-conclusion',
                name='Conclusion',
                statement='A conclusion',
            ),
        ],
        existing_proposition_uuids=[],
        new_arguments=[
            NewArgument(
                premises=[NewPropositionReference(id='local-premise')],
                conclusion=NewPropositionReference(id='local-conclusion'),
            )
        ],
        existing_argument_uuids=[],
    )
    _, _, _, extraction_module, reasoning_module, repository = _components(
        units=[_unit('target')],
        context=context,
        extraction=_Extraction({'target': candidates}),
    )
    source_reader = SourceReader(_SourceRepository([_unit('target')]))
    budget = _budget()
    extraction_stage = UnitKnowledgeExtractionNode(
        source_reader,
        extraction_module,
        budget,
        ImagePayloadBudget(4, 100_000, 100_000),
        before_tokens=1024,
        after_tokens=1024,
    )
    state: KnowledgeState = {
        'user_uuid': 'owner',
        'unit_uuid': 'target',
        'understanding': 'target',
        'context': context,
    }
    original_context = state['context']
    pending = asyncio.run(extraction_stage.run(state))['pending']
    assert pending is not None
    assert state['context'] is original_context
    assert [item.uuid for item in state['context'].propositions] == [
        'canonical-premise'
    ]

    committer = KnowledgeCommitter(
        repository,
        _Embedder(),
        'test-model',
        auditor=IdentityAuditor(
            repository,
            _Embedder(),
            'test-model',
            _Identity(),
            budget,
            candidate_limit=16,
        ),
    )
    committed = asyncio.run(
        KnowledgeCommitNode(committer).run({**state, 'pending': pending})
    )
    canonical_conclusion = next(
        item
        for item in committed['context'].propositions
        if item.statement == 'A conclusion'
    )
    [argument] = committed['context'].arguments
    assert argument.premise_uuids == ['canonical-premise']
    assert argument.conclusion_uuid == canonical_conclusion.uuid
    assert repository.saves[-1]['existing_proposition_uuids'] == [
        'canonical-premise'
    ]

    reasoning_node = KnowledgeReasoningNode(
        reasoning_module, budget, candidate_limit=64, argument_limit=64
    )
    asyncio.run(reasoning_node.run({**state, 'context': committed['context']}))
    assert reasoning_module.requests[0].context.arguments == [argument]
    assert premise in reasoning_module.requests[0].context.propositions
    assert (
        canonical_conclusion
        in reasoning_module.requests[0].context.propositions
    )

    empty = UnitExtractionResult(
        source_uuid='source',
        target_unit_uuid='target',
        candidates=UnitExtractionCandidates(
            new_propositions=[],
            existing_proposition_uuids=[],
            new_arguments=[],
            existing_argument_uuids=[],
        ),
    )
    unchanged = asyncio.run(
        KnowledgeCommitNode(committer).run(
            {**state, 'context': committed['context'], 'pending': empty}
        )
    )
    assert unchanged['context'] == committed['context']
    assert unchanged['stored'] == StoredKnowledge(propositions=[], arguments=[])
