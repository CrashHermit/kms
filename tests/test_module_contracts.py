from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kms.construction import (
    block_corrector,
    entity_enrichment,
    formatter,
    governance_judge,
    predicate_enrichment,
    procedure_enrichment,
    statement_procedure_builder,
    triplet_extractor,
)
from kms.core import context_window, models


def _node(position: int, text: str = 'text') -> context_window.ContextNode:
    return context_window.ContextNode(
        position=position,
        type='paragraph',
        content=text,
    )


def test_governance_judge_uses_text_only_inputs_and_validates_boolean():
    judge = governance_judge.GovernanceJudge.__new__(
        governance_judge.GovernanceJudge
    )
    instruction_nodes = [
        _node(0, 'Use this rule.')
    ]
    statement_nodes = [
        context_window.ContextNode(
            position=0,
            type='image',
            content='A diagram of the rule.',
            assets=[models.VisualAsset(path='figure.png')],
        )
    ]
    context_before = [
        context_window.ContextNode(
            position=0,
            type='paragraph',
            content='Before',
        )
    ]
    context_after = [
        context_window.ContextNode(
            position=0,
            type='paragraph',
            content='After',
        )
    ]

    encoded = judge.encode(
        instruction_nodes, context_before, statement_nodes, context_after
    )
    assert all(
        isinstance(item, models.NodeInput)
        for field in encoded.values()
        for item in field
    )
    assert encoded['statement_nodes'][0].text == 'A diagram of the rule.'
    assert encoded['context_before'][0].text == 'Before'
    assert encoded['context_after'][0].text == 'After'
    assert not hasattr(encoded['statement_nodes'][0], 'assets')
    assert 'confidence' not in encoded
    assert judge.decode(
        SimpleNamespace(governs=True), **encoded
    ) is True
    with pytest.raises(ValueError, match='governs must be a boolean'):
        judge.decode(SimpleNamespace(governs='true'), **encoded)

def test_role_and_partition_modules_validate_positions_against_original_nodes():
    role_typer = statement_procedure_builder.RoleTyper.__new__(
        statement_procedure_builder.RoleTyper
    )
    nodes = [
        _node(0, 'A proof.'),
        context_window.ContextNode(
            position=1,
            type='image',
            content='Diagram of the proof structure.',
            assets=[models.VisualAsset(path='figure.png')],
        ),
    ]
    role_encoded = role_typer.encode(current_nodes=nodes)
    assert all(
        isinstance(item, models.NodeInput)
        for item in role_encoded['current_nodes']
    )
    assert role_encoded['current_nodes'][1].text == (
        'Diagram of the proof structure.'
    )
    assert not hasattr(role_encoded['current_nodes'][1], 'assets')
    assert role_typer.decode(
        SimpleNamespace(has_statement=True, has_procedure=False),
        **role_encoded,
    ) == (True, False)

    partitioner = statement_procedure_builder.StatementPartitioner.__new__(
        statement_procedure_builder.StatementPartitioner
    )
    encoded = partitioner.encode(current_nodes=nodes)
    assert all(
        isinstance(item, models.NodeInput)
        for item in encoded['current_nodes']
    )
    empty_image = partitioner.encode(
        current_nodes=[
            context_window.ContextNode(position=2, type='image', content=None)
        ]
    )['current_nodes']
    assert empty_image == [
        models.NodeInput(index=3, node_type='image', text='')
    ]
    assert partitioner.decode(
        SimpleNamespace(statement_positions=[1]), current_nodes=nodes
    ) == [0]
    with pytest.raises(ValueError, match='one-based'):
        partitioner.decode(
            SimpleNamespace(statement_positions=[3]), current_nodes=nodes
        )

def test_formatter_and_corrector_routers_reject_non_boolean_predictions():
    formatter_router = formatter.FormatterRouter.__new__(
        formatter.FormatterRouter
    )
    assert formatter_router.encode('1. text') == {'lines': '1. text'}
    assert formatter_router.decode(
        SimpleNamespace(needs_formatting=False), lines='1. text'
    ) is False
    with pytest.raises(ValueError, match='needs_formatting must be a boolean'):
        formatter_router.decode(
            SimpleNamespace(needs_formatting=0), lines='1. text'
        )

    corrector_router = block_corrector.BlockCorrectionRouter.__new__(
        block_corrector.BlockCorrectionRouter
    )
    assert corrector_router.decode(
        SimpleNamespace(needs_correction=True), lines='1. text'
    ) is True
    with pytest.raises(ValueError, match='needs_correction must be a boolean'):
        corrector_router.decode(
            SimpleNamespace(needs_correction='false'), lines='1. text'
        )


def test_source_procedure_writer_contract():
    writer = procedure_enrichment.SourceProcedureWriter.__new__(
        procedure_enrichment.SourceProcedureWriter
    )
    statement = [
        models.TextNodeInput(
            local_index=0, node_type='paragraph', node_text='Solve x = 1.'
        )
    ]
    source_procedure = [
        models.TextNodeInput(
            local_index=1, node_type='paragraph', node_text='Subtract one.'
        )
    ]
    encoded = writer.encode(statement, source_procedure, '')
    assert encoded['statement'] == statement
    assert encoded['source_procedure'] == source_procedure
    assert writer.decode(SimpleNamespace(procedure='  proof  ')) == '  proof  '
    with pytest.raises(ValueError, match='procedure must be a non-empty string'):
        writer.decode(SimpleNamespace(procedure=''))


def test_entity_and_predicate_enrichment_use_directional_text_inputs():
    context_before = [_node(0, 'Before')]
    target_node = context_window.ContextNode(
        position=0,
        type='image',
        content='A diagram',
        assets=[models.VisualAsset(path='figure.png')],
    )
    context_after = [_node(0, 'After')]
    entity = entity_enrichment.EntityEnricher.__new__(
        entity_enrichment.EntityEnricher
    )
    predicate = predicate_enrichment.PredicateEnricher.__new__(
        predicate_enrichment.PredicateEnricher
    )
    entity_request = models.TermEnrichmentInput(
        context_before=[context_window.node_input(node) for node in context_before],
        target_node=context_window.node_input(target_node),
        context_after=[
            context_window.node_input(node) for node in context_after
        ],
        terms=['vector'],
    )
    predicate_request = entity_request.model_copy(update={'terms': ['moves']})
    entity_input = entity.encode(entity_request)
    predicate_input = predicate.encode(predicate_request)
    for encoded in (entity_input, predicate_input):
        assert list(encoded) == ['request']
        assert encoded['request'].target_node.text == 'A diagram'
        assert encoded['request'].target_node.index == 1
        assert not hasattr(encoded['request'].target_node, 'assets')
        assert not hasattr(encoded['request'].target_node, 'path')
    entity_result = entity_enrichment.TermDescription(
        term='vector', description='A directed quantity.'
    )
    predicate_result = predicate_enrichment.TermDescription(
        term='moves', description='Changes position.'
    )
    assert entity.decode(
        SimpleNamespace(description=entity_result.description),
        **entity_input,
    ) == [entity_result]
    assert predicate.decode(
        SimpleNamespace(description=predicate_result.description),
        **predicate_input,
    ) == [predicate_result]
    with pytest.raises(ValueError, match='non-empty string'):
        entity.decode(SimpleNamespace(description=''), **entity_input)
    with pytest.raises(ValueError, match='non-empty string'):
        predicate.decode(SimpleNamespace(description=[]), **predicate_input)
    with pytest.raises(ValueError, match='expects one input term'):
        entity.decode(
            SimpleNamespace(description='A description.'),
            request=entity_request.model_copy(update={'terms': ['vector', 'field']}),
        )


def test_predicate_enrichment_rejects_verbose_source_restatements():
    predicate = predicate_enrichment.PredicateEnricher.__new__(
        predicate_enrichment.PredicateEnricher
    )
    terms = ['is defined as']
    result = predicate.decode(
        SimpleNamespace(
            description=(
                'The function f is defined by the expression x squared '
                'plus y squared at point P.'
            )
        ),
        request=models.TermEnrichmentInput(
            context_before=[],
            target_node=models.NodeInput(index=1, node_type='paragraph', text=''),
            context_after=[],
            terms=terms,
        ),
    )
    assert result == [
        predicate_enrichment.TermDescription(
            term='is defined as',
            description='is defined as',
        )
    ]
    prompt = predicate_enrichment.PredicateEnrichmentSignature.__doc__
    assert '2–12 plain-language words' in prompt
    assert 'no LaTeX' in prompt


def test_enrichment_request_records_all_use_standard_node_keys():
    before = [context_window.ContextNode(position=0, type='paragraph', content='Before')]
    target = context_window.ContextNode(position=0, type='paragraph', content='Target')
    after = [context_window.ContextNode(position=0, type='paragraph', content='After')]
    request = models.TermEnrichmentInput(
        context_before=[context_window.node_input(n, i) for i, n in enumerate(before)],
        target_node=context_window.node_input(target),
        context_after=[context_window.node_input(n, i) for i, n in enumerate(after)],
        terms=['term'],
    )
    entity = entity_enrichment.EntityEnricher.__new__(entity_enrichment.EntityEnricher)
    predicate = predicate_enrichment.PredicateEnricher.__new__(
        predicate_enrichment.PredicateEnricher
    )
    for enricher in (entity, predicate):
        encoded = enricher.encode(request)
        assert list(encoded) == ['request']
        for record in (
            encoded['request'].context_before,
            encoded['request'].context_after,
        ):
            for node in record:
                assert set(node.model_dump()) == {'index', 'node_type', 'text'}
        target_record = encoded['request'].target_node
        assert set(target_record.model_dump()) == {'index', 'node_type', 'text'}


def test_governance_projection_uses_standard_keys_and_never_exposes_assets():
    instruction = context_window.ContextNode(
        position=0,
        type='paragraph',
        content='Instruction',
        assets=[models.VisualAsset(path='img.png')],
    )
    statement = context_window.ContextNode(
        position=0,
        type='paragraph',
        content='Statement',
        marker='target',
    )
    judge = governance_judge.GovernanceJudge.__new__(governance_judge.GovernanceJudge)
    encoded = judge.encode(
        instruction_nodes=[instruction],
        context_before=[context_window.ContextNode(position=0, type='paragraph', content='B')],
        statement_nodes=[statement],
        context_after=[context_window.ContextNode(position=0, type='paragraph', content='A')],
    )
    for field_name in ('instruction_nodes', 'context_before', 'statement_nodes', 'context_after'):
        for node in encoded[field_name]:
            assert set(node.model_dump()) == {'index', 'node_type', 'text'}
            assert not hasattr(node, 'assets')
            assert not hasattr(node, 'marker')


def test_signature_field_inventories_match_canonical_boundary():
    assert set(triplet_extractor._FactSignature.input_fields) == {'request'}
    assert set(triplet_extractor._FactSignature.output_fields) == {'facts'}
    assert set(entity_enrichment.EntityEnrichmentSignature.input_fields) == {'request'}
    assert set(predicate_enrichment.PredicateEnrichmentSignature.input_fields) == {'request'}
    assert set(
        governance_judge.GovernanceJudgeSignature.input_fields
    ) == {'instruction_nodes', 'context_before', 'statement_nodes', 'context_after'}



def test_hub_mentions_normalize_once_at_the_adjudication_boundary():
    from kms.construction import local_entity_hubs, local_predicate_hubs

    mention = models.HubMentionInput.from_record(
        {
            'name': 'vector',
            'aliases': ['vector', 'directed segment'],
            'description': 'An oriented quantity.',
        }
    )
    assert mention.name == 'vector'
    assert mention.aliases == ['directed segment']
    assert mention.description == 'An oriented quantity.'
    with pytest.raises(ValueError, match='extra'):
        models.HubMentionInput(name='vector', unsupported=True)

    left = models.HubMentionInput(name='vector')
    right = models.HubMentionInput(name='field')
    for adjudicator_type in (
        local_entity_hubs.EntityHubAdjudicator,
        local_predicate_hubs.PredicateHubAdjudicator,
    ):
        encoded = adjudicator_type.__new__(adjudicator_type).encode(
            left=left,
            right=right,
            scope='test scope',
        )
        comparison = encoded['comparison']
        assert isinstance(comparison, models.HubMentionComparisonInput)
        assert comparison.left is left
        assert comparison.right is right
        assert comparison.scope == 'test scope'


def test_hub_synthesis_and_adjudication_outputs_are_strict():
    from kms.construction import (
        local_entity_hubs,
        local_predicate_hubs,
        local_procedure_hubs,
        local_statement_hubs,
        triplet_hubs,
    )

    synthesis_cases = [
        (local_entity_hubs.EntityHubSynthesizer, local_entity_hubs.EntityHubDefinition),
        (
            local_predicate_hubs.PredicateHubSynthesizer,
            local_predicate_hubs.PredicateHubDefinition,
        ),
        (
            local_statement_hubs.LocalStatementHubSynthesizer,
            local_statement_hubs.LocalStatementHubResult,
        ),
        (
            local_procedure_hubs.LocalProcedureHubSynthesizer,
            local_procedure_hubs.LocalProcedureHubResult,
        ),
        (
            triplet_hubs._TripletDefinitionSynthesizer,
            triplet_hubs._TripletDefinition,
        ),
    ]
    for synthesizer_type, result_type in synthesis_cases:
        synthesizer = synthesizer_type.__new__(synthesizer_type)
        result = result_type(canonical_name='Concept', description='Meaning.')
        assert synthesizer.decode(SimpleNamespace(result=result)) == (
            'Concept',
            'Meaning.',
        )
        with pytest.raises(ValueError, match='canonical_name'):
            synthesizer.decode(
                SimpleNamespace(
                    result=result_type(canonical_name='', description='Meaning.')
                )
            )

    statement = local_statement_hubs.LocalStatementHubAdjudicator.__new__(
        local_statement_hubs.LocalStatementHubAdjudicator
    )
    procedure = local_procedure_hubs.LocalProcedureHubAdjudicator.__new__(
        local_procedure_hubs.LocalProcedureHubAdjudicator
    )
    assert statement.decode(
        SimpleNamespace(result=models.MergeDecision(should_merge=True))
    ) is True
    assert procedure.decode(
        SimpleNamespace(result=models.MergeDecision(should_merge=False))
    ) is False
    with pytest.raises(ValidationError):
        statement.decode(SimpleNamespace(result={'should_merge': 'true'}))
    with pytest.raises(ValidationError):
        procedure.decode(SimpleNamespace(result={'should_merge': 1}))


def test_entity_and_predicate_adjudicators_use_boolean_decisions():
    from kms.construction import local_entity_hubs, local_predicate_hubs

    entity = local_entity_hubs.EntityHubAdjudicator.__new__(
        local_entity_hubs.EntityHubAdjudicator
    )
    predicate = local_predicate_hubs.PredicateHubAdjudicator.__new__(
        local_predicate_hubs.PredicateHubAdjudicator
    )
    result = models.MergeDecision(should_merge=True)
    assert entity.decode(SimpleNamespace(result=result)) is True
    assert predicate.decode(SimpleNamespace(result=result)) is True
    with pytest.raises(ValidationError):
        entity.decode(SimpleNamespace(result={'should_merge': 'true'}))
    with pytest.raises(ValidationError):
        predicate.decode(SimpleNamespace(result={'should_merge': 1}))
