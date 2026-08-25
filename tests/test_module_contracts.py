from types import SimpleNamespace

import pytest

from kms.construction import (
    block_corrector,
    entity_enrichment,
    formatter,
    governance_judge,
    predicate_enrichment,
    procedure_enrichment,
    statement_procedure_builder,
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
        isinstance(item, governance_judge.GovernanceNodeInput)
        for field in encoded.values()
        for item in field
    )
    assert encoded['statement_nodes'][0].node_text == 'A diagram of the rule.'
    assert encoded['context_before'][0].node_text == 'Before'
    assert encoded['context_after'][0].node_text == 'After'
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
        isinstance(
            item, statement_procedure_builder.StatementProcedureNodeInput
        )
        for item in role_encoded['current_nodes']
    )
    assert role_encoded['current_nodes'][1].node_text == (
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
        isinstance(
            item, statement_procedure_builder.StatementProcedureNodeInput
        )
        for item in encoded['current_nodes']
    )
    empty_image = partitioner.encode(
        current_nodes=[
            context_window.ContextNode(position=2, type='image', content=None)
        ]
    )['current_nodes']
    assert empty_image == [
        statement_procedure_builder.StatementProcedureNodeInput(
            local_index=2,
            node_type='image',
            node_text='',
        )
    ]
    assert partitioner.decode(
        SimpleNamespace(statement_positions=[0]), current_nodes=nodes
    ) == [0]
    with pytest.raises(ValueError, match='outside'):
        partitioner.decode(
            SimpleNamespace(statement_positions=[2]), current_nodes=nodes
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
    entity_input = entity.encode(
        context_before, target_node, context_after, ['vector']
    )
    predicate_input = predicate.encode(
        context_before, target_node, context_after, ['moves']
    )
    for encoded in (entity_input, predicate_input):
        assert list(encoded) == [
            'context_before',
            'target_node',
            'context_after',
            'terms',
        ]
        assert encoded['target_node'].node_text == 'A diagram'
        assert not hasattr(encoded['target_node'], 'assets')
        assert not hasattr(encoded['target_node'], 'path')
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
            terms=['vector', 'field'],
        )


def test_hub_mentions_normalize_once_at_the_adjudication_boundary():
    from kms.construction import entity_hubs, predicate_hubs

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
        entity_hubs.EntityHubAdjudicator,
        predicate_hubs.PredicateHubAdjudicator,
    ):
        encoded = adjudicator_type.__new__(adjudicator_type).encode(
            left=left,
            right=right,
            scope='test scope',
        )
        assert encoded == {
            'left': left,
            'right': right,
            'scope': 'test scope',
        }
        assert encoded['left'] is left
        assert encoded['right'] is right


def test_hub_synthesis_and_adjudication_outputs_are_strict():
    from kms.construction import (
        entity_hubs,
        local_procedure_hubs,
        local_statement_hubs,
        predicate_hubs,
        triplet_hubs,
    )

    synthesis_cases = [
        (entity_hubs.EntityHubSynthesizer, entity_hubs.EntityHubDefinition),
        (
            predicate_hubs.PredicateHubSynthesizer,
            predicate_hubs.PredicateHubDefinition,
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
    assert statement.decode(SimpleNamespace(should_merge=True)) is True
    assert procedure.decode(SimpleNamespace(should_merge=False)) is False
    with pytest.raises(ValueError, match='should_merge must be a boolean'):
        statement.decode(SimpleNamespace(should_merge='true'))
    with pytest.raises(ValueError, match='should_merge must be a boolean'):
        procedure.decode(SimpleNamespace(should_merge=1))


def test_dspy_structured_output_types_are_not_silently_coerced():
    from kms.construction import entity_hubs, predicate_hubs

    entity = entity_hubs.EntityHubAdjudicator.__new__(
        entity_hubs.EntityHubAdjudicator
    )
    predicate = predicate_hubs.PredicateHubAdjudicator.__new__(
        predicate_hubs.PredicateHubAdjudicator
    )
    entity_result = entity_hubs.EntityHubAdjudication(
        decision='separate', reason='different concepts'
    )
    predicate_result = predicate_hubs.PredicateHubAdjudication(
        decision='separate', reason='different relations'
    )
    assert entity.decode(SimpleNamespace(result=entity_result)) is entity_result
    assert (
        predicate.decode(SimpleNamespace(result=predicate_result))
        is predicate_result
    )
    with pytest.raises(TypeError, match='EntityHubAdjudication'):
        entity.decode(SimpleNamespace(result={'decision': 'separate'}))
    with pytest.raises(TypeError, match='PredicateHubAdjudication'):
        predicate.decode(SimpleNamespace(result={'decision': 'separate'}))
