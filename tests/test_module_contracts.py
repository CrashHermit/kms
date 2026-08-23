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
from kms.core import content, walker


def _node(position: int, text: str = 'text') -> walker.WindowNode:
    return walker.WindowNode(
        position=position,
        type='paragraph',
        content=text,
    )


def test_governance_judge_preserves_three_multimodal_inputs_and_validates():
    judge = governance_judge.GovernanceJudge.__new__(
        governance_judge.GovernanceJudge
    )
    instruction = content.Content.from_text('Use this rule.')
    statement = content.Content.from_text('A statement.')
    context = [_node(0, 'Context.')]

    encoded = judge.encode(instruction, statement, context)
    assert all(
        isinstance(encoded[name], content.ContentParts)
        for name in (
            'instruction_directive',
            'statement_content',
            'context_window',
        )
    )
    assert judge.decode(
        SimpleNamespace(governs=True, confidence=0.75), **encoded
    ) == (True, 0.75)
    with pytest.raises(ValueError, match='governs must be a boolean'):
        judge.decode(
            SimpleNamespace(governs='true', confidence=0.75), **encoded
        )


def test_role_and_partition_modules_validate_positions_against_original_nodes():
    role_typer = statement_procedure_builder.RoleTyper.__new__(
        statement_procedure_builder.RoleTyper
    )
    role_encoded = role_typer.encode(content.Content.from_text('A proof.'))
    assert isinstance(role_encoded['contents'], content.ContentParts)
    assert role_typer.decode(
        SimpleNamespace(has_statement=True, has_procedure=False),
        **role_encoded,
    ) == (True, False)

    nodes = [_node(0), _node(1)]
    partitioner = statement_procedure_builder.StatementPartitioner.__new__(
        statement_procedure_builder.StatementPartitioner
    )
    encoded = partitioner.encode(nodes)
    assert isinstance(encoded['current_nodes'], content.ContentParts)
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


def test_procedure_writer_encodes_empty_source_as_structured_content():
    writer = procedure_enrichment.ProcedureWriter.__new__(
        procedure_enrichment.ProcedureWriter
    )
    encoded = writer.encode(content.Content.from_text('Solve x = 1.'), None, '')
    source = encoded['source_procedure']
    assert isinstance(source, content.ContentParts)
    assert source.content.parts[0].text == ''
    assert writer.decode(SimpleNamespace(procedure='  proof  ')) == '  proof  '
    with pytest.raises(ValueError, match='procedure must be a non-empty string'):
        writer.decode(SimpleNamespace(procedure=''))


def test_entity_and_predicate_enrichment_require_single_structured_results():
    passage = content.Content.from_text('A vector moves.')
    entity = entity_enrichment.EntityEnricher.__new__(
        entity_enrichment.EntityEnricher
    )
    predicate = predicate_enrichment.PredicateEnricher.__new__(
        predicate_enrichment.PredicateEnricher
    )
    entity_input = entity.encode(passage, ['vector'])
    predicate_input = predicate.encode(passage, ['moves'])
    entity_result = entity_enrichment.TermDescription(
        term='vector', description='A directed quantity.'
    )
    predicate_result = predicate_enrichment.TermDescription(
        term='moves', description='Changes position.'
    )
    assert entity.decode(
        SimpleNamespace(description=entity_result.description), **entity_input
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


def test_hub_synthesis_and_adjudication_outputs_are_strict():
    from kms.construction import (
        entity_hubs,
        predicate_hubs,
        procedure_hubs,
        statement_hubs,
        triplet_hubs,
    )

    synthesis_cases = [
        (entity_hubs.EntityHubSynthesizer, entity_hubs.EntityHubDefinition),
        (
            predicate_hubs.PredicateHubSynthesizer,
            predicate_hubs.PredicateHubDefinition,
        ),
        (statement_hubs.StatementHubSynthesizer, statement_hubs.StatementHubResult),
        (procedure_hubs.ProcedureHubSynthesizer, procedure_hubs.ProcedureHubResult),
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

    statement = statement_hubs.StatementHubAdjudicator.__new__(
        statement_hubs.StatementHubAdjudicator
    )
    procedure = procedure_hubs.ProcedureHubAdjudicator.__new__(
        procedure_hubs.ProcedureHubAdjudicator
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
