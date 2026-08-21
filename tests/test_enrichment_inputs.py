"""Tests for pure enrichment input assembly."""

from kms.construction import statement_enrichment, procedure_enrichment
from kms.core import models


def _assertion(uuid: str, evidence: set[int]) -> models.KnowledgeAssertion:
    concept = models.CanonicalConcept('c', 'concept', 'concept')
    predicate = models.CanonicalPredicate('p', 'relates', 'relation')
    return models.KnowledgeAssertion(
        uuid=uuid,
        name=uuid,
        description=uuid,
        subject=concept,
        predicate=predicate,
        object=concept,
        evidence_node_ids=frozenset(evidence),
    )


def _bundle() -> models.ConstructionBundle:
    return models.ConstructionBundle(
        source=models.Source(key='book'),
        nodes=[
            models.Node(uuid='node-1', content='statement'),
            models.Node(uuid='node-2', content='procedure'),
        ],
        statements=[models.Statement(block=[1], members=[0], uuid='statement-1')],
        knowledge_index=models.KnowledgeIndex(
            source='book',
            assertions=(_assertion('statement-fact', {0}), _assertion('procedure-fact', {1})),
        ),
    )


def test_statement_enrichment_input_is_pure_and_typed() -> None:
    bundle = _bundle()
    statement = bundle.statements[0]

    result = statement_enrichment.statement_enrichment_input(bundle, statement)

    assert result.statement.render() == 'statement'
    assert 'statement-fact' in result.canonical_knowledge


def test_procedure_enrichment_input_explicitly_unions_statement_and_procedure_knowledge() -> None:
    bundle = _bundle()
    procedure = models.Procedure(
        block=[1],
        statement_uuid='statement-1',
        uuid='procedure-1',
        members=[1],
    )

    result = procedure_enrichment.procedure_enrichment_input(
        bundle, bundle.statements[0], procedure
    )

    assert 'procedure-fact' in result.canonical_knowledge
    assert result.statement.render() == 'statement'
    assert result.procedure.render() == 'procedure'