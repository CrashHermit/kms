"""Tests for core construction data models."""

from kms.core import models


def test_construction_bundle_has_independent_collection_defaults() -> None:
    first = models.ConstructionBundle(source=models.Source(key='first'))
    second = models.ConstructionBundle(source=models.Source(key='second'))

    first.nodes.append(models.SourceNode(content='one'))
    first.derived['embedding'] = [1.0]
    first.diagnostics.append('warning')

    assert second.nodes == []
    assert second.derived == {}
    assert second.diagnostics == []


def test_construction_bundle_retains_typed_source_records() -> None:
    source = models.Source(key='book')
    node = models.SourceNode(type=models.NodeType.PARAGRAPH, content='A fact.')
    instruction = models.Instruction(block=[0], member_positions=[0])
    statement = models.Statement(block=[0], member_positions=[0])
    procedure = models.Procedure(block=[1], member_positions=[1])
    triplet = models.Triplet('A', 'is', 'fact', evidence_positions=[0])

    bundle = models.ConstructionBundle(
        source=source,
        nodes=[node],
        instructions=[instruction],
        statements=[statement],
        procedures=[procedure],
        triplets=[triplet],
    )

    assert bundle.source is source
    assert bundle.nodes == [node]
    assert bundle.instructions == [instruction]
    assert bundle.statements == [statement]
    assert bundle.procedures == [procedure]
    assert bundle.triplets == [triplet]


def test_construction_bundle_uses_plain_typed_collections() -> None:
    bundle = models.ConstructionBundle(source=models.Source(key='source'))

    assert isinstance(bundle.nodes, list)
    assert isinstance(bundle.instructions, list)
    assert isinstance(bundle.statements, list)
    assert isinstance(bundle.procedures, list)
    assert isinstance(bundle.triplets, list)
    assert isinstance(bundle.derived, dict)
    assert isinstance(bundle.indexes, dict)


def _complete_bundle() -> models.ConstructionBundle:
    source = models.Source(
        key='book',
        documents=[models.Document(index=3, image_path='page.png')],
    )
    nodes = [
        models.SourceNode(uuid='node-0', document_index=3, content='first'),
        models.SourceNode(uuid='node-1', document_index=3, content='second'),
    ]
    return models.ConstructionBundle(
        source=source,
        nodes=nodes,
        instructions=[models.Instruction(block=[0, 1], member_positions=[1, 0])],
        statements=[models.Statement(block=[0], member_positions=[0])],
        procedures=[models.Procedure(block=[1], member_positions=[1])],
        triplets=[models.Triplet('first', 'is', 'second', evidence_positions=[1, 0])],
    )


def test_validate_bundle_accepts_valid_noncontiguous_ids() -> None:
    models.validate_bundle(_complete_bundle())


def test_validate_bundle_rejects_missing_and_duplicate_node_ids() -> None:
    bundle = _complete_bundle()
    # Validation now checks for missing UUIDs
    bundle.nodes[0].uuid = None
    bundle.nodes[1].uuid = None

    try:
        models.validate_bundle(bundle, require_identities=True)
    except models.BundleValidationError as error:
        assert any('missing a uuid' in message for message in error.errors)
    else:
        raise AssertionError('expected BundleValidationError')

    bundle = _complete_bundle()
    bundle.nodes[1].uuid = bundle.nodes[0].uuid
    try:
        models.validate_bundle(bundle)
    except models.BundleValidationError as error:
        assert any('nodes contain duplicate uuids' in message for message in error.errors)
    else:
        raise AssertionError('expected BundleValidationError')


def test_validate_bundle_rejects_bad_ownership_membership_and_evidence() -> None:
    bundle = _complete_bundle()
    bundle.nodes[0].document_index = 99
    bundle.instructions[0].member_positions = [42, 42, 999]
    bundle.triplets[0].evidence_positions = [42, 999, 42]

    try:
        models.validate_bundle(bundle)
    except models.BundleValidationError as error:
        assert any(
            'references missing document 99' in message
            for message in error.errors
        )
        assert 'instruction 0 member_positions contain duplicates' in error.errors
        assert 'instruction 0 references missing node 999' in error.errors
        assert 'triplet 0 evidence contains duplicates' in error.errors
        assert 'triplet 0 references missing node 999' in error.errors
    else:
        raise AssertionError('expected BundleValidationError')


def test_validate_bundle_rejects_source_and_page_contracts() -> None:
    source = models.Source(
        key=' ',
        documents=[
            models.Document(index=1, image_path='one.png'),
            models.Document(index=1, image_path='two.png'),
        ],
    )
    bundle = models.ConstructionBundle(
        source=source,
        nodes=[models.SourceNode(uuid='node-1', document_index=1)],
        triplets=[models.Triplet('a', 'b', 'c')],
    )

    try:
        models.validate_bundle(bundle)
    except models.BundleValidationError as error:
        assert 'source key must be non-empty' in error.errors
        assert 'source documents contain duplicate indexes' in error.errors
        assert 'triplet 0 must have evidence node positions' in error.errors
    else:
        raise AssertionError('expected BundleValidationError')


def test_validate_bundle_does_not_mutate_bundle() -> None:
    bundle = _complete_bundle()
    before = repr(bundle)

    models.validate_bundle(bundle)

    assert repr(bundle) == before
