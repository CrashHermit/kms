import asyncio

from kms.core import identity, models
from kms.graph import instructions, nodes, writer


def _instruction(block=None, member_positions=None):
    block = block if block is not None else [0, 1]
    return models.Instruction(
        block=list(block),
        member_positions=member_positions
        if member_positions is not None
        else list(block),
        uuid=instructions.instruction_uuid('ea2e.pdf', list(block)),
    )


def test_instruction_uuid_is_deterministic():
    assert instructions.instruction_uuid('ea2e.pdf', [0, 1]) == (
        instructions.instruction_uuid('ea2e.pdf', [0, 1])
    )


def test_instruction_uuid_distinguishes_block_and_source():
    assert instructions.instruction_uuid('ea2e.pdf', [0, 1]) != (
        instructions.instruction_uuid('ea2e.pdf', [1, 0])
    )
    assert instructions.instruction_uuid('ea2e.pdf', [0, 1]) != (
        instructions.instruction_uuid('other.pdf', [0, 1])
    )


def test_instruction_uuids_are_disjoint_from_other_tiers():
    key = instructions.instruction_uuid('book.pdf', [7])
    # Create a mock node with the expected UUID
    mock_node = models.SourceNode(uuid='node-uuid-7', document_index=0)
    assert key != nodes.node_uuid('book.pdf', mock_node)
    assert key != identity.statement_uuid('book.pdf', [7])
    assert key != nodes.source_uuid('book.pdf')


def test_properties_carry_only_identity():
    props = instructions.instruction_properties(_instruction(), 'ea2e.pdf')
    assert props['uuid'] == instructions.instruction_uuid('ea2e.pdf', [0, 1])
    assert props['source'] == nodes.source_uuid('ea2e.pdf')
    assert 'text' not in props
    assert 'directive' not in props


def test_instruction_member_pairs_one_per_member():
    node_stream = [
        models.SourceNode(uuid=f'node-{index}') for index in range(4)
    ]
    pairs = instructions.instruction_member_pairs(
        [_instruction(block=[0], member_positions=[1, 2, 3])],
        node_stream,
        'ea2e.pdf',
    )
    assert len(pairs) == 3
    assert {pair['node'] for pair in pairs} == {
        nodes.node_uuid('ea2e.pdf', node_stream[index]) for index in [1, 2, 3]
    }
    assert {pair['instruction'] for pair in pairs} == {
        instructions.instruction_uuid('ea2e.pdf', [0])
    }


def test_instruction_member_pairs_empty_without_members():
    pairs = instructions.instruction_member_pairs(
        [_instruction(block=[0], member_positions=[])], [], 'ea2e.pdf'
    )
    assert pairs == []


def test_instruction_governance_pairs_link_instruction_to_statement():
    instruction = _instruction(block=[0], member_positions=[0])
    statement = models.Statement(
        block=[1],
        uuid=identity.statement_uuid('ea2e.pdf', [1]),
        instruction_uuids=[instruction.uuid],
    )
    assert instructions.instruction_governance_pairs(
        [statement], [instruction], 'ea2e.pdf'
    ) == [
        {
            'instruction': instruction.uuid,
            'statement': statement.uuid,
        }
    ]


def test_instruction_governance_pairs_reject_unknown_instruction():
    statement = models.Statement(
        block=[1],
        uuid=identity.statement_uuid('ea2e.pdf', [1]),
        instruction_uuids=['missing'],
    )
    try:
        instructions.instruction_governance_pairs([statement], [], 'ea2e.pdf')
    except ValueError as exc:
        assert 'unknown instruction' in str(exc)
    else:
        raise AssertionError('unknown instruction was accepted')


class _FakeSession:
    def __init__(self, log):
        self._log = log

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self._log.append((query, params))
        return None


class _FakeDriver:
    def __init__(self):
        self.queries = []

    def session(self, **_):
        return _FakeSession(self.queries)


def _node_stream():
    return [models.SourceNode(uuid=f'node-{index}') for index in range(3)]


def test_persist_instructions_writes_hubs_then_member_edges():
    fake = _FakeDriver()

    asyncio.run(
        writer.persist_instructions(
            [_instruction(block=[0], member_positions=[1, 2])],
            _node_stream(),
            'ea2e.pdf',
            session_factory=lambda: fake.session(database='neo4j'),
        )
    )

    assert len(fake.queries) == 2
    hub_query, hub_params = fake.queries[0]
    assert f'MERGE (i:{instructions.INSTRUCTION_LABEL}' in hub_query
    assert len(hub_params['rows']) == 1

    edge_query, edge_params = fake.queries[1]
    assert 'MEMBER_OF]->(i)' in edge_query
    assert len(edge_params['pairs']) == 2


def test_persist_instruction_governance_writes_governs_edges():
    fake = _FakeDriver()
    instruction = _instruction(block=[0], member_positions=[0])
    statement = models.Statement(
        block=[1],
        uuid=identity.statement_uuid('ea2e.pdf', [1]),
        instruction_uuids=[instruction.uuid],
    )
    asyncio.run(
        writer.persist_instruction_governance(
            [statement],
            [instruction],
            'ea2e.pdf',
            session_factory=lambda: fake.session(database='neo4j'),
        )
    )
    assert len(fake.queries) == 1
    query, params = fake.queries[0]
    assert 'GOVERNS]->(s)' in query
    assert params['pairs'] == [
        {'instruction': instruction.uuid, 'statement': statement.uuid}
    ]


def test_persist_instructions_is_a_noop_when_empty():
    fake = _FakeDriver()
    asyncio.run(
        writer.persist_instructions(
            [],
            [],
            'ea2e.pdf',
            session_factory=lambda: fake.session(database='neo4j'),
        )
    )
    assert fake.queries == []
