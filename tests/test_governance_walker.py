import asyncio

from kms.construction import governance_walker
from kms.core import models


class _Judge:
    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return True


def test_governance_judges_complete_statement_with_static_context():
    nodes = [
        models.SourceNode(uuid='node-0', type='paragraph', content='Instruction'),
        models.SourceNode(uuid='node-1', type='paragraph', content='Before'),
        models.SourceNode(uuid='node-2', type='paragraph', content='Statement'),
        models.SourceNode(uuid='node-3', type='paragraph', content='After'),
    ]
    instruction = models.Instruction(
        block=[0], member_positions=[0], uuid='instruction-uuid'
    )
    statement = models.Statement(block=[2], member_positions=[2], uuid='statement-uuid')
    judge = _Judge()
    node = governance_walker.GovernanceStatementWalkerNode(
        judge=judge,
        backward_budget=100,
        forward_budget=100,
    )

    result = asyncio.run(
        node.run(
            {
                'nodes': nodes,
                'instructions': [instruction],
                'statements': [statement],
                'procedures': [],
            }
        )
    )

    assert result['statements'][0].instruction_uuids == ['instruction-uuid']
    assert 'procedures' not in result
    call = judge.calls[0]
    assert all(
        isinstance(item, governance_walker.context_window.ContextNode)
        for item in call['statement_nodes']
    )
    assert call['statement_nodes'][0].content == 'Statement'
    assert [node.content for node in call['context_before']] == [
        'Instruction',
        'Before',
    ]
    assert [node.content for node in call['context_after']] == ['After']
