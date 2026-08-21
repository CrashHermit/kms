import asyncio

from kms.construction import governance_walker
from kms.core import models


class _Judge:
    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        return True, 1.0


def test_governance_judges_complete_statement_with_static_context():
    nodes = [
        models.Node(uuid='node-0', type='paragraph', content='Instruction'),
        models.Node(uuid='node-1', type='paragraph', content='Before'),
        models.Node(uuid='node-2', type='paragraph', content='Statement'),
        models.Node(uuid='node-3', type='paragraph', content='After'),
    ]
    instruction = models.Instruction(
        block=[0], members=[0], uuid='instruction-uuid'
    )
    statement = models.Statement(block=[2], members=[2], uuid='statement-uuid')
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
    assert call['statement_content'].render() == 'Statement'
    # Check that context window has the right structure with positions
    context_window = call['context_window']
    assert len(context_window) == 4
    assert [item.position for item in context_window] == [0, 1, 2, 3]
    assert [item.marker for item in context_window] == [
        None,
        None,
        'statement',
        None,
    ]
