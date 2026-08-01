"""Member partitioners: line-finding for both-blocks, pure logic with a
scripted module."""

import asyncio

from kms.core import models
from kms.ingestion import partitioner


class _ScriptedModule:
    def __init__(self, positions):
        self._positions = list(positions)

    async def aforward(self, current_nodes):
        return self._positions.pop(0)


def _nodes():
    return {
        0: models.ParagraphNode(content='Example 4.2. Compute ...', id=0),
        1: models.ParagraphNode(content='Integrate ...', id=1),
        2: models.ParagraphNode(content='Hence the value is 4.', id=2),
    }


def _statement(members=(0, 1, 2)):
    return models.Statement(block=[0, 1, 2], members=list(members))


def _procedure(block=(0, 1, 2), members=(0, 1, 2)):
    return models.Procedure(block=list(block), members=list(members))


def test_partition_statements_narrows_a_both_block_statement():
    statement = _statement()
    procedure = _procedure()
    module = _ScriptedModule([[0]])
    asyncio.run(
        partitioner.partition_statements(
            [statement], [procedure], _nodes(), module
        )
    )
    assert statement.members == [0]
    # The block (the identity) is untouched by partitioning.
    assert statement.block == [0, 1, 2]


def test_partition_procedures_narrows_a_both_block_procedure():
    statement = _statement()
    procedure = _procedure()
    module = _ScriptedModule([[1, 2]])
    asyncio.run(
        partitioner.partition_procedures(
            [statement], [procedure], _nodes(), module
        )
    )
    assert procedure.members == [1, 2]
    assert procedure.block == [0, 1, 2]


def test_partition_statements_skips_single_role_statements():
    statement = _statement()
    module = _ScriptedModule([])  # would fail if called
    asyncio.run(
        partitioner.partition_statements([statement], [], _nodes(), module)
    )
    assert statement.members == [0, 1, 2]


def test_partition_procedures_skips_single_role_procedures():
    procedure = _procedure()
    module = _ScriptedModule([])  # would fail if called
    asyncio.run(
        partitioner.partition_procedures([], [procedure], _nodes(), module)
    )
    assert procedure.members == [0, 1, 2]


def test_an_empty_selection_keeps_the_full_block():
    statement = _statement()
    procedure = _procedure()
    module = _ScriptedModule([[]])
    asyncio.run(
        partitioner.partition_statements(
            [statement], [procedure], _nodes(), module
        )
    )
    assert statement.members == [0, 1, 2]  # never emptied by a bad answer


def test_out_of_range_positions_are_dropped():
    statement = _statement()
    procedure = _procedure()
    module = _ScriptedModule([[0, 99]])
    asyncio.run(
        partitioner.partition_statements(
            [statement], [procedure], _nodes(), module
        )
    )
    assert statement.members == [0]
