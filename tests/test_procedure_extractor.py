"""Procedure extractor: nearest-preceding attachment, orphan handling, universal decomposition,
and the graph-node wrapper. The LLM decomposition call is stubbed."""

import asyncio

from kms.core import models
from kms.entity import procedure_extractor


def _nodes():
    # 0 theorem, 1 its proof, 2 example, 3 its solution
    contents = [
        'Theorem 1.1. Every group has an identity.',
        'Proof. Take e. Then e is unique.',
        'Example 2. Solve x + 1 = 3.',
        'Solution. Subtract 1. So x = 2.',
    ]
    return [
        models.ASTNode(
            type=models.NodeType.PARAGRAPH, content=text, id=i, segment_index=0
        )
        for i, text in enumerate(contents)
    ]


class _ScriptedModule:
    """A stand-in Module that splits content on ' | ' so decomposition is deterministic."""

    def __init__(self):
        self.seen: list[str] = []

    async def steps(self, contents):
        self.seen.append(contents)
        return [part.strip() for part in contents.split('.') if part.strip()]


def _by_id(nodes):
    return {node.id: node for node in nodes}


def _order(nodes):
    return {node.id: i for i, node in enumerate(nodes)}


# --- attachment ---


def test_attaches_each_procedure_to_the_nearest_preceding_block():
    nodes = _nodes()
    entities = [
        models.Entity(members=[0], type='theorem'),
        models.Entity(members=[2], type='example'),
    ]
    by_entity, orphans = procedure_extractor.attach(
        entities, [[1], [3]], _order(nodes)
    )
    assert by_entity == {0: [[1]], 1: [[3]]}
    assert orphans == []


def test_a_procedure_preceding_every_block_is_an_orphan_not_a_drop():
    nodes = _nodes()
    entities = [models.Entity(members=[2], type='example')]
    by_entity, orphans = procedure_extractor.attach(
        entities, [[1]], _order(nodes)
    )
    assert by_entity == {}
    assert orphans == [[1]]  # kept for a later attachment pass, never discarded


def test_a_block_can_own_several_procedures_in_document_order():
    nodes = _nodes()
    entities = [models.Entity(members=[0], type='theorem')]
    by_entity, orphans = procedure_extractor.attach(
        entities, [[3], [1]], _order(nodes)
    )
    assert by_entity == {0: [[1], [3]]}  # sorted into document order
    assert orphans == []


# --- decomposition ---


def test_decomposition_is_universal_so_a_solution_gets_steps_too():
    nodes = _nodes()
    entities = [
        models.Entity(members=[0], type='theorem'),
        models.Entity(members=[2], type='example'),
    ]
    module = _ScriptedModule()
    orphans = asyncio.run(
        procedure_extractor.extract_procedures(
            entities, [[1], [3]], _by_id(nodes), module
        )
    )
    assert orphans == []
    assert entities[0].procedures[0].steps == [
        'Proof',
        'Take e',
        'Then e is unique',
    ]
    # the example's solution is decomposed exactly like the proof — no Thm/Def restriction
    assert entities[1].procedures[0].steps == [
        'Solution',
        'Subtract 1',
        'So x = 2',
    ]


def test_a_procedure_carries_its_own_members_for_provenance():
    nodes = _nodes()
    entities = [models.Entity(members=[0], type='theorem')]
    asyncio.run(
        procedure_extractor.extract_procedures(
            entities, [[1]], _by_id(nodes), _ScriptedModule()
        )
    )
    assert entities[0].procedures[0].members == [1]


def test_several_procedures_on_one_block_are_indexed_in_order():
    nodes = _nodes()
    entities = [models.Entity(members=[0], type='theorem')]
    asyncio.run(
        procedure_extractor.extract_procedures(
            entities, [[1], [3]], _by_id(nodes), _ScriptedModule()
        )
    )
    assert [proc.index for proc in entities[0].procedures] == [0, 1]


def test_orphans_are_decomposed_and_returned_not_dropped():
    nodes = _nodes()
    entities = [models.Entity(members=[2], type='example')]
    orphans = asyncio.run(
        procedure_extractor.extract_procedures(
            entities, [[1]], _by_id(nodes), _ScriptedModule()
        )
    )
    assert len(orphans) == 1
    assert orphans[0].steps  # extracted content survives, it just has no owner
    assert entities[0].procedures == []


# --- graph node ---


def test_node_run_writes_the_entities_channel():
    nodes = _nodes()
    entities = [models.Entity(members=[0], type='theorem')]
    node = procedure_extractor.ProcedureExtractorNode(module=_ScriptedModule())
    out = asyncio.run(
        node.run(
            {'nodes': nodes, 'entities': entities, 'procedure_spans': [[1]]}
        )
    )
    assert list(out) == ['entities']
    assert out['entities'][0].procedures[0].steps


def test_node_run_without_procedure_spans_leaves_blocks_untouched():
    # absence is structural: no procedure span means no procedure, with no semantic call
    nodes = _nodes()
    entities = [models.Entity(members=[0], type='definition')]
    node = procedure_extractor.ProcedureExtractorNode(module=_ScriptedModule())
    out = asyncio.run(
        node.run({'nodes': nodes, 'entities': entities, 'procedure_spans': []})
    )
    assert out['entities'][0].procedures == []
