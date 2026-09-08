import asyncio

from kms.postprocessing.learning import entity_cards as entity_learning


class FakeGate:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inputs = []

    async def aforward(self, **inputs):
        self.inputs.append(inputs)
        return next(self.decisions)


def _row(local_uuid='local-1'):
    return {
        'global_uuid': 'global-1',
        'global_name': 'Group',
        'global_description': 'An algebraic structure.',
        'local_uuid': local_uuid,
        'local_name': 'Group',
        'local_description': 'A set with an operation.',
        'source': 'book.pdf',
        'evidence': [
            {'uuid': 'entity-1', 'name': 'Group', 'description': 'A set.'}
        ],
    }


def test_candidate_uses_structured_context_and_evidence():
    candidate = entity_learning.EntityLearningCandidate.from_row(_row())

    assert candidate.global_hub.name == 'Group'
    assert candidate.local_hub.description == 'A set with an operation.'
    assert candidate.evidence[0].description == 'A set.'


def test_indexed_decisions_require_complete_batch_coverage():
    candidate = entity_learning.EntityLearningCandidate.from_row(_row())
    candidate = candidate.with_index(1)
    batch, _ = entity_learning._batches([candidate], evidence=False)

    assert batch[0][0].local_hub.index == 1
    assert entity_learning.EntityLearningDecision(index=1, learnable=True)


def test_assess_runs_local_gate_only_after_global_true(monkeypatch):
    rows = [_row('local-1'), _row('local-2')]
    monkeypatch.setattr(
        entity_learning.queries,
        'entity_learning_candidates',
        lambda session_factory: _rows(rows),
    )
    global_gate = FakeGate(
        [
            [
                entity_learning.EntityLearningDecision(index=1, learnable=True),
                entity_learning.EntityLearningDecision(
                    index=2, learnable=False
                ),
            ]
        ]
    )
    local_gate = FakeGate(
        [[entity_learning.EntityLearningDecision(index=1, learnable=True)]]
    )

    result = asyncio.run(
        entity_learning.assess(
            session_factory=object(),
            global_gate=global_gate,
            evidence_gate=local_gate,
        )
    )

    assert result['global_accepted'] == 1
    assert result['local_accepted'] == 1
    assert len(local_gate.inputs) == 1
    assert isinstance(
        local_gate.inputs[0]['batch'],
        entity_learning.EntityLearningLocalBatchInput,
    )


def test_local_stage_preserves_sparse_global_indexes(monkeypatch):
    rows = [_row('local-1'), _row('local-2')]
    monkeypatch.setattr(
        entity_learning.queries,
        'entity_learning_candidates',
        lambda session_factory: _rows(rows),
    )
    global_gate = FakeGate(
        [
            [
                entity_learning.EntityLearningDecision(
                    index=1, learnable=False
                ),
                entity_learning.EntityLearningDecision(index=2, learnable=True),
            ]
        ]
    )
    local_gate = FakeGate(
        [[entity_learning.EntityLearningDecision(index=2, learnable=True)]]
    )

    result = asyncio.run(
        entity_learning.assess(
            session_factory=object(),
            global_gate=global_gate,
            evidence_gate=local_gate,
        )
    )

    assert result['local_decisions'][0]['local_uuid'] == 'local-2'
    assert result['local_decisions'][0]['learnable'] is True


async def _rows(rows):
    return rows
