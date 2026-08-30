import asyncio

from kms.core import models
from kms.graph import learning
from kms.postprocessing.learning import cards


class _Designer:
    def __init__(self):
        self.inputs = []

    async def create_cards(self, worker_input):
        self.inputs.append(worker_input)
        card = models.Card(
            uuid=learning.card_uuid(
                worker_input.target.uuid, content_key='default'
            ),
            target_uuid=worker_input.target.uuid,
            target_kind=worker_input.target.kind,
        )
        return cards.ComponentCardResult(
            target_uuid=worker_input.target.uuid,
            eligible=True,
            cards=(card,),
        )


def _input(uuid='target-1'):
    return cards.CardWorkerInput(
        hub_context=cards.CardHubContextInput(
            local_uuid='hub', source='source', local_description='topic'
        ),
        target=cards.CardTargetInput(
            uuid=uuid,
            kind=models.CardTargetKind.ENTITY,
            source='source',
            content='Thing: description',
        ),
    )


def test_dispatch_sends_one_isolated_worker_per_target():
    executor = cards.HubCardExecutor(
        session_factory=object(),
        target_kind=models.CardTargetKind.ENTITY,
        designer=_Designer(),
    )
    state = {
        'hub_context': _input().hub_context,
        'targets': [_input('one').target, _input('two').target],
    }
    sends = executor.node.dispatch(state)
    assert len(sends) == 2
    assert (
        sends[0].arg['worker_input'].hub_context
        == sends[1].arg['worker_input'].hub_context
    )
    assert sends[0].arg['worker_input'].target.uuid == 'one'
    assert sends[1].arg['worker_input'].target.uuid == 'two'


def test_worker_returns_component_result():
    designer = _Designer()
    executor = cards.HubCardExecutor(
        session_factory=object(),
        target_kind=models.CardTargetKind.ENTITY,
        designer=designer,
    )
    result = asyncio.run(executor.node.worker({'worker_input': _input()}))
    assert result['worker_results'][0].target_uuid == 'target-1'
    assert designer.inputs[0].target.uuid == 'target-1'
