"""Hub-scoped, component-owned card creation."""

import operator
from collections.abc import Callable
from typing import Annotated, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from kms.core import models
from kms.graph import queries, writer


class CardHubContextInput(BaseModel):
    """The selected local hub plus its optional global curriculum context."""

    model_config = ConfigDict(extra='forbid')

    local_uuid: str = Field(min_length=1)
    local_name: str = ''
    local_description: str = ''
    global_uuid: str | None = None
    global_name: str | None = None
    global_description: str | None = None
    source: str = Field(min_length=1)


class CardTargetInput(BaseModel):
    """One source-local component selected from a local hub."""

    model_config = ConfigDict(extra='forbid')

    uuid: str = Field(min_length=1)
    kind: models.CardTargetKind
    source: str = Field(min_length=1)
    content: str = Field(min_length=1)
    context: dict = Field(default_factory=dict)


class CardWorkerInput(BaseModel):
    """Immutable input for one isolated component card worker."""

    model_config = ConfigDict(extra='forbid')

    hub_context: CardHubContextInput
    target: CardTargetInput


class CardDraft(BaseModel):
    """One card wording proposed for a component variant."""

    model_config = ConfigDict(extra='forbid')

    content_key: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)


class CardVerification(BaseModel):
    """Explicit verification decision for a generated card."""

    model_config = ConfigDict(extra='forbid')

    supported: bool


class ComponentCardResult(BaseModel):
    """The terminal outcome for one component worker."""

    model_config = ConfigDict(extra='forbid', arbitrary_types_allowed=True)

    target_uuid: str = Field(min_length=1)
    eligible: bool
    cards: tuple[models.Card, ...] = ()
    rejection_reason: str | None = None


class ComponentCardDesigner(Protocol):
    """Creates verified cards for one immutable component target."""

    async def create_cards(
        self, worker_input: CardWorkerInput
    ) -> ComponentCardResult: ...


class CardCreationState(TypedDict, total=False):
    """State for the independent hub card creation graph."""

    hub_context: CardHubContextInput
    targets: list[CardTargetInput]
    worker_input: CardWorkerInput
    worker_results: Annotated[list[ComponentCardResult], operator.add]
    report: dict[str, int | str]


class HubCardExecutorNode:
    """Dispatches one worker for every source-local component target."""

    def __init__(self, executor: 'HubCardExecutor') -> None:
        self.executor = executor

    def dispatch(self, state: CardCreationState) -> list[Send] | str:
        """Send each target to an isolated worker or continue to collection."""
        hub_context = state['hub_context']
        sends = [
            Send(
                'card_worker',
                {
                    'worker_input': CardWorkerInput(
                        hub_context=hub_context,
                        target=target,
                    )
                },
            )
            for target in state.get('targets', [])
        ]
        return sends or 'card_collect'

    async def worker(self, state: CardCreationState) -> CardCreationState:
        """Run one component's complete designer pipeline."""
        worker_input = state['worker_input']
        try:
            result = await self.executor.designer.create_cards(worker_input)
        except ValueError as error:
            raise ValueError(
                f'card creation failed for target={worker_input.target.uuid}: {error}'
            ) from error
        if result.target_uuid != worker_input.target.uuid:
            raise ValueError(
                f'card creation returned a different target: '
                f'{result.target_uuid} != {worker_input.target.uuid}'
            )
        return {'worker_results': [result]}

    async def collect(self, state: CardCreationState) -> CardCreationState:
        """Deduplicate verified cards and persist component ownership once."""
        results = state.get('worker_results', [])
        cards_by_uuid = {
            card.uuid: card for result in results for card in result.cards
        }
        cards = list(cards_by_uuid.values())
        await writer.persist_cards(
            cards, session_factory=self.executor.session_factory
        )
        await writer.persist_card_target_edges(
            cards, session_factory=self.executor.session_factory
        )
        eligible = sum(result.eligible for result in results)
        generated = sum(len(result.cards) for result in results)
        verified = len(cards)
        return {
            'report': {
                'hub_uuid': state['hub_context'].local_uuid,
                'target_kind': self.executor.target_kind.value,
                'targets': len(state.get('targets', [])),
                'eligible': eligible,
                'generated': generated,
                'verified': verified,
                'rejected': len(results) - eligible,
                'persisted': len(cards),
            }
        }


class HubCardExecutor:
    """Runs component-targeted card creation for one selected local hub."""

    def __init__(
        self,
        *,
        session_factory: Callable,
        target_kind: models.CardTargetKind,
        designer: ComponentCardDesigner,
    ) -> None:
        self.session_factory = session_factory
        self.target_kind = target_kind
        self.designer = designer
        self.node = HubCardExecutorNode(self)
        graph = StateGraph(CardCreationState)
        graph.add_node('card_worker', self.node.worker)
        graph.add_node('card_collect', self.node.collect)
        graph.add_edge(START, 'card_dispatch')
        graph.add_node('card_dispatch', lambda state: state)
        graph.add_conditional_edges('card_dispatch', self.node.dispatch)
        graph.add_edge('card_worker', 'card_collect')
        graph.add_edge('card_collect', END)
        self.graph = graph.compile()

    async def create(self, *, hub_uuid: str) -> dict[str, int | str]:
        """Load one hub's targets and run the compiled worker graph."""
        hub_context = CardHubContextInput.model_validate(
            await queries.card_hub_context(
                self.session_factory,
                target_kind=self.target_kind,
                hub_uuid=hub_uuid,
            )
        )
        targets = [
            CardTargetInput.model_validate(row)
            for row in await queries.card_targets_for_hub(
                self.session_factory,
                target_kind=self.target_kind,
                hub_uuid=hub_uuid,
            )
        ]
        result = await self.graph.ainvoke(
            {
                'hub_context': hub_context,
                'targets': targets,
                'worker_results': [],
            }
        )
        return result['report']


async def create_cards_for_hub(
    *,
    session_factory: Callable,
    target_kind: models.CardTargetKind,
    hub_uuid: str,
    designer: ComponentCardDesigner,
) -> dict[str, int | str]:
    """Creates component-owned cards for all targets in one selected hub."""
    executor = HubCardExecutor(
        session_factory=session_factory,
        target_kind=target_kind,
        designer=designer,
    )
    return await executor.create(hub_uuid=hub_uuid)
