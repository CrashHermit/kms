import asyncio

import dspy
from dspy.utils import DummyLM
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from kms.learning.probe.goal.graph import build_goal_graph
from kms.learning.probe.goal.inference import (
    GoalClarificationModule,
    GoalClarificationSignature,
)
from kms.learning.probe.goal.input import GoalInputNode
from kms.learning.probe.goal.models import (
    ClarificationQuestion,
    ClarifiedGoal,
    GoalClarificationInput,
)
from kms.learning.probe.goal.node import GoalClarificationNode
from kms.learning.probe.goal.state import GoalClarificationState
from kms.learning.session.models import SessionMessage


class _Predictor(dspy.Module):
    def __init__(self, decisions):
        super().__init__()
        self.decisions = iter(decisions)
        self.requests = []

    async def aforward(self, *, request):
        self.requests.append(request)
        return dspy.Prediction(decision=next(self.decisions))


def test_parent_resumes_clarification_subgraph():
    async def run():
        predictor = _Predictor(
            [
                ClarificationQuestion(
                    question='What would you like to use calculus for?'
                ),
                ClarifiedGoal(
                    goal='Understand derivatives and instantaneous velocity.'
                ),
            ]
        )
        child = build_goal_graph(
            GoalInputNode(),
            GoalClarificationNode(GoalClarificationModule(predictor)),
        )
        graph = StateGraph(GoalClarificationState)
        graph.add_node('clarify_goal', child)
        graph.add_edge(START, 'clarify_goal')
        graph.add_edge('clarify_goal', END)
        serde = JsonPlusSerializer(
            allowed_msgpack_modules=[
                ('kms.learning.session.models', 'SessionMessage'),
                ('kms.learning.probe.goal.models', 'ClarificationQuestion'),
                ('kms.learning.probe.goal.models', 'ClarifiedGoal'),
            ]
        )
        parent = graph.compile(checkpointer=InMemorySaver(serde=serde))
        config = {'configurable': {'thread_id': 'session'}}
        state = GoalClarificationState(user_uuid='u', session_uuid='s')
        first = await parent.ainvoke(state.model_dump(), config)
        first_question = first['__interrupt__'][0].value['question']
        assert predictor.requests == []
        second = await parent.ainvoke(Command(resume='Calculus'), config)
        question = second['__interrupt__'][0].value['question']
        assert question == 'What would you like to use calculus for?'
        result = await parent.ainvoke(
            Command(resume='Understanding motion'), config
        )
        state = GoalClarificationState.model_validate(result)
        assert (
            state.decision.goal
            == 'Understand derivatives and instantaneous velocity.'
        )
        assert [
            (message.role, message.content) for message in state.messages
        ] == [
            ('assistant', first_question),
            ('user', 'Calculus'),
            ('assistant', question),
            ('user', 'Understanding motion'),
        ]
        assert len(predictor.requests) == 2
        assert predictor.requests[-1].messages == state.messages

    asyncio.run(run())


def test_dspy_structured_decisions():
    async def run():
        predictor = dspy.Predict(GoalClarificationSignature)
        predictor.set_lm(
            DummyLM(
                [
                    {
                        'decision': {
                            'kind': 'clarify',
                            'question': 'Which topic?',
                        }
                    },
                    {
                        'decision': {
                            'kind': 'ready',
                            'goal': 'Learn derivatives',
                        }
                    },
                ]
            )
        )
        module = GoalClarificationModule(predictor)
        request = GoalClarificationInput(
            messages=[SessionMessage(role='user', content='Calculus')]
        )
        assert (await module.acall(request=request)).question == 'Which topic?'
        assert (await module.acall(request=request)).goal == 'Learn derivatives'

    asyncio.run(run())
