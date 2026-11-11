"""Pause for the learner's goal or clarification answer."""

from langgraph.types import interrupt

from kms.learning.probe.goal.models import ClarificationQuestion
from kms.learning.probe.goal.state import GoalClarificationState
from kms.learning.session.models import SessionMessage


class GoalInputNode:
    """Present the current question and record its answer on resume."""

    def run(
        self, state: GoalClarificationState
    ) -> dict[str, list[SessionMessage]]:
        """Wait for an answer and append the completed exchange."""
        question = state.decision
        assert isinstance(question, ClarificationQuestion)
        answer = interrupt({'question': question.question})
        return {
            'messages': [
                *state.messages,
                SessionMessage(role='assistant', content=question.question),
                SessionMessage(role='user', content=answer),
            ]
        }
