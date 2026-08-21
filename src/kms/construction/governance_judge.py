"""Binary judge: does an instruction govern a statement?"""

import dspy

from kms import config
from kms.core import content, module, recording


class GovernanceJudgeSignature(dspy.Signature):
    """
    TASK
    Decide whether this INSTRUCTION governs this STATEMENT.

    The statement is the complete semantic unit being evaluated. The context
    window contains the statement's nodes plus nearby document nodes that may
    clarify a lead-in, heading, qualifier, or section boundary.

    Return True if the instruction's directive applies to the statement.
    Return False if the statement is independent or belongs to another section.
    """

    instruction_directive: content.ContentParts = dspy.InputField(
        description="The instruction's text and figures, composed from its member nodes."
    )
    statement_content: content.ContentParts = dspy.InputField(
        description='The complete statement content to evaluate.'
    )
    context_window: content.ContentParts = dspy.InputField(
        description=(
            "The statement's marked node window, including nearby context."
        )
    )
    governs: bool = dspy.OutputField(
        description='True if the instruction governs this statement.'
    )
    confidence: float = dspy.OutputField(
        description='Confidence in the decision (0.0 to 1.0).'
    )


class GovernanceJudge(module.Module):
    """Binary judge: does instruction govern this statement?"""

    signature = GovernanceJudgeSignature
    record_name = 'governance_judge'

    def __init__(
        self,
        language_model: dspy.LM | None = None,
        recorder: recording.Recorder | None = None,
    ) -> None:
        super().__init__(
            language_model or config.get_settings().modules.governance_judge,
            recorder=recorder,
        )

    def encode(
        self,
        instruction_directive: content.Content,
        statement_content: content.Content,
        context_window: list,
    ) -> dict:
        """Builds the judge signature kwargs."""
        return {
            'instruction_directive': content.ContentParts(
                content=instruction_directive
            ),
            'statement_content': content.ContentParts(
                content=statement_content
            ),
            'context_window': content.labeled_content_parts(context_window),
        }

    def decode(self, prediction, **inputs) -> tuple[bool, float]:
        """Returns (governs, confidence)."""
        return prediction.governs, prediction.confidence


async def governs(
    judge: GovernanceJudge,
    instruction_directive: content.Content,
    statement_content: content.Content,
    context_window: list,
) -> tuple[bool, float]:
    """Convenience function to judge one instruction-statement pair."""
    return await judge.acall(
        instruction_directive=instruction_directive,
        statement_content=statement_content,
        context_window=context_window,
    )
