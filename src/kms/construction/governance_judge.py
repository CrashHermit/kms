"""Binary judge: does an instruction govern a statement?"""

import dspy

from kms import config
from kms.core import context_window, models, module, recording


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

    instruction_nodes: list[models.NodeInput] = dspy.InputField(
        description=(
            'Ordered instruction text records. Image descriptions appear as '
            'text; no image assets or bytes are included.'
        )
    )
    context_before: list[models.NodeInput] = dspy.InputField(
        description='Ordered context records before statement_nodes; reference only.',
    )
    statement_nodes: list[models.NodeInput] = dspy.InputField(
        description='Ordered statement records being governed.',
    )
    context_after: list[models.NodeInput] = dspy.InputField(
        description='Ordered context records after statement_nodes; reference only.',
    )
    governs: bool = dspy.OutputField(
        description='True if the instruction governs the statement.'
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
        instruction_nodes: list[context_window.ContextNode],
        context_before: list[context_window.ContextNode],
        statement_nodes: list[context_window.ContextNode],
        context_after: list[context_window.ContextNode],
    ) -> dict[str, object]:
        """Projects modality-neutral nodes into structured text records."""
        return {
            'instruction_nodes': _governance_inputs(instruction_nodes),
            'context_before': _governance_inputs(context_before),
            'statement_nodes': _governance_inputs(statement_nodes),
            'context_after': _governance_inputs(context_after),
        }

    def decode(self, prediction, **inputs) -> bool:
        return module.require_bool(prediction.governs, 'governs')


def _governance_inputs(
    nodes: list[context_window.ContextNode],
) -> list[models.NodeInput]:
    """Projects context nodes without exposing assets or source identity."""
    return [
        context_window.node_input(node, index)
        for index, node in enumerate(nodes)
    ]


async def governs(
    judge: GovernanceJudge,
    instruction_nodes: list[context_window.ContextNode],
    context_before: list[context_window.ContextNode],
    statement_nodes: list[context_window.ContextNode],
    context_after: list[context_window.ContextNode],
) -> bool:
    """Convenience function to judge one instruction-statement pair."""
    return await judge.acall(
        instruction_nodes=instruction_nodes,
        context_before=context_before,
        statement_nodes=statement_nodes,
        context_after=context_after,
    )
