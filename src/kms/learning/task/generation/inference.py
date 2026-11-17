"""Generate one grounded open-response task."""

import dspy

from kms.learning.task.generation.models import TaskDraft, TaskGenerationRequest


class TaskGenerationSignature(dspy.Signature):
    """Create one learner-facing free-response task and hidden solution.

    Assess only the single target selected in the request: a proposition's
    meaning/truth conditions, an argument's premise-to-conclusion justification,
    a specified task solution, or the newly generated task's own solution.
    Supporting knowledge may inform the question but does not add independently
    graded objectives. For an existing solution target, preserve the supplied
    original task's problem conditions and reference answer while asking for
    that specific solution or an explanation of it.

    Ask one clear question that elicits the learner's own response; do not use
    multiple-choice options, trick wording, unnecessary reading difficulty, or
    solving instructions copied from evidence. Ask for optional work or
    explanation only when useful, and do not require explanation when the
    answer alone can establish the target. Never reveal the answer or solution
    in the task prompt. Provide a complete reference solution and criteria in
    this same response before any learner response exists. Criteria must accept
    alternative valid methods and must not automatically fail an answer merely
    for omitting unnecessary explanation. Reference only supplied images by
    their exact native-catalog image_position; never invent image paths,
    payloads, citations, or facts. Grounding must be a subset of supplied
    proposition, argument, and evidence-unit UUIDs. Source-unit UUIDs may
    ground but never become assessment targets. The supplied inputs have
    sufficient grounding; do not search for sources or produce a fallback.
    """

    request: TaskGenerationRequest = dspy.InputField()
    content_images: list[dspy.Image] = dspy.InputField()
    task: TaskDraft = dspy.OutputField()


class TaskGenerationModule(dspy.Module):
    """Run task generation through an injected predictor."""

    def __init__(self, predictor: dspy.Module) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, *, request: TaskGenerationRequest) -> TaskDraft:
        return self.predictor(request=request).task

    async def aforward(self, *, request: TaskGenerationRequest) -> TaskDraft:
        return (await self.predictor.acall(request=request)).task
