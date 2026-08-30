"""Component-targeted event card designer."""

import dspy

from kms.core import llm, models, module
from kms.graph import learning
from kms.postprocessing.learning.cards import (
    CardDraft,
    CardVerification,
    CardWorkerInput,
    ComponentCardResult,
)


class EventCardSuitability(module.Module):
    """Decides whether one event has independently testable event knowledge.

    Accept the event's identity, participants, timing, conditions, mechanism,
    and explicitly stated significance. Reject instructions, questions,
    unsupported causal claims, and details belonging only to an outcome or
    related event. Preserve LaTeX exactly, including delimiters, commands,
    braces, superscripts, subscripts, fractions, alignment, and escaping;
    never convert it to Unicode or plain text.
    """

    record_name = 'event_card_suitability'

    class signature(dspy.Signature):
        r"""Decide whether the event supports a grounded recall card.

        Preserve LaTeX exactly, including $...$, $$...$$, \(...\), \[...\]
        delimiters, commands, braces, superscripts, subscripts, fractions,
        alignment, and escaping. Never convert it to Unicode or plain text.
        """
        hub_context: str = dspy.InputField()
        event: str = dspy.InputField()
        eligible: bool = dspy.OutputField()

    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        super().__init__(
            language_model or llm.module_lm(self.record_name), recorder=recorder
        )

    def encode(self, worker_input: CardWorkerInput) -> dict:
        return {
            'hub_context': worker_input.hub_context.model_dump_json(),
            'event': worker_input.target.content,
        }

    def decode(self, prediction, **inputs) -> bool:
        return module.require_bool(prediction.eligible, 'eligible')


class EventCardGenerator(module.Module):
    """Creates one concise recall card about the event itself.

    Ask what happened, who or what participated, when or under what conditions,
    or what mechanism the source explicitly states. Do not turn the card into
    an outcome, consequence, or broad historical summary.
    """

    record_name = 'event_card_generator'

    class signature(dspy.Signature):
        r"""Create a forward-recall card for the exact event.

        Preserve source wording and LaTeX exactly, including $...$, $$...$$,
        \(...\), \[...\] delimiters, commands, braces, superscripts,
        subscripts, fractions, alignment, and escaping. Never convert
        mathematical LaTeX to Unicode or plain text.
        """
        hub_context: str = dspy.InputField()
        event: str = dspy.InputField()
        card: CardDraft = dspy.OutputField()


    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        super().__init__(
            language_model or llm.module_lm(self.record_name), recorder=recorder
        )

    def encode(self, worker_input: CardWorkerInput) -> dict:
        return {
            'hub_context': worker_input.hub_context.model_dump_json(),
            'event': worker_input.target.content,
        }

    def decode(self, prediction, **inputs) -> CardDraft:
        return CardDraft.model_validate(prediction.card)


class EventCardVerifier(module.Module):
    """Verifies that the card preserves only the supplied event.

    Confirm that the response is supported by EVENT, tests one recallable
    statement, and does not conflate the event with its consequence, cause, or
    a related event.
    """

    record_name = 'event_card_verifier'

    class signature(dspy.Signature):
        r"""Reject unsupported event cards or cards with corrupted LaTeX.

        Verify exact LaTeX preservation, including delimiters, commands, braces,
        superscripts, subscripts, fractions, alignment, and escaping. Reject
        Unicode/plain-text conversion.
        """
        event: str = dspy.InputField()
        prompt: str = dspy.InputField()
        response: str = dspy.InputField()
        verification: CardVerification = dspy.OutputField()

    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        super().__init__(
            language_model or llm.module_lm(self.record_name), recorder=recorder
        )

    def encode(self, worker_input: CardWorkerInput, draft: CardDraft) -> dict:
        return {
            'event': worker_input.target.content,
            'prompt': draft.prompt,
            'response': draft.response,
        }

    def decode(self, prediction, **inputs) -> CardVerification:
        return CardVerification.model_validate(prediction.verification)


class EventCardDesigner:
    """Creates verified cards owned by a single Event component."""

    def __init__(
        self,
        suitability: EventCardSuitability,
        generator: EventCardGenerator,
        verifier: EventCardVerifier,
    ) -> None:
        self.suitability = suitability
        self.generator = generator
        self.verifier = verifier

    async def create_cards(
        self, worker_input: CardWorkerInput
    ) -> ComponentCardResult:
        if worker_input.target.kind != models.CardTargetKind.EVENT:
            raise ValueError('event designer received a non-event target')
        if not await self.suitability.aforward(worker_input=worker_input):
            return ComponentCardResult(
                target_uuid=worker_input.target.uuid,
                eligible=False,
                rejection_reason='not learnable',
            )
        draft = await self.generator.aforward(worker_input=worker_input)
        verification = await self.verifier.aforward(
            worker_input=worker_input, draft=draft
        )
        if not verification.supported:
            return ComponentCardResult(
                target_uuid=worker_input.target.uuid,
                eligible=True,
                rejection_reason='verification failed',
            )
        card = models.Card(
            uuid=learning.card_uuid(
                worker_input.target.uuid, content_key=draft.content_key
            ),
            target_uuid=worker_input.target.uuid,
            target_kind=models.CardTargetKind.EVENT,
            prompt=draft.prompt,
            response=draft.response,
        )
        return ComponentCardResult(
            target_uuid=worker_input.target.uuid,
            eligible=True,
            cards=(card,),
        )
