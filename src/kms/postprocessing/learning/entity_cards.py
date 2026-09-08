"""Component-targeted entity card designer."""

import dspy

from kms.core import llm, models, module
from kms.graph import learning
from kms.postprocessing.learning.cards import (
    CardDraft,
    CardVerification,
    CardWorkerInput,
    ComponentCardResult,
)


class EntityCardSuitability(module.Module):
    """Decides whether one entity has durable, independently testable knowledge.

    Accept definitions, identity criteria, properties, classifications,
    measurements, formulas, stated relationships, and other information
    explicitly contained in the entity content or hub context. Reject
    instructions, questions, unsupported calculations, and facts belonging to
    a neighboring component. Preserve LaTeX exactly, including delimiters,
    commands, braces, superscripts, subscripts, fractions, alignment, and
    escaping; never convert it to Unicode or plain text.
    """

    class signature(dspy.Signature):
        r"""Decide whether the entity supports a grounded recall card.

        Preserve LaTeX exactly, including $...$, $$...$$, \(...\), \[...\]
        delimiters, commands, braces, superscripts, subscripts, fractions,
        alignment, and escaping. Never convert it to Unicode or plain text.
        """

        hub_context: str = dspy.InputField()
        entity: str = dspy.InputField()
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
            'entity': worker_input.target.content,
        }

    def decode(self, prediction, **inputs) -> bool:
        return module.require_bool(prediction.eligible, 'eligible')


class EntityCardGenerator(module.Module):
    """Creates one concise recall card about the entity itself.

    Ask about a specific property, definition, relation, or formula supplied in
    ENTITY. Do not ask the learner to identify the entity from the answer, and
    do not add background knowledge. Preserve notation, qualifiers, and units.
    """

    record_name = 'entity_card_generator'

    class signature(dspy.Signature):
        r"""Create a concise recall card about the supplied entity.

        Preserve only supplied knowledge and preserve LaTeX exactly: keep
        $...$, $$...$$, \(...\), and \[...\] delimiters, commands, braces,
        superscripts, subscripts, fractions, alignment, and escaping. Never
        convert mathematical LaTeX to Unicode or plain text.
        """

        hub_context: str = dspy.InputField()
        entity: str = dspy.InputField()
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
            'entity': worker_input.target.content,
        }

    def decode(self, prediction, **inputs) -> CardDraft:
        return CardDraft.model_validate(prediction.card)


class EntityCardVerifier(module.Module):
    """Verifies that the card tests only the supplied entity knowledge.

    Confirm that the response is entailed by ENTITY, contains no invented
    claims, answers one recall question, and preserves every relevant
    qualifier, formula, unit, and condition.
    """

    record_name = 'entity_card_verifier'

    class signature(dspy.Signature):
        r"""Reject cards with unsupported claims or corrupted LaTeX.

        Verify exact LaTeX preservation, including delimiters, commands,
        braces, superscripts, subscripts, fractions, alignment, and escaping.
        Reject any conversion to Unicode or plain text.
        """

        entity: str = dspy.InputField()
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
            'entity': worker_input.target.content,
            'prompt': draft.prompt,
            'response': draft.response,
        }

    def decode(self, prediction, **inputs) -> CardVerification:
        return CardVerification.model_validate(prediction.verification)


class EntityCardDesigner:
    """Creates verified cards owned by a single Entity component."""

    def __init__(
        self,
        suitability: EntityCardSuitability,
        generator: EntityCardGenerator,
        verifier: EntityCardVerifier,
    ) -> None:
        self.suitability = suitability
        self.generator = generator
        self.verifier = verifier

    async def create_cards(
        self, worker_input: CardWorkerInput
    ) -> ComponentCardResult:
        if worker_input.target.kind != models.CardTargetKind.ENTITY:
            raise ValueError('entity designer received a non-entity target')
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
            target_kind=models.CardTargetKind.ENTITY,
            prompt=draft.prompt,
            response=draft.response,
        )
        return ComponentCardResult(
            target_uuid=worker_input.target.uuid,
            eligible=True,
            cards=(card,),
        )
