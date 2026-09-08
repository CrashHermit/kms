"""Component-targeted procedure card designer."""

import dspy

from kms.core import llm, models, module
from kms.graph import learning
from kms.postprocessing.learning.cards import (
    CardDraft,
    CardVerification,
    CardWorkerInput,
    ComponentCardResult,
)


class ProcedureCardSuitability(module.Module):
    """Filters complete, source-backed procedures suitable for recall.

    Accept a method whose purpose, ordered steps, inputs, conditions, and
    outputs are explicit enough to teach. Reject incomplete procedures,
    ambiguous instructions, exercise-only noise, and generated solutions.
    Preserve LaTeX exactly, including delimiters, commands, braces,
    superscripts, subscripts, fractions, alignment, and escaping; never
    convert it to Unicode or plain text.
    """

    class signature(dspy.Signature):
        r"""Decide whether the procedure is complete and suitable for recall.

        Preserve LaTeX exactly, including $...$, $$...$$, \(...\), \[...\]
        delimiters, commands, braces, superscripts, subscripts, fractions,
        alignment, and escaping. Never convert it to Unicode or plain text.
        """

        hub_context: str = dspy.InputField()
        procedure: str = dspy.InputField()
        context: str = dspy.InputField()
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
            'procedure': worker_input.target.content,
            'context': str(worker_input.target.context),
        }

    def decode(self, prediction, **inputs) -> bool:
        return module.require_bool(prediction.eligible, 'eligible')


class ProcedureCardGenerator(module.Module):
    """Creates procedure cards that test the method, not just its result.

    Produce an overview, a required step, a condition, or a final result only
    when that variant is independently answerable from PROCEDURE. Preserve
    step order, notation, conditions, and all required intermediate work,
    including LaTeX exactly as supplied.
    """

    record_name = 'procedure_card_generator'

    class signature(dspy.Signature):
        r"""Create procedure recall cards while preserving the source exactly.

        Keep LaTeX delimiters, commands, braces, superscripts, subscripts,
        fractions, alignment, and escaping exactly as supplied. Never convert
        mathematical LaTeX to Unicode or plain text; keep display math separate
        from prose.
        """

        procedure: str = dspy.InputField()
        context: str = dspy.InputField()
        cards: list[CardDraft] = dspy.OutputField()

    def __init__(
        self, language_model: dspy.LM | None = None, recorder=None
    ) -> None:
        super().__init__(
            language_model or llm.module_lm(self.record_name), recorder=recorder
        )

    def encode(self, worker_input: CardWorkerInput) -> dict:
        return {
            'procedure': worker_input.target.content,
            'context': str(worker_input.target.context),
        }

    def decode(self, prediction, **inputs) -> list[CardDraft]:
        return [
            CardDraft.model_validate(card)
            for card in module.as_list(prediction.cards)
        ]


class ProcedureCardVerifier(module.Module):
    """Verifies each procedure variant against the complete source procedure.

    Confirm that the answer preserves required steps, order, conditions,
    intermediate expressions, and final outputs for the declared variant.
    Reject shortcuts that omit necessary reasoning or alter the method.
    """

    record_name = 'procedure_card_verifier'

    class signature(dspy.Signature):
        r"""Reject variants that omit required work or corrupt LaTeX.

        Verify exact preservation of delimiters, commands, braces, superscripts,
        subscripts, fractions, alignment, and escaping. Reject Unicode/plain-
        text conversions and malformed mathematical expressions.
        """

        procedure: str = dspy.InputField()
        content_key: str = dspy.InputField()
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
            'procedure': worker_input.target.content,
            'content_key': draft.content_key,
            'prompt': draft.prompt,
            'response': draft.response,
        }

    def decode(self, prediction, **inputs) -> CardVerification:
        return CardVerification.model_validate(prediction.verification)


class ProcedureCardDesigner:
    """Creates verified cards owned by a single source Procedure."""

    def __init__(
        self,
        suitability: ProcedureCardSuitability,
        generator: ProcedureCardGenerator,
        verifier: ProcedureCardVerifier,
    ) -> None:
        self.suitability = suitability
        self.generator = generator
        self.verifier = verifier

    async def create_cards(
        self, worker_input: CardWorkerInput
    ) -> ComponentCardResult:
        if worker_input.target.kind != models.CardTargetKind.PROCEDURE:
            raise ValueError(
                'procedure designer received a non-procedure target'
            )
        if (
            worker_input.target.context.get('kind')
            == models.ProcedureKind.GENERATED
        ):
            return ComponentCardResult(
                target_uuid=worker_input.target.uuid,
                eligible=False,
                rejection_reason='generated procedure',
            )
        if worker_input.target.context.get('kind') == 'generated':
            return ComponentCardResult(
                target_uuid=worker_input.target.uuid,
                eligible=False,
                rejection_reason='generated procedure',
            )
        if not await self.suitability.aforward(worker_input=worker_input):
            return ComponentCardResult(
                target_uuid=worker_input.target.uuid,
                eligible=False,
                rejection_reason='not learnable',
            )
        drafts = await self.generator.aforward(worker_input=worker_input)
        keys = [draft.content_key for draft in drafts]
        if len(keys) != len(set(keys)):
            raise ValueError(
                'procedure generator returned duplicate content keys'
            )
        cards: list[models.Card] = []
        for draft in drafts:
            verification = await self.verifier.aforward(
                worker_input=worker_input, draft=draft
            )
            if verification.supported:
                cards.append(
                    models.Card(
                        uuid=learning.card_uuid(
                            worker_input.target.uuid,
                            content_key=draft.content_key,
                        ),
                        target_uuid=worker_input.target.uuid,
                        target_kind=models.CardTargetKind.PROCEDURE,
                        prompt=draft.prompt,
                        response=draft.response,
                    )
                )
        return ComponentCardResult(
            target_uuid=worker_input.target.uuid,
            eligible=True,
            cards=tuple(cards),
            rejection_reason=None if cards else 'verification failed',
        )
