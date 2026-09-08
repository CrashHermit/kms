"""Component-targeted triplet card designer."""

import dspy

from kms.core import llm, models, module
from kms.graph import learning
from kms.postprocessing.learning.cards import (
    CardDraft,
    CardVerification,
    CardWorkerInput,
    ComponentCardResult,
)


class TripletCardSuitability(module.Module):
    """Filters exact, concrete, independently learnable assertions.

    Accept definitions, measurements, quantities, formulas, named objects,
    possession, membership, and explicit mathematical or causal relations when
    the subject, predicate, and object are sufficiently clear. Reject
    instructions, questions, unsupported implications, graph-infrastructure
    predicates, duplicate/converse restatements, and vague relations.
    """

    record_name = 'triplet_card_suitability'

    class signature(dspy.Signature):
        r"""Accept only concrete, self-contained, learnable relations.

        Reject graph-infrastructure predicates, duplicate/converse restatements,
        uninformative relations, and targets with ambiguous endpoints. Preserve
        LaTeX exactly, including $...$, $$...$$, \(...\), \[...\] delimiters,
        commands, braces, superscripts, subscripts, fractions, alignment, and
        escaping. Never convert it to Unicode or plain text.
        """

        hub_context: str = dspy.InputField()
        triplet: str = dspy.InputField()
        endpoints: str = dspy.InputField()
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
            'triplet': worker_input.target.content,
            'endpoints': str(worker_input.target.context),
        }

    def decode(self, prediction, **inputs) -> bool:
        return module.require_bool(prediction.eligible, 'eligible')


class TripletCardGenerator(module.Module):
    """Creates a forward-recall card for the exact persisted assertion.

    Ask for the relation or property of the named subject. Do not ask the
    learner to identify an endpoint from the answer, infer an unstated fact,
    reverse the relation, or change negation. Preserve notation and qualifiers.
    """

    record_name = 'triplet_card_generator'

    class signature(dspy.Signature):
        r"""Create a forward-recall card preserving the exact relation.

        Do not invert predicate polarity or negation or infer an endpoint.
        Preserve LaTeX exactly, including $...$, $$...$$, \(...\), \[...\]
        delimiters, commands, braces, superscripts, subscripts, fractions,
        alignment, and escaping. Never convert it to Unicode or plain text.
        """

        hub_context: str = dspy.InputField()
        triplet: str = dspy.InputField()
        endpoints: str = dspy.InputField()
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
            'triplet': worker_input.target.content,
            'endpoints': str(worker_input.target.context),
        }

    def decode(self, prediction, **inputs) -> CardDraft:
        return CardDraft.model_validate(prediction.card)


class TripletCardVerifier(module.Module):
    """Verifies exact subject, predicate polarity, object, and context.

    Reject any answer that reverses the relation, drops or introduces negation,
    broadens a measurement or formula, changes an endpoint, or adds outside
    knowledge.
    """

    record_name = 'triplet_card_verifier'

    class signature(dspy.Signature):
        r"""Reject unsupported answers or any corrupted LaTeX.

        Verify exact preservation of delimiters, commands, braces, superscripts,
        subscripts, fractions, alignment, and escaping. Reject Unicode/plain-
        text conversion and malformed mathematical expressions.
        """

        triplet: str = dspy.InputField()
        endpoints: str = dspy.InputField()
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
            'triplet': worker_input.target.content,
            'endpoints': str(worker_input.target.context),
            'prompt': draft.prompt,
            'response': draft.response,
        }

    def decode(self, prediction, **inputs) -> CardVerification:
        return CardVerification.model_validate(prediction.verification)


class TripletCardDesigner:
    """Creates verified cards owned by a single Triplet occurrence."""

    def __init__(
        self,
        suitability: TripletCardSuitability,
        generator: TripletCardGenerator,
        verifier: TripletCardVerifier,
    ) -> None:
        self.suitability = suitability
        self.generator = generator
        self.verifier = verifier

    async def create_cards(
        self, worker_input: CardWorkerInput
    ) -> ComponentCardResult:
        if worker_input.target.kind != models.CardTargetKind.TRIPLET:
            raise ValueError('triplet designer received a non-triplet target')
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
            target_kind=models.CardTargetKind.TRIPLET,
            prompt=draft.prompt,
            response=draft.response,
        )
        return ComponentCardResult(
            target_uuid=worker_input.target.uuid,
            eligible=True,
            cards=(card,),
        )
