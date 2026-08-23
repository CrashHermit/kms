"""Create atomic learning cards from one EntityHub description."""

from __future__ import annotations

import dspy
from pydantic import BaseModel, Field

from kms.core import llm, models, module
from kms.graph import learning


class EntityCardDraft(BaseModel):
    """One generated card before it receives graph identity and FSRS state."""

    prompt: str = Field(min_length=1)
    response: str = Field(min_length=1)


class EntityCardRouterSignature(dspy.Signature):
    r"""
    Decide whether an entity description contains useful, testable knowledge.

    Return True when the description states at least one standalone fact about
    the named entity that a learner could retrieve as a short question and
    answer. Return False for empty, vague, purely referential, navigational,
    or non-informational descriptions. Do not create facts or cards.

    Return only True or False.
    """

    canonical_name: str = dspy.InputField(
        description='The canonical name of the entity.'
    )
    description: str = dspy.InputField(
        description='The canonical learner-facing entity description.'
    )
    has_testable_knowledge: bool = dspy.OutputField(
        description='Whether the description contains testable knowledge.'
    )


class EntityCardRouter(module.Module):
    """Gates entity card creation when no testable knowledge is present."""

    signature = EntityCardRouterSignature
    record_name = 'entity_card_router'

    def __init__(
        self,
        language_model: dspy.LM | None = None,
        recorder=None,
    ) -> None:
        super().__init__(
            language_model or llm.module_lm('entity_card_router'),
            recorder=recorder,
        )

    def encode(self, canonical_name: str, description: str) -> dict:
        return {
            'canonical_name': canonical_name,
            'description': description,
        }

    def decode(self, prediction, **inputs) -> bool:
        """Returns the validated knowledge-routing decision."""
        return module.require_bool(
            prediction.has_testable_knowledge, 'has_testable_knowledge'
        )


class EntityFactGeneratorSignature(dspy.Signature):
    r"""
    Decompose the entity description into standalone atomic facts.

    Emit the smallest useful units of knowledge. Each fact must express one
    assertion about the canonical entity, be a concise complete sentence, and
    contain every condition or qualifier needed to stand alone. Split
    independent claims joined by "and" or a list into separate facts. Do not
    emit fragments, card questions, explanations, aliases, metadata, or facts
    that are not supported by the description. Do not add examples or general
    knowledge unless the description states them.

    PRESERVE LATEX FORMATTING EXACTLY:
    - All mathematical notation must remain in its original LaTeX form
    - Inline math delimiters: $...$ must stay $...$
    - Display math delimiters: $$...$$ must stay $$...$$
    - Do NOT convert to Unicode (e.g., keep \\alpha not α, keep \\leq not ≤)
    - Do NOT strip or change any LaTeX delimiters
    - Do NOT reformat mathematical expressions
    Return an empty list when no standalone testable fact can be extracted.
    """

    canonical_name: str = dspy.InputField(
        description='The canonical name of the entity.'
    )
    description: str = dspy.InputField(
        description='The canonical entity description to decompose.'
    )
    facts: list[str] = dspy.OutputField(
        description='Concise standalone atomic facts with preserved LaTeX, or an empty list.'
    )


class EntityFactGenerator(module.Module):
    """Splits one entity description into atomic standalone fact strings."""

    signature = EntityFactGeneratorSignature
    record_name = 'entity_fact_generator'

    def __init__(
        self,
        language_model: dspy.LM | None = None,
        recorder=None,
    ) -> None:
        super().__init__(
            language_model or llm.module_lm('entity_fact_generator'),
            recorder=recorder,
        )

    def encode(self, canonical_name: str, description: str) -> dict:
        return {
            'canonical_name': canonical_name,
            'description': description,
        }

    def decode(self, prediction, **inputs) -> list[str]:
        """Returns non-empty fact strings without repairing model output."""
        facts = module.as_list(prediction.facts)
        for index, fact in enumerate(facts):
            if not isinstance(fact, str) or not fact.strip():
                raise ValueError(
                    f'facts[{index}] must be a non-empty string'
                )
        return facts


class EntityCardGeneratorSignature(dspy.Signature):
    r"""
    Turn one atomic entity fact into one small flash card.

    The card must ask one concise, fact-specific question. Its answer
    must be the supplied FACT exactly. Do not shorten it, paraphrase it,
    answer it with a fragment, or remove its subject, conditions, qualifiers,
    or punctuation. Do not use a generic repeated question such as
    "What is the entity?".

    REQUIRED LATEX PRESERVATION:
    - The response field MUST copy the input fact character-for-character
    - All $...$ inline math delimiters must remain unchanged
    - All $$...$$ display math delimiters must remain unchanged
    - All LaTeX commands (\\alpha, \\leq, \\sum, etc.) must remain unchanged
    - No Unicode substitution for mathematical symbols
    - No delimiter stripping or reformatting

    Required output shape for FACT = "A group has an identity element." and
    CANONICAL NAME = "Group":

    card = {
      "prompt": "What element does a group have?",
      "response": "A group has an identity element."
    }

    Do not add facts, explanations, examples, aliases, tags, or priorities.
    Keep questions short and atomic. Preserve mathematical meaning and LaTeX.
    """

    canonical_name: str = dspy.InputField(
        description='The canonical name of the entity.'
    )
    fact: str = dspy.InputField(
        description='One concise standalone atomic fact about the entity.'
    )
    card: EntityCardDraft = dspy.OutputField(
        description=(
            'The prompt is a concise fact-specific question. The response '
            'must exactly copy the input fact including all LaTeX delimiters '
            'and mathematical notation.'
        )
    )


class EntityCardGenerator(module.Module):
    """Creates one card for one atomic entity fact."""

    signature = EntityCardGeneratorSignature
    record_name = 'entity_card_generator'

    def __init__(
        self,
        language_model: dspy.LM | None = None,
        recorder=None,
    ) -> None:
        super().__init__(
            language_model or llm.module_lm('entity_card_generator'),
            recorder=recorder,
        )

    def encode(self, canonical_name: str, fact: str) -> dict:
        return {'canonical_name': canonical_name, 'fact': fact}

    def decode(self, prediction, **inputs) -> list[EntityCardDraft]:
        card = EntityCardDraft.model_validate(prediction.card)
        return [card]


async def create_entity_cards(
    hub_uuid: str,
    canonical_name: str,
    description: str,
    *,
    router: EntityCardRouter | None = None,
    fact_generator: EntityFactGenerator | None = None,
    card_generator: EntityCardGenerator | None = None,
    language_model: dspy.LM | None = None,
    recorder=None,
) -> list[models.Card]:
    """Create graph-ready cards for one EntityHub description.

    ``hub_uuid`` is persistence context and is not sent to any LLM module.
    The router gates fact extraction; each extracted fact is then processed by
    the card generator independently.
    """
    if not canonical_name.strip() or not description.strip():
        return []

    router = router or EntityCardRouter(language_model, recorder=recorder)
    if not await router.aforward(
        canonical_name=canonical_name, description=description
    ):
        return []

    fact_generator = fact_generator or EntityFactGenerator(
        language_model, recorder=recorder
    )
    card_generator = card_generator or EntityCardGenerator(
        language_model, recorder=recorder
    )
    facts = await fact_generator.aforward(
        canonical_name=canonical_name, description=description
    )

    cards: list[models.Card] = []
    for fact in facts:
        drafts = await card_generator.aforward(
            canonical_name=canonical_name, fact=fact
        )
        for draft in drafts:
            cards.append(
                models.Card(
                    uuid=learning.card_uuid(hub_uuid, content_key=fact),
                    hub_uuid=hub_uuid,
                    hub_kind='entity',
                    prompt=draft.prompt,
                    response=draft.response,
                )
            )
    return cards
