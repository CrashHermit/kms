"""Construction for instruction governance."""

from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.instruction_governance.examples import (
    instruction_governance_examples,
)
from kms.source.processing.instruction_governance.inference import (
    InstructionGovernanceModule,
    InstructionGovernanceSignature,
)
from kms.source.processing.instruction_governance.node import (
    InstructionGovernanceNode,
)


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> InstructionGovernanceNode:
    source = settings.source_processing
    stage = source.instruction_governance
    image_budget = ImagePayloadBudget(**stage.image_payload.model_dump())
    examples = instruction_governance_examples()
    budget = build_token_budget(
        settings,
        tokenizers,
        stage.inference,
        input_token_budget=stage.input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=InstructionGovernanceSignature,
        examples=examples,
    )
    module = InstructionGovernanceModule(
        predictors.create(
            InstructionGovernanceModule,
            stage.inference,
            InstructionGovernanceSignature,
            examples=examples,
            budget=budget,
            image_budget=image_budget,
        )
    )
    return InstructionGovernanceNode(
        module,
        stage,
        budget=budget,
        image_budget=image_budget,
        token_counters=tokenizers.counters(stage.inference),
    )
