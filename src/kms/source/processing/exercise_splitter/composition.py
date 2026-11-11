from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.media.budget import ImagePayloadBudget
from kms.source.processing.exercise_splitter.examples import (
    exercise_splitter_examples,
    exercise_strip_router_examples,
)
from kms.source.processing.exercise_splitter.inference import (
    ExerciseSplitterModule,
    ExerciseSplitterSignature,
    ExerciseStripRouterModule,
    ExerciseStripRouterSignature,
)
from kms.source.processing.exercise_splitter.node import ExerciseSplitterNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ExerciseSplitterNode:
    source = settings.source_processing.exercise_splitter
    router_examples = exercise_strip_router_examples()
    splitter_examples = exercise_splitter_examples()
    router_budget = build_token_budget(
        settings,
        tokenizers,
        source.router,
        input_token_budget=source.router_input_token_budget,
        safety_margin_tokens=source.safety_margin_tokens,
        signature=ExerciseStripRouterSignature,
        examples=router_examples,
    )
    splitter_budget = build_token_budget(
        settings,
        tokenizers,
        source.splitter,
        input_token_budget=source.splitter_input_token_budget,
        safety_margin_tokens=source.safety_margin_tokens,
        signature=ExerciseSplitterSignature,
        examples=splitter_examples,
    )
    image_budget = ImagePayloadBudget(**source.image_payload.model_dump())
    router = ExerciseStripRouterModule(
        predictors.create(
            ExerciseStripRouterModule,
            source.router,
            ExerciseStripRouterSignature,
            examples=router_examples,
            budget=router_budget,
            image_budget=image_budget,
        )
    )
    splitter = ExerciseSplitterModule(
        predictors.create(
            ExerciseSplitterModule,
            source.splitter,
            ExerciseSplitterSignature,
            examples=splitter_examples,
            budget=splitter_budget,
            image_budget=image_budget,
        )
    )
    return ExerciseSplitterNode(
        router,
        splitter,
        source.context_window,
        batch_size=source.batch_size,
        token_counters=tokenizers.counters(source.router, source.splitter),
        image_budget=image_budget,
        router_budget=router_budget,
        splitter_budget=splitter_budget,
    )
