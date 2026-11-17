"""Composition for procedure splitting."""

from kms.config.settings import Settings
from kms.inference.predictors import (
    PredictorFactory,
    build_token_budget,
    resolve_image_budget,
)
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.source.processing.procedure_splitter.examples import (
    procedure_split_router_examples,
    procedure_splitter_examples,
)
from kms.source.processing.procedure_splitter.inference import (
    ProcedureSplitRouterModule,
    ProcedureSplitRouterSignature,
    ProcedureSplitterModule,
    ProcedureSplitterSignature,
)
from kms.source.processing.procedure_splitter.node import ProcedureSplitterNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> ProcedureSplitterNode:
    stage = settings.source_processing.procedure_splitter
    router_examples = procedure_split_router_examples()
    splitter_examples = procedure_splitter_examples()
    router_budget = build_token_budget(
        settings,
        tokenizers,
        stage.router,
        input_token_budget=stage.router_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ProcedureSplitRouterSignature,
        examples=router_examples,
    )
    splitter_budget = build_token_budget(
        settings,
        tokenizers,
        stage.splitter,
        input_token_budget=stage.splitter_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=ProcedureSplitterSignature,
        examples=splitter_examples,
    )
    image_budget = resolve_image_budget(settings, stage.router, stage.splitter)
    router = ProcedureSplitRouterModule(
        predictors.create(
            ProcedureSplitRouterModule,
            stage.router,
            ProcedureSplitRouterSignature,
            examples=router_examples,
            budget=router_budget,
            image_budget=image_budget,
        )
    )
    splitter = ProcedureSplitterModule(
        predictors.create(
            ProcedureSplitterModule,
            stage.splitter,
            ProcedureSplitterSignature,
            examples=splitter_examples,
            budget=splitter_budget,
            image_budget=image_budget,
        )
    )
    return ProcedureSplitterNode(
        router,
        splitter,
        stage.context_window,
        batch_size=stage.batch_size,
        token_counters=tokenizers.counters(stage.router, stage.splitter),
        image_budget=image_budget,
        router_budget=router_budget,
        splitter_budget=splitter_budget,
    )
