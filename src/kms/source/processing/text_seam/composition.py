from kms.config.settings import Settings
from kms.inference.predictors import PredictorFactory, build_token_budget
from kms.inference.tokenization.resources import LanguageModelTokenizers
from kms.source.processing.text_seam.examples import (
    text_seam_judge_examples,
    text_seam_rewrite_examples,
)
from kms.source.processing.text_seam.inference import (
    TextSeamJudgeModule,
    TextSeamRewriterModule,
    TextSeamRewriteSignature,
    TextSeamSignature,
)
from kms.source.processing.text_seam.node import TextSeamNode


def build_node(
    settings: Settings,
    *,
    predictors: PredictorFactory,
    tokenizers: LanguageModelTokenizers,
) -> TextSeamNode:
    source = settings.source_processing
    stage = source.text_seam
    judge_examples = text_seam_judge_examples()
    rewrite_examples = text_seam_rewrite_examples()
    judge_budget = build_token_budget(
        settings,
        tokenizers,
        stage.judge,
        input_token_budget=stage.judge_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=TextSeamSignature,
        examples=judge_examples,
    )
    rewrite_budget = build_token_budget(
        settings,
        tokenizers,
        stage.rewriter,
        input_token_budget=stage.rewrite_input_token_budget,
        safety_margin_tokens=stage.safety_margin_tokens,
        signature=TextSeamRewriteSignature,
        examples=rewrite_examples,
    )
    judge = TextSeamJudgeModule(
        predictors.create(
            TextSeamJudgeModule,
            stage.judge,
            TextSeamSignature,
            examples=judge_examples,
            budget=judge_budget,
        )
    )
    rewriter = TextSeamRewriterModule(
        predictors.create(
            TextSeamRewriterModule,
            stage.rewriter,
            TextSeamRewriteSignature,
            examples=rewrite_examples,
            budget=rewrite_budget,
        )
    )
    return TextSeamNode(
        judge,
        rewriter,
        judge_budget=judge_budget,
        rewrite_budget=rewrite_budget,
        judge_batch_size=stage.judge_batch_size,
        rewrite_batch_size=stage.rewrite_batch_size,
    )
