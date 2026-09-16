import dspy

from kms2 import composition
from kms2.config import (
    ContextWindowSettings,
    ExerciseFinderSettings,
    ImageDescriptionSettings,
    InstructionFinderSettings,
    InstructionGovernanceSettings,
    OCRSettings,
    PedagogicalFinderSettings,
    PredictorStrategy,
    Settings,
    SourceSettings,
    SplitterSettings,
    StageInferenceSettings,
    StatementProcedureSettings,
    TextSeamSettings,
)
from kms2.database.client import DatabaseClient
from kms2.langgraph.semantic.graph import SemanticGraph
from kms2.langgraph.source.graph import SourceGraph
from kms2.module.semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
)
from kms2.module.semantic.source_event_hub_judge import (
    SourceEventHubJudgeModule,
)
from kms2.module.semantic.source_predicate_hub_judge import (
    SourcePredicateHubJudgeModule,
)
from kms2.node.source.content_correction import ContentCorrectionNode
from kms2.node.source.embedding import EmbeddingNode
from kms2.node.source.exercise_finder import ExerciseFinderNode
from kms2.node.source.formatting import FormattingNode
from kms2.node.source.image_description import ImageDescriptionNode
from kms2.node.source.image_seam import ImageSeamNode
from kms2.node.source.instruction_finder import InstructionFinderNode
from kms2.node.source.instruction_governance import InstructionGovernanceNode
from kms2.node.source.ocr import OCRNode
from kms2.node.source.pedagogical_finder import PedagogicalFinderNode
from kms2.node.source.persistence import SourcePersistenceNode
from kms2.node.source.splitter import SplitterNode
from kms2.node.source.statement_procedure import StatementProcedureNode
from kms2.node.source.text_seam import TextSeamNode
from kms2.ocr.mistral import MistralOCRProvider


class _RecordingRuntime:
    embedding = object()
    reranker = object()
    calls: list[tuple[str, PredictorStrategy, type[dspy.Signature]]]

    def predictor(
        self,
        inference: StageInferenceSettings,
        signature: type[dspy.Signature],
    ) -> dspy.Module:
        self.calls.append(
            (
                inference.model_server_profile,
                inference.strategy,
                signature,
            )
        )
        if inference.strategy is PredictorStrategy.PREDICT:
            return dspy.Predict(signature)
        return dspy.ChainOfThought(signature)


def _stage_inference(
    model_server_profile: str,
    strategy: PredictorStrategy,
) -> StageInferenceSettings:
    return StageInferenceSettings(
        model_server_profile=model_server_profile,
        strategy=strategy,
    )


def _settings() -> Settings:
    return Settings(
        source=SourceSettings(
            content_correction=_stage_inference(
                'content-correction', PredictorStrategy.PREDICT
            ),
            formatting=_stage_inference(
                'formatting', PredictorStrategy.CHAIN_OF_THOUGHT
            ),
            text_seam=TextSeamSettings(
                judge=_stage_inference('text-judge', PredictorStrategy.PREDICT),
                rewriter=_stage_inference(
                    'text-rewriter', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            image_seam=_stage_inference(
                'image-judge', PredictorStrategy.PREDICT
            ),
            image_description=ImageDescriptionSettings(
                inference=_stage_inference(
                    'image-description', PredictorStrategy.PREDICT
                ),
                context_window=ContextWindowSettings(
                    backward_budget=200,
                    forward_budget=200,
                ),
            ),
            splitter=SplitterSettings(
                router=_stage_inference(
                    'splitter-router', PredictorStrategy.PREDICT
                ),
                splitter=_stage_inference(
                    'splitter', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
                context_window=ContextWindowSettings(
                    backward_budget=4,
                    forward_budget=12,
                ),
            ),
            instruction_finder=InstructionFinderSettings(
                start_router=_stage_inference(
                    'instruction-start', PredictorStrategy.PREDICT
                ),
                boundary_router=_stage_inference(
                    'instruction-boundary', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            pedagogical_finder=PedagogicalFinderSettings(
                start_router=_stage_inference(
                    'pedagogical-start', PredictorStrategy.PREDICT
                ),
                boundary_router=_stage_inference(
                    'pedagogical-boundary', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            statement_procedure=StatementProcedureSettings(
                role_typer=_stage_inference(
                    'role-typer', PredictorStrategy.PREDICT
                ),
                statement_partitioner=_stage_inference(
                    'statement-partitioner', PredictorStrategy.PREDICT
                ),
                procedure_partitioner=_stage_inference(
                    'procedure-partitioner', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            exercise_finder=ExerciseFinderSettings(
                start_router=_stage_inference(
                    'exercise-start', PredictorStrategy.PREDICT
                ),
                boundary_router=_stage_inference(
                    'exercise-boundary', PredictorStrategy.CHAIN_OF_THOUGHT
                ),
            ),
            instruction_governance=InstructionGovernanceSettings(
                inference=_stage_inference(
                    'instruction-governance', PredictorStrategy.PREDICT
                ),
            ),
        ),
        ocr=OCRSettings(api_key='test-key'),
    )


def test_build_source_graph_composes_all_source_dependencies():
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)

    graph = composition.build_source_graph(settings, local_models, database)

    assert isinstance(graph, SourceGraph)
    assert isinstance(graph.ocr, OCRNode)
    assert isinstance(graph.content_correction, ContentCorrectionNode)
    assert isinstance(graph.formatter, FormattingNode)
    assert isinstance(graph.ocr.provider, MistralOCRProvider)
    assert graph.ocr.provider._settings is settings.ocr
    assert isinstance(graph.text_seam, TextSeamNode)
    assert isinstance(graph.image_seam, ImageSeamNode)
    assert isinstance(graph.embedding, EmbeddingNode)
    assert isinstance(graph.image_description, ImageDescriptionNode)
    assert isinstance(graph.splitter, SplitterNode)
    assert isinstance(graph.persistence, SourcePersistenceNode)
    assert (
        graph.splitter._context_window
        == settings.source.splitter.context_window
    )
    assert isinstance(graph.instruction_finder, InstructionFinderNode)
    assert isinstance(graph.pedagogical_finder, PedagogicalFinderNode)
    assert isinstance(graph.statement_procedure, StatementProcedureNode)
    assert isinstance(graph.exercise_finder, ExerciseFinderNode)
    assert isinstance(graph.instruction_governance, InstructionGovernanceNode)

    assert type(graph.content_correction._corrector.predictor) is dspy.Predict
    assert type(graph.formatter._formatter.predictor) is dspy.ChainOfThought
    assert type(graph.text_seam._judge.predictor) is dspy.Predict
    assert type(graph.text_seam._rewriter.predictor) is dspy.ChainOfThought
    assert type(graph.image_seam._judge.predictor) is dspy.Predict
    assert (
        graph.image_description._context_window
        == settings.source.image_description.context_window
    )
    assert type(graph.image_description._describer.predictor) is dspy.Predict
    assert type(graph.splitter._router.predictor) is dspy.Predict
    assert type(graph.splitter._splitter.predictor) is dspy.ChainOfThought

    assert (
        type(graph.instruction_finder._start_router.predictor) is dspy.Predict
    )
    assert (
        type(graph.instruction_finder._boundary_router.predictor)
        is dspy.ChainOfThought
    )
    assert (
        type(graph.pedagogical_finder._start_router.predictor) is dspy.Predict
    )
    assert (
        type(graph.pedagogical_finder._boundary_router.predictor)
        is dspy.ChainOfThought
    )
    assert type(graph.statement_procedure._role_typer.predictor) is dspy.Predict
    assert (
        type(graph.statement_procedure._statement_partitioner.predictor)
        is dspy.Predict
    )
    assert (
        type(graph.statement_procedure._procedure_partitioner.predictor)
        is dspy.ChainOfThought
    )
    assert type(graph.exercise_finder._start_router.predictor) is dspy.Predict
    assert (
        type(graph.exercise_finder._boundary_router.predictor)
        is dspy.ChainOfThought
    )
    assert type(graph.instruction_governance._judge.predictor) is dspy.Predict

    assert [
        model_server_profile
        for model_server_profile, _, _ in local_models.calls
    ] == [
        'content-correction',
        'formatting',
        'text-judge',
        'text-rewriter',
        'image-judge',
        'image-description',
        'splitter-router',
        'splitter',
        'instruction-start',
        'instruction-boundary',
        'pedagogical-start',
        'pedagogical-boundary',
        'role-typer',
        'statement-partitioner',
        'procedure-partitioner',
        'exercise-start',
        'exercise-boundary',
        'instruction-governance',
    ]


def test_build_semantic_graph_composes_independent_hub_judges():
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)

    graph = composition.build_semantic_graph(settings, local_models, database)

    assert isinstance(graph, SemanticGraph)
    assert isinstance(
        graph.source_entity_hub._judge_module,
        SourceEntityHubJudgeModule,
    )
    assert isinstance(
        graph.source_event_hub._judge_module,
        SourceEventHubJudgeModule,
    )
    assert isinstance(
        graph.source_predicate_hub._judge_module,
        SourcePredicateHubJudgeModule,
    )
    assert local_models.reranker is graph.source_entity_hub._reranker
    assert local_models.reranker is graph.source_event_hub._reranker
    assert local_models.reranker is graph.source_predicate_hub._reranker
    assert [profile for profile, _, _ in local_models.calls[-6:]] == [
        'gemma-text-32k',
        'gemma-text-32k',
        'gemma-text-32k',
        'gemma-text-32k',
        'gemma-text-32k',
        'gemma-text-32k',
    ]
