import dspy

from kms2.composition.global_semantic import build_global_semantic_graph
from kms2.composition.source_processing import build_source_processing_graph
from kms2.composition.source_semantic import build_source_semantic_graph
from kms2.config.inference import (
    ContextWindowSettings,
    PredictorStrategy,
    StageInferenceSettings,
    TextInferenceSettings,
    VisionInferenceSettings,
)
from kms2.config.services import OCRSettings
from kms2.config.settings import Settings
from kms2.config.source_processing import (
    ExerciseFinderSettings,
    ExerciseSplitterSettings,
    ImageDescriptionSettings,
    InstructionFinderSettings,
    InstructionGovernanceSettings,
    PedagogicalFinderSettings,
    SourceProcessingSettings,
    StatementProcedureSettings,
    TextSeamSettings,
)
from kms2.database.client import DatabaseClient
from kms2.langgraph.source_processing.graph import SourceProcessingGraph
from kms2.langgraph.source_semantic.graph import SourceSemanticGraph
from kms2.module.source_processing.content_correction import (
    ContentCorrectorModule,
)
from kms2.module.source_semantic.source_entity_hub_judge import (
    SourceEntityHubJudgeModule,
)
from kms2.module.source_semantic.source_event_hub_judge import (
    SourceEventHubJudgeModule,
)
from kms2.module.source_semantic.source_fact_extraction import (
    SourceFactExtractorModule,
)
from kms2.module.source_semantic.source_predicate_hub_judge import (
    SourcePredicateHubJudgeModule,
)
from kms2.module.source_semantic.source_triplet_hub import (
    SourceTripletHubModule,
    SourceTripletHubSignature,
)
from kms2.node.source_processing.content_correction import ContentCorrectionNode
from kms2.node.source_processing.embedding import EmbeddingNode
from kms2.node.source_processing.exercise_finder import ExerciseFinderNode
from kms2.node.source_processing.exercise_splitter import ExerciseSplitterNode
from kms2.node.source_processing.formatting import FormattingNode
from kms2.node.source_processing.image_description import ImageDescriptionNode
from kms2.node.source_processing.image_seam import ImageSeamNode
from kms2.node.source_processing.instruction_finder import InstructionFinderNode
from kms2.node.source_processing.instruction_governance import (
    InstructionGovernanceNode,
)
from kms2.node.source_processing.ocr import OCRNode
from kms2.node.source_processing.pedagogical_finder import PedagogicalFinderNode
from kms2.node.source_processing.persistence import SourcePersistenceNode
from kms2.node.source_processing.statement_procedure import (
    StatementProcedureNode,
)
from kms2.node.source_processing.text_seam import TextSeamNode
from kms2.node.source_semantic.source_triplet_hub import SourceTripletHubNode
from kms2.ocr.mistral import MistralOCRProvider
from kms2.train.recorder import Recorder, RecordingModule


class _RecordingRuntime:
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
) -> TextInferenceSettings:
    return TextInferenceSettings(
        model_server_profile=model_server_profile,
        strategy=strategy,
    )


def _vision_inference(
    model_server_profile: str,
    strategy: PredictorStrategy,
) -> VisionInferenceSettings:
    return VisionInferenceSettings(
        model_server_profile=model_server_profile,
        strategy=strategy,
    )


def _settings() -> Settings:
    return Settings(
        source_processing=SourceProcessingSettings(
            content_correction=_vision_inference(
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
            image_seam=_vision_inference(
                'image-judge', PredictorStrategy.PREDICT
            ),
            image_description=ImageDescriptionSettings(
                inference=_vision_inference(
                    'image-description', PredictorStrategy.PREDICT
                ),
                context_window=ContextWindowSettings(
                    backward_budget=200,
                    forward_budget=200,
                ),
            ),
            exercise_splitter=ExerciseSplitterSettings(
                router=_stage_inference(
                    'exercise-splitter-router', PredictorStrategy.PREDICT
                ),
                splitter=_stage_inference(
                    'exercise-splitter', PredictorStrategy.CHAIN_OF_THOUGHT
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


def _assert_recorded_modules(modules: list[dspy.Module]) -> None:
    assert modules
    assert all(
        isinstance(module.predictor, RecordingModule) for module in modules
    )
    assert all(module.predictor.module is type(module) for module in modules)


def test_build_source_processing_graph_composes_all_source_dependencies():
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)

    graph = build_source_processing_graph(settings, local_models, database)

    assert isinstance(graph, SourceProcessingGraph)
    assert isinstance(graph.ocr, OCRNode)
    assert isinstance(graph.content_correction, ContentCorrectionNode)
    assert isinstance(graph.formatter, FormattingNode)
    assert isinstance(graph.ocr.provider, MistralOCRProvider)
    assert graph.ocr.provider._settings is settings.ocr
    assert isinstance(graph.text_seam, TextSeamNode)
    assert isinstance(graph.image_seam, ImageSeamNode)
    assert isinstance(graph.embedding, EmbeddingNode)
    assert isinstance(graph.image_description, ImageDescriptionNode)
    assert isinstance(graph.exercise_splitter, ExerciseSplitterNode)
    assert isinstance(graph.persistence, SourcePersistenceNode)
    assert not hasattr(graph.persistence, '_schema_initializer')
    assert (
        graph.exercise_splitter._context_window
        == settings.source_processing.exercise_splitter.context_window
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
        == settings.source_processing.image_description.context_window
    )
    assert type(graph.image_description._describer.predictor) is dspy.Predict
    assert type(graph.exercise_splitter._router.predictor) is dspy.Predict
    assert (
        type(graph.exercise_splitter._exercise_splitter.predictor)
        is dspy.ChainOfThought
    )

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
        'exercise-splitter-router',
        'exercise-splitter',
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


def test_all_composed_dspy_modules_are_recorded(tmp_path):
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)
    recorder = Recorder(tmp_path)

    processing_graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        recorder=recorder,
    )
    semantic_graph = build_source_semantic_graph(
        settings,
        local_models,
        database,
        recorder=recorder,
    )
    global_graph = build_global_semantic_graph(
        settings,
        local_models,
        database,
        recorder=recorder,
    )

    _assert_recorded_modules(
        [
            processing_graph.content_correction._corrector,
            processing_graph.formatter._formatter,
            processing_graph.text_seam._judge,
            processing_graph.text_seam._rewriter,
            processing_graph.image_seam._judge,
            processing_graph.image_description._describer,
            processing_graph.exercise_splitter._router,
            processing_graph.exercise_splitter._exercise_splitter,
            processing_graph.instruction_finder._start_router,
            processing_graph.instruction_finder._boundary_router,
            processing_graph.pedagogical_finder._start_router,
            processing_graph.pedagogical_finder._boundary_router,
            processing_graph.statement_procedure._role_typer,
            processing_graph.statement_procedure._statement_partitioner,
            processing_graph.statement_procedure._procedure_partitioner,
            processing_graph.exercise_finder._start_router,
            processing_graph.exercise_finder._boundary_router,
            processing_graph.instruction_governance._judge,
        ]
    )
    _assert_recorded_modules(
        [
            semantic_graph.source_fact_extraction._extractor,
            semantic_graph.source_triplet_decomposition._decomposer,
            semantic_graph.source_entity_description._module,
            semantic_graph.source_event_description._module,
            semantic_graph.source_predicate_description._module,
            semantic_graph.source_statement_description._module,
            semantic_graph.source_procedure_description._module,
            semantic_graph.source_entity_hub._module,
            semantic_graph.source_entity_hub._judge_module,
            semantic_graph.source_event_hub._module,
            semantic_graph.source_event_hub._judge_module,
            semantic_graph.source_predicate_hub._module,
            semantic_graph.source_predicate_hub._judge_module,
            semantic_graph.source_statement_hub._module,
            semantic_graph.source_statement_hub._judge_module,
            semantic_graph.source_procedure_hub._module,
            semantic_graph.source_procedure_hub._judge_module,
            semantic_graph.source_triplet_hub._module,
        ]
    )
    _assert_recorded_modules(
        [
            global_graph.global_entity_hub._module,
            global_graph.global_entity_hub._judge_module,
            global_graph.global_event_hub._module,
            global_graph.global_event_hub._judge_module,
            global_graph.global_predicate_hub._module,
            global_graph.global_predicate_hub._judge_module,
            global_graph.global_statement_hub._module,
            global_graph.global_statement_hub._judge_module,
            global_graph.global_procedure_hub._module,
            global_graph.global_procedure_hub._judge_module,
        ]
    )


def test_build_source_processing_graph_wraps_predictors_when_recording_is_enabled(
    tmp_path,
):
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)
    recorder = Recorder(tmp_path)

    graph = build_source_processing_graph(
        settings,
        local_models,
        database,
        recorder=recorder,
    )

    content_predictor = graph.content_correction._corrector.predictor
    assert isinstance(content_predictor, RecordingModule)
    assert type(content_predictor.predictor) is dspy.Predict
    assert content_predictor.module is ContentCorrectorModule


def test_build_source_semantic_graph_wraps_predictors_when_recording_is_enabled(
    tmp_path,
):
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)
    recorder = Recorder(tmp_path)

    graph = build_source_semantic_graph(
        settings,
        local_models,
        database,
        recorder=recorder,
    )

    predictor = graph.source_fact_extraction._extractor.predictor
    assert isinstance(predictor, RecordingModule)
    assert type(predictor.predictor) is dspy.Predict
    assert predictor.module is SourceFactExtractorModule


def test_build_source_semantic_graph_composes_independent_hub_judges():
    settings = _settings()
    local_models = _RecordingRuntime()
    local_models.calls = []
    database = DatabaseClient(settings.database)

    graph = build_source_semantic_graph(settings, local_models, database)

    assert isinstance(graph, SourceSemanticGraph)
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
    assert isinstance(graph.source_triplet_hub, SourceTripletHubNode)
    assert isinstance(
        graph.source_triplet_hub._module,
        SourceTripletHubModule,
    )
    assert type(graph.source_triplet_hub._module.predictor) is dspy.Predict
    assert [
        profile
        for profile, _, signature in local_models.calls
        if signature is SourceTripletHubSignature
    ] == ['qwen3.8-9b-distill-text']
    assert [profile for profile, _, _ in local_models.calls[:2]] == [
        'qwen3.8-9b-distill-text',
        'qwen3.8-9b-distill-text',
    ]
    assert [profile for profile, _, _ in local_models.calls[2:12]] == [
        'qwen3.8-9b-distill-text',
    ] * 10
