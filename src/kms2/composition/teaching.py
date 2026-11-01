"""Dependency composition for local and optional external teaching evidence."""

from kms2.composition.evidence_acquisition import (
    build_evidence_acquisition_graph,
)
from kms2.composition.knowledge import build_knowledge_admission_graph
from kms2.composition.predictors import PredictorFactory, build_token_budget
from kms2.composition.research import ResearchResources
from kms2.composition.services import (
    build_knowledge_search_service,
    build_knowledge_service,
    build_learner_observation_search_service,
    build_learner_observation_service,
    build_source_evidence_service,
    build_source_search_service,
    build_teaching_service,
)
from kms2.composition.source_admission import build_source_admission_graph
from kms2.config.settings import Settings
from kms2.core.embedding import EmbeddingClient
from kms2.core.media_budget import ImagePayloadBudget
from kms2.core.reranking import RerankerClient
from kms2.database.client import DatabaseClient
from kms2.inference.token_counting import InferenceTokenizers
from kms2.langgraph.research.graph import ResearchGraph
from kms2.langgraph.teaching.graph import TeachingGraph
from kms2.langgraph.teaching.interaction_graph import TeachingInteractionGraph
from kms2.local_models.runtime import LocalModelRuntime
from kms2.module.research.examples.researcher import researcher_examples
from kms2.module.research.researcher import (
    ResearcherModule,
    ResearcherSignature,
)
from kms2.module.source_processing.examples.formatting import (
    formatting_examples,
)
from kms2.module.source_processing.formatting import (
    FormatterModule,
    FormatterSignature,
)
from kms2.module.teaching.examples.planner import planner_examples
from kms2.module.teaching.examples.prober import prober_examples
from kms2.module.teaching.examples.teacher import teacher_examples
from kms2.module.teaching.planner import PlannerModule, PlannerSignature
from kms2.module.teaching.prober import ProberModule, ProberSignature
from kms2.module.teaching.teacher import TeacherModule, TeacherSignature
from kms2.node.research.researcher import ResearcherNode
from kms2.node.research.tools import ResearchToolsNode
from kms2.node.teaching.history import TeachingHistoryNode
from kms2.node.teaching.interaction import TeachingInteractionNode
from kms2.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms2.node.teaching.knowledge import TeachingKnowledgeNode
from kms2.node.teaching.observation import TeachingObservationNode
from kms2.node.teaching.persistence import TeachingPersistenceNode
from kms2.node.teaching.teacher import TeachingTeacherNode
from kms2.service.research_evidence import ResearchEvidenceService
from kms2.train.recorder import Recorder


def build_teaching_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embedding_client: EmbeddingClient,
    reranker_client: RerankerClient,
    tokenizers: InferenceTokenizers,
    research_resources: ResearchResources | None = None,
    recorder: Recorder | None = None,
) -> TeachingGraph:
    """Compose the parent turn graph and its bounded interaction children."""
    teaching = settings.teaching
    research_settings = settings.research
    if research_settings.enabled and research_resources is None:
        raise RuntimeError(
            'External research resources are required when research is enabled'
        )
    if not research_settings.enabled and research_resources is not None:
        raise RuntimeError(
            'External research resources must not be acquired when disabled'
        )

    predictors = PredictorFactory(settings, local_models, recorder)
    prober = ProberModule(
        predictors.create(
            ProberModule,
            teaching.inference,
            ProberSignature,
            examples=prober_examples(),
        )
    )
    planner = PlannerModule(
        predictors.create(
            PlannerModule,
            teaching.inference,
            PlannerSignature,
            examples=planner_examples(),
        )
    )
    teacher = TeacherModule(
        predictors.create(
            TeacherModule,
            teaching.inference,
            TeacherSignature,
            examples=teacher_examples(),
        )
    )
    budget = build_token_budget(
        settings,
        tokenizers,
        teaching.inference,
        input_token_budget=teaching.input_token_budget,
        safety_margin_tokens=teaching.safety_margin_tokens,
    )
    teaching_image_budget = ImagePayloadBudget(
        max_images=teaching.max_context_images,
        max_bytes=teaching.max_context_image_bytes,
        max_pixels=teaching.max_context_image_pixels,
    )
    source_search = build_source_search_service(
        database,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        settings=settings.source_search,
    )
    source_evidence = build_source_evidence_service(database)
    knowledge_service = build_knowledge_service(database)
    knowledge_search = build_knowledge_search_service(
        database,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        settings=settings.knowledge.search,
    )
    knowledge_graph = build_knowledge_admission_graph(
        settings,
        database,
        predictors=predictors,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        tokenizers=tokenizers,
    )

    researcher_graph: ResearchGraph | None = None
    source_admission = None
    if research_resources is not None:
        researcher_module = ResearcherModule(
            predictors.create(
                ResearcherModule,
                research_settings.inference,
                ResearcherSignature,
                examples=researcher_examples(),
            )
        )
        research_budget = build_token_budget(
            settings,
            tokenizers,
            research_settings.inference,
            input_token_budget=research_settings.input_token_budget,
            safety_margin_tokens=research_settings.safety_margin_tokens,
        )
        researcher = ResearcherNode(
            researcher_module,
            research_budget,
            image_budget=ImagePayloadBudget(
                max_images=research_settings.max_context_images,
                max_bytes=research_settings.max_context_image_bytes,
                max_pixels=research_settings.max_context_image_pixels,
            ),
        )
        formatter = FormatterModule(
            predictors.create(
                FormatterModule,
                research_settings.formatting.inference,
                FormatterSignature,
                examples=formatting_examples(),
            )
        )
        formatting_budget = build_token_budget(
            settings,
            tokenizers,
            research_settings.formatting.inference,
            input_token_budget=research_settings.formatting.input_token_budget,
            safety_margin_tokens=(
                research_settings.formatting.safety_margin_tokens
            ),
        )
        evidence_service = ResearchEvidenceService(
            web_client=research_resources.tavily,
            media_client=research_resources.media,
            formatter=formatter,
            formatting_budget=formatting_budget,
            research_budget=research_budget,
            ocr_settings=settings.ocr,
            temporary_directory=research_resources.temporary_directory,
            passage_token_budget=research_settings.passage_token_budget,
            formatting_batch_size=research_settings.formatting.batch_size,
            pdf_search_result_limit=research_settings.pdf_search_result_limit,
            max_image_pixels=research_settings.max_image_pixels,
            max_download_bytes=research_settings.max_download_bytes,
            max_concurrency=research_settings.max_concurrency,
        )
        source_admission = build_source_admission_graph(
            settings,
            local_models,
            database,
            embedding_client=embedding_client,
            tokenizers=tokenizers,
            predictors=predictors,
            recorder=recorder,
        )
        researcher_graph = ResearchGraph(
            researcher,
            ResearchToolsNode(evidence_service, research_resources.tavily),
            research_settings.max_tool_calls,
            research_settings.catalog_page_size,
        )

    evidence_graph = build_evidence_acquisition_graph(
        source_search,
        source_evidence,
        knowledge_search,
        knowledge_service,
        researcher=researcher_graph,
        source_admission=source_admission,
        discovery_preview_characters=teaching.discovery_preview_characters,
        catalog_page_size=teaching.evidence_catalog_page_size,
    )
    interaction_graph = TeachingInteractionGraph(
        TeachingTeacherNode(
            prober,
            budget,
            teaching.max_tool_calls,
            image_budget=teaching_image_budget,
            discovery_preview_characters=teaching.discovery_preview_characters,
            research_enabled=research_settings.enabled,
        ),
        TeachingTeacherNode(
            planner,
            budget,
            teaching.max_tool_calls,
            image_budget=teaching_image_budget,
            discovery_preview_characters=teaching.discovery_preview_characters,
            research_enabled=research_settings.enabled,
        ),
        TeachingTeacherNode(
            teacher,
            budget,
            teaching.max_tool_calls,
            image_budget=teaching_image_budget,
            discovery_preview_characters=teaching.discovery_preview_characters,
            research_enabled=research_settings.enabled,
        ),
        TeachingInteractionToolsNode(evidence_graph),
        max_tool_calls=teaching.max_tool_calls,
    )
    observation_service = build_learner_observation_service(database)
    observation_search = build_learner_observation_search_service(
        database,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        settings=teaching.learner_history,
    )
    return TeachingGraph(
        TeachingHistoryNode(observation_search),
        TeachingInteractionNode(interaction_graph),
        TeachingObservationNode(observation_service, embedding_client),
        TeachingKnowledgeNode(
            knowledge_graph, image_budget=teaching_image_budget
        ),
        TeachingPersistenceNode(build_teaching_service(database)),
    )
