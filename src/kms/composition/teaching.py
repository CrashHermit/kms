"""Dependency composition for local and optional external teaching evidence."""

from kms.composition.embedding import EmbeddingResources
from kms.composition.evidence_acquisition import (
    build_evidence_acquisition_node,
)
from kms.composition.knowledge import build_knowledge_admission_graph
from kms.composition.predictors import PredictorFactory, build_token_budget
from kms.composition.research import ResearchResources
from kms.composition.search_selection import build_search_selection_service
from kms.composition.services import (
    build_knowledge_search_service,
    build_knowledge_service,
    build_learner_observation_search_service,
    build_learner_observation_service,
    build_source_evidence_service,
    build_source_search_service,
    build_teaching_service,
)
from kms.composition.source_admission import build_source_admission_service
from kms.composition.tokenization import LanguageModelTokenizers
from kms.config.settings import Settings
from kms.core.media_budget import ImagePayloadBudget
from kms.core.reranking import Reranker
from kms.database.client import DatabaseClient
from kms.langgraph.teaching.graph import TeachingGraph
from kms.local_models.runtime import LocalModelRuntime
from kms.module.research.examples.researcher import researcher_examples
from kms.module.research.researcher import (
    ResearcherModule,
    ResearcherSignature,
)
from kms.module.teaching.examples.planner import planner_examples
from kms.module.teaching.examples.prober import prober_examples
from kms.module.teaching.examples.teacher import teacher_examples
from kms.module.teaching.planner import PlannerModule, PlannerSignature
from kms.module.teaching.prober import ProberModule, ProberSignature
from kms.module.teaching.teacher import TeacherModule, TeacherSignature
from kms.node.research.researcher import ResearcherNode
from kms.node.research.tools import ResearchToolsNode
from kms.node.teaching.history import TeachingHistoryNode
from kms.node.teaching.interaction import TeachingInteractionNode
from kms.node.teaching.interaction_tools import TeachingInteractionToolsNode
from kms.node.teaching.knowledge import TeachingKnowledgeNode
from kms.node.teaching.observation import TeachingObservationNode
from kms.node.teaching.persistence import TeachingPersistenceNode
from kms.node.teaching.teacher import TeachingTeacherNode
from kms.service.research_source_capture import (
    ResearchSourceCaptureService,
)
from kms.service.source_workspace import SourceWorkspaceService
from kms.train.recorder import Recorder


def build_teaching_graph(
    settings: Settings,
    local_models: LocalModelRuntime,
    database: DatabaseClient,
    *,
    embeddings: EmbeddingResources,
    reranker: Reranker,
    tokenizers: LanguageModelTokenizers,
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
    selection = build_search_selection_service(
        settings, predictors=predictors, tokenizers=tokenizers
    )
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
        embedder=embeddings.embedder,
        reranker=reranker,
        settings=settings.source_search,
        selection=selection,
    )
    source_evidence = build_source_evidence_service(database)
    source_workspace = SourceWorkspaceService(source_evidence)
    knowledge_service = build_knowledge_service(database)
    knowledge_search = build_knowledge_search_service(
        database,
        embedder=embeddings.embedder,
        reranker=reranker,
        settings=settings.knowledge.search,
        selection=selection,
    )
    knowledge_graph = build_knowledge_admission_graph(
        settings,
        database,
        predictors=predictors,
        embedder=embeddings.embedder,
        reranker=reranker,
        tokenizers=tokenizers,
    )

    researcher: ResearcherNode | None = None
    research_tools: ResearchToolsNode | None = None
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
        researcher = ResearcherNode(researcher_module, research_budget)
        capture_service = ResearchSourceCaptureService(
            media_client=research_resources.media,
            gotenberg_client=research_resources.gotenberg,
            max_concurrency=research_settings.max_concurrency,
        )
        source_admission = build_source_admission_service(
            settings,
            local_models,
            database,
            embeddings=embeddings,
            tokenizers=tokenizers,
            predictors=predictors,
            recorder=recorder,
        )
        research_tools = ResearchToolsNode(
            capture_service,
            research_resources.tavily,
            source_admission,
        )

    observation_service = build_learner_observation_service(database)
    observation_search = build_learner_observation_search_service(
        database,
        embedder=embeddings.embedder,
        reranker=reranker,
        settings=teaching.learner_observation_search,
        selection=selection,
    )
    evidence_node = build_evidence_acquisition_node(
        source_workspace,
        source_search,
        source_evidence,
        knowledge_search,
        knowledge_service,
        learner_observation_search=observation_search,
        learner_observations=observation_service,
        image_budget=teaching_image_budget,
        discovery_preview_characters=teaching.discovery_preview_characters,
        max_tool_calls=research_settings.max_tool_calls,
        catalog_page_size=teaching.evidence_catalog_page_size,
    )
    prober_node = TeachingTeacherNode(
        prober,
        budget,
        teaching.max_tool_calls,
        image_budget=teaching_image_budget,
        discovery_preview_characters=teaching.discovery_preview_characters,
        research_enabled=research_settings.enabled,
    )
    planner_node = TeachingTeacherNode(
        planner,
        budget,
        teaching.max_tool_calls,
        image_budget=teaching_image_budget,
        discovery_preview_characters=teaching.discovery_preview_characters,
        research_enabled=research_settings.enabled,
    )
    teacher_node = TeachingTeacherNode(
        teacher,
        budget,
        teaching.max_tool_calls,
        image_budget=teaching_image_budget,
        discovery_preview_characters=teaching.discovery_preview_characters,
        research_enabled=research_settings.enabled,
    )
    return TeachingGraph(
        TeachingHistoryNode(observation_search),
        TeachingInteractionNode(source_evidence, observation_service),
        prober_node,
        planner_node,
        teacher_node,
        TeachingInteractionToolsNode(evidence_node),
        TeachingObservationNode(observation_service, embeddings.embedder),
        TeachingKnowledgeNode(
            knowledge_graph, image_budget=teaching_image_budget
        ),
        TeachingPersistenceNode(build_teaching_service(database)),
        researcher,
        research_tools,
        evidence_node.postresearch,
        max_tool_calls=teaching.max_tool_calls,
        max_research_tool_calls=research_settings.max_tool_calls,
    )
