# RAGA: Reading-And-Graph-building-Agent for Autonomous Knowledge Graph Construction and Retrieval-Augmented Generation

Chengrui Han  Zesheng Cheng  Affiliation: Qingdao University 

###### Abstract

Existing LLM-driven knowledge graph (KG) construction methods predominantly employ stateless batch processing pipelines, exhibiting structural deficiencies in cross-chunk semantic relation capture, entity disambiguation, and construction process interpretability. These limitations undermine KG quality, retrieval precision, and deployment trust in high-stakes domains.

We propose RAGA (Reading And Graph-building Agent), an LLM-based autonomous KG construction and retrieval fusion framework. RAGA provides an atomic toolset supporting full KG lifecycle CRUD operations and embeds a “Read–Search–Verify–Construct” cognitive constraint into a ReAct tool loop. A KG-vector synchronization mechanism enables hybrid symbolic-vector retrieval, while evidence-anchored verification links every knowledge entry to its source text for auditable provenance.

Preliminary experiments on a subset of the QASPER scientific QA dataset indicate that RAGA’s fusion retrieval outperforms zero-shot baselines, with KG integration providing measurable gains in both answer and evidence quality. The framework design and experimental baseline serve as a reference for agent-driven autonomous KG construction.

##  1 Introduction

Knowledge Graphs (KGs) organize heterogeneous information as computable and inferable graph structures with entities as nodes and relations as edges. In natural language processing, KGs provide explicit world-knowledge constraints for semantic search, question answering, and text understanding [Zhu et al., 2024]. In scientific discovery, KGs are employed to extract domain knowledge from the literature and construct evolvable disciplinary knowledge networks [Dasigi et al., 2021, Zhang and Soh, 2024]. With the proliferation of Large Language Models (LLMs), the synergistic integration of KGs and LLMs has become a prominent research direction [Zhu et al., 2024].

Traditional KG construction relies on manual annotation and expert-defined rules, incurring high costs and limited scalability. Researchers have explored leveraging LLMs’ semantic understanding capabilities to automatically extract entities and relations from unstructured text, yielding a series of LLM-driven construction methods [Zhang and Soh, 2024, Lairgi et al., 2024]. While effective in controlled settings, these methods exhibit three structural deficiencies in large-scale, incremental, multi-source heterogeneous data scenarios.

Retrieval-Augmented Generation (RAG) technology provides a technical pathway for deep integration of LLMs and KGs. The RAG framework proposed by Lewis et al. [Lewis et al., 2020] combines external knowledge bases with parametric language models, effectively mitigating LLM hallucination. Gao et al. [Gao et al., 2023] conducted a systematic survey of RAG techniques, noting the evolution from simple vector retrieval toward structured knowledge retrieval. KGs provide LLMs with precise and verifiable factual grounding as structured external knowledge sources. LLMs, in turn, offer semantic understanding capabilities for KG construction and completion.

In LLM-driven KG construction, Edge et al. [Edge et al., 2024] proposed GraphRAG, employing a “local-to-global” construction strategy that performs local knowledge extraction on text chunks and constructs global summary graphs via community detection. Guo et al. [Guo et al., 2025] proposed LightRAG, optimizing KG query efficiency through a dual-layer retrieval mechanism. Liang et al. [Liang et al., 2025] proposed KAG, designing knowledge-enhanced generation pipelines for professional domains. Lairgi et al. [Lairgi et al., 2024] proposed iText2KG, adopting an incremental construction strategy supporting progressive KG construction from zero-shot scenarios. These methods follow fixed batch processing pipelines and lack dynamic regulation over the construction process.

Three structural deficiencies characterize existing methods. First, cross-chunk long-range semantic relation loss. Existing methods segment long documents into fixed-length text chunks and perform independent knowledge extraction on each chunk, severing cross-chunk semantic associations. For example, a method introduced in the introduction of a scientific paper may be concretely described in the experimental section and comparatively evaluated in the discussion. If each chunk is extracted independently, these cross-chunk causal, comparative, and evolutionary relations cannot be effectively captured. Although GraphRAG [Edge et al., 2024] establishes global associations through community detection, its global summaries remain aggregations of local information and do not recover fine-grained cross-chunk relations.

Second, entity redundancy and insufficient disambiguation. When the same entity appears with different surface forms in text, traditional methods cannot recognize it as the same node, producing redundant semantically overlapping nodes in the KG. “Convolutional neural network,” “CNN,” and “Convolutional Neural Network” all refer to the same concept. Without effective entity linking and disambiguation, they will be created as multiple independent nodes. As data sources increase, semantic redundancy grows exponentially, diluting the KG’s information density. The EDC framework [Zhang and Soh, 2024] proposed an entity canonicalization workflow, but its disambiguation capability in incremental construction scenarios is limited.

Third, the construction process is uninterpretable and unauditable. Traditional methods treat knowledge extraction as an end-to-end black box: input text, output triples. Researchers cannot trace which original texts knowledge entries originate from or what reasoning process was involved. In domains demanding high interpretability such as scientific research and medical decision-making, KGs lacking transparent construction processes struggle to earn deployment trust. Sarthi et al. [Sarthi et al., 2024] proposed RAPTOR, enhancing retrieval hierarchy through recursive abstractive processing, but lacking fine-grained provenance. Dasigi et al. [Dasigi et al., 2021] constructed the QASPER dataset emphasizing the importance of evidence anchoring, but existing methods rarely treat evidence provenance as a core design objective.

Researchers have attempted to apply agent technologies to KG construction, framing it as a dynamic cognitive process. In this paradigm, agents iteratively perceive text, retrieve existing knowledge, verify new discoveries, and update the knowledge base. This enables incremental and interactive knowledge construction. Yao et al. [Yao et al., 2023] proposed the ReAct paradigm that interleaves reasoning and action, using chain-of-thought to guide LLMs in multi-step decision-making, forming a key foundation for agent-driven KG construction. Jiang et al. [Jiang et al., 2025] proposed KG-Agent, enabling complex reasoning over KGs through tool invocation. However, the framework only supports read operations on existing KGs. It lacks create, update, and delete capabilities, precluding autonomous KG construction. Anokhin et al. [Anokhin et al., 2024] proposed AriGraph, using a dual-layer memory architecture to build world-model KG representations for LLM agents, but lacks vector space integration and active entity disambiguation.

Incomplete tool capabilities constitute a primary limitation. KG-Agent focuses on complex reasoning over existing KGs and its toolset is optimized for query and retrieval, lacking write operations required for autonomous KG construction. KG construction is a continuously evolving knowledge management process requiring entity creation, attribute updates, erroneous information deletion, and duplicate node merging. An agent lacking full CRUD capabilities cannot autonomously complete the full lifecycle management. Opaque cognitive workflows represent another constraint. While iText2KG [Lairgi et al., 2024] supports sequential processing of text streams, its internal extraction logic remains a black box and lacks explicit cognitive phase delineation. Human experts constructing KGs progress through reading, understanding, verification, and construction stages; current methods do not structurally embed this workflow into the construction process. The separation between memory and vector spaces also demands attention. AriGraph’s [Anokhin et al., 2024] dual-layer memory architecture distinguishes episodic and semantic memory, but the semantic memory employs symbolic graph storage without real-time alignment with dense vector representations. Modern RAG systems treat vector retrieval and graph retrieval as complementary knowledge access modalities; if the symbolic and vector layers remain desynchronized over time, retrieval inconsistency will result [Sarmah et al., 2024].

To address these structural deficiencies, this paper proposes an LLM-based autonomous KG construction and retrieval fusion method, with the Reading And Graph-building Agent (RAGA) framework as its implementation. The main contributions are:

  * •

An autonomous knowledge-operating toolset. The toolset is designed around reading behaviors, including paragraph reading, context browsing, fusion retrieval, entity and relation CRUD operations, merge operations, human review markers, deferred tasks, and progress queries. The toolset enables agents to autonomously manage the full KG lifecycle.

  * •

An LLM-driven Read–Search–Verify–Construct cognitive loop. Structuring the human expert’s knowledge construction process into a ReAct-style multi-turn tool-calling cycle. The reading phase parses text chunks and identifies important information; the search phase retrieves relevant evidence using existing KGs and context; the verification phase judges new knowledge reliability using original text and tool-returned results; the construction phase writes verified knowledge into the KG in standardized form. A reading-progress state machine manages long-document processing with four states: PENDING, READING, VERIFIED, and ARCHIVED.

  * •

A KG-vector synchronization mechanism. After writing KG structural objects, the system supplements chunk, entity, or HyperNode vector representations and performs cross-storage reference write-back. On vector write failure, the system compensates by removing already-written graph objects and recording alerts. This enables hybrid retrieval where agents can simultaneously leverage graph structure reasoning and vector semantic matching.

  * •

Evidence-anchored verification. All primary knowledge entries in the KG are associated with their original textual evidence. The system maintains structured provenance records including metadata such as source text chunk, evidence snippet, operation type, and confidence level, enabling reverse-source tracing of knowledge entries.




##  2 Related Work

###  2.1 Agent-Based Knowledge Graph Construction

Applying agent technology to KG construction seeks to leverage LLM reasoning and planning capabilities to transform knowledge extraction from fixed batch pipelines into dynamically interactive cognitive processes.

KG-Agent [Jiang et al., 2025] is a representative work in this direction. The framework supports multi-hop reasoning over KGs through modular tool interfaces, encapsulating knowledge querying, path reasoning, and answer generation as independent tool functions. KG-Agent’s primary advantages lie in multi-hop reasoning accuracy and speed. However, its toolset is limited to reading and reasoning over existing KGs and lacks write operation capabilities (entity creation, relation addition, or knowledge correction). This precludes its use for incremental KG construction from scratch.

AriGraph [Anokhin et al., 2024] features a dual-layer memory model: episodic memory stores agent interaction trajectories, while semantic memory preserves structured world-model knowledge in KG form. AriGraph’s semantic memory only supports symbolic graph queries without integration with dense vector representations, unable to perform semantic retrieval via vector similarity. AriGraph lacks an active entity disambiguation mechanism; when the same entity appears with different surface forms, the system creates new nodes rather than merging with existing ones.

UrbanKGent [Ning and Liu, 2024] targeted urban KG construction with an agent-driven construction and completion pipeline, leveraging agent planning capabilities to coordinate geographic entity recognition, spatial relation extraction, and domain knowledge completion sub-modules. While demonstrating the application potential of agent frameworks in vertical domains, its geocoding rules and spatial relation templates are difficult to transfer to other domains.

In general agent memory management, MemGPT [Packer et al., 2023] analogizes LLMs to operating systems, distinguishing limited-capacity “main context” from pageable “external memory” for dynamic context resource allocation. MemoryBank [Zhong et al., 2024] designed long-term memory storage and retrieval mechanisms based on temporal decay and importance sampling. Both works focus on general dialogue scenarios without optimization for KG-specific structured characteristics.

###  2.2 LLM-Driven Knowledge Graph Extraction

LLM-driven KG extraction aims to leverage LLMs’ semantic understanding and generation capabilities to automatically extract entity, relation, and attribute information from unstructured text. By processing strategy, existing methods fall into batch-processing and incremental categories.

Batch-processing methods primarily include GraphRAG [Edge et al., 2024] and LightRAG [Guo et al., 2025]. GraphRAG employs a local-to-global construction approach: first segmenting documents into fixed-length text chunks, using LLMs to extract entities and relations from each chunk to form local knowledge subgraphs, then generating global summaries through community detection. This strategy suits offline large-scale document collections requiring global consistency, but the chunking process sacrifices fine-grained cross-chunk semantic association capture and lacks cross-chunk entity alignment, readily producing semantically redundant nodes. LightRAG employs a dual-layer retrieval mechanism: concrete entity retrieval at the low level and abstract concept retrieval at the high level, improving retrieval efficiency. However, its construction phase uses batch processing, and its incremental update capability is primarily limited to document append rather than complex entity disambiguation and relation revision.

Incremental methods are primarily represented by iText2KG [Lairgi et al., 2024] and the EDC framework [Zhang and Soh, 2024]. iText2KG maintains a continuously growing entity reference table, comparing existing entities during each extraction pass to identify equivalent expressions and perform merging. However, its parsing workflow follows a fixed pipeline; when encountering conflicting information, the system cannot actively backtrack or verify, only handling situations according to preset rules. The EDC framework formalizes knowledge extraction into three consecutive stages—Extract, Define, and Canonicalize—showing good performance in entity canonicalization, but its batch processing design struggles to adapt to continuous streaming document input.

KAG [Liang et al., 2025] designed knowledge-enhanced generation pipelines for professional domains such as healthcare and law. RAPTOR [Sarthi et al., 2024] processes data through recursive abstraction to generate tree-structured retrieval indices, progressively aggregating fine-grained text information into abstract concepts. These works are effective in specific scenarios but lack explicit cognitive phase delineation and interpretable verification mechanisms during construction.

Wang et al. [2025] proposed PIKE-RAG, a multi-layer heterogeneous graph framework targeting industrial knowledge extraction and rationale-augmented generation. PIKE-RAG organizes knowledge into an information source layer, corpus layer, and distilled knowledge layer, enabling both semantic understanding and rationale-based retrieval. A feedback loop between knowledge organization and extraction refines the knowledge base iteratively. While PIKE-RAG shares a similar layered architectural philosophy with RAGA, its primary focus is rationale-augmented generation for knowledge-intensive tasks rather than autonomous CRUD operations and evidence-anchored provenance tracking.

Li et al. [2025] proposed StructRAG, which introduces inference-time hybrid information structurization for knowledge-intensive reasoning. StructRAG employs a hybrid structure router (trained via DPO) to select the optimal structure type (table, graph, or catalogue), a scattered knowledge structurizer to transform raw documents into structured knowledge, and a structured knowledge utilizer to decompose complex questions for precise answer inference. While StructRAG dynamically selects structural representations at inference time, it does not provide autonomous KG lifecycle management with evidence-anchored provenance.

In contrast, the proposed agent-driven approach embeds the Read–Search–Verify–Construct cognitive constraint into the LLM tool loop. It replaces fixed single-pass extraction pipelines, endows the Agent with complete CRUD operation capabilities, and requires each knowledge entry to be anchored to original textual evidence. This enables the Agent to proactively repair construction errors and resolve semantic conflicts.

###  2.3 Agent Memory and Knowledge Operations

Agent memory management constitutes foundational infrastructure for accomplishing complex tasks, accumulating domain knowledge, and maintaining long-term consistency.

The ReAct paradigm [Yao et al., 2023] interleaves reasoning and action, enabling LLM agents to make multi-step decisions in dynamic environments by alternately generating thought chains and action commands. ReAct demonstrates that explicit reasoning processes enhance agent action quality, a principle influencing subsequent memory system design directions—namely, how to organize agent interaction histories into memory structures amenable to efficient retrieval and utilization.

AtomMem [Huo et al., 2026] treats agent memory as atomic knowledge units, supporting CRUD operations and confidence assessment. However, AtomMem’s memory representations are flat text fragment collections lacking structured relational organization, making it difficult to support complex knowledge reasoning. A-MEM [Xu et al., 2025] designed a three-layer architecture comprising working memory, short-term memory, and long-term memory, capable of assessing memory importance and supporting associative retrieval and temporal decay. All-Mem [Lv et al., 2026] employs dynamic topology evolution for lifelong agent memory management, enabling automatic adjustment of inter-node connection structures based on new information, though the synchronization problem between symbolic KGs and vector representations remains incompletely resolved.

In adaptive retrieval, Self-RAG [Asai et al., 2024] trains LLMs to learn when to retrieve, what to retrieve, and how to utilize retrieval results, achieving self-reflective retrieval behavior. Adaptive-RAG [Jeong et al., 2024] dynamically selects retrieval strategies based on question complexity—simple questions directly generate answers, while complex questions trigger multi-step retrieval. IRCoT [Trivedi et al., 2023] interleaves the retrieval process with chain-of-thought reasoning, dynamically retrieving relevant knowledge during reasoning step generation.

###  2.4 Knowledge Graph Retrieval Augmentation

KG Retrieval-Augmented Generation leverages KGs’ structured knowledge to enhance LLM reasoning capability and answer accuracy in knowledge-intensive tasks. By the coupling mode of retrieval and reasoning, existing methods fall into graph-query and graph-reasoning categories.

Graph-query methods treat KGs as structured external databases, retrieving relevant subgraphs through graph traversal and feeding them as context to LLMs for answer generation. Think-on-Graph [Sun et al., 2024] enables LLMs to perform deep reasoning over KGs, enhancing interpretability through iterative reasoning path expansion and path relevance evaluation. RoG [Luo et al., 2024] trains LLMs to learn reasoning strategies faithful to graph facts. GNN-RAG [Mavromatis and Karypis, 2025] leverages GNN graph encoding capabilities for structure-aware representation of retrieved subgraphs. G-Retriever [He et al., 2024] combines graph structure retrieval with text generation, effective in scenarios such as scientific literature graphs.

In agent-driven retrieval augmentation, RAG-Critic [Dong et al., 2025] employs an automated critic agent to guide retrieval-augmented generation, optimizing retrieval quality through iterative critique and feedback. HippoRAG [Gutierrez et al., 2024] mimics hippocampal indexing and cortical storage mechanisms to design long-term memory systems supporting persistent knowledge storage and associative retrieval, with its indexing mechanism informing the KG-vector synchronization design in this work.

In structured and unstructured knowledge fusion, HybridRAG [Sarmah et al., 2024] simultaneously utilizes KG structured retrieval and vector database semantic retrieval, integrating both result sets through hybrid ranking. HybridRAG validates the complementarity of structured knowledge and dense vector representations, but its retrieval-level fusion does not extend to the construction level—KG updates do not automatically trigger corresponding adjustments to vector indices. The KG-vector synchronization mechanism in this work addresses this gap.

Lee et al. [2025] proposed HybGRAG, a hybrid retrieval-augmented generation framework for textual and relational knowledge bases. HybGRAG addresses “hybrid” questions that require both textual and relational information from semi-structured knowledge bases, employing a retriever bank and a critic module for adaptive retrieval refinement. Its agentic design automatically refines retrieval outputs through critic feedback, achieving 51% relative improvement in Hit@1 on the STaRK benchmark. While HybGRAG demonstrates strong hybrid retrieval capability, its retrieval-level fusion does not extend to construction-level KG-vector synchronization.

Luo et al. [2025] proposed Graph-R1, the first agentic GraphRAG framework trained end-to-end via reinforcement learning. Graph-R1 introduces lightweight knowledge hypergraph construction, models retrieval as multi-turn agent–environment interaction (“think–retrieve–rethink–generate”), and optimizes the agent process through an end-to-end reward mechanism integrating generation quality, retrieval relevance, and structural reliability. While Graph-R1 leverages RL for multi-turn retrieval optimization, RAGA employs prompt-engineered cognitive constraints (Read–Search–Verify–Construct) without requiring RL training, offering a training-free alternative for scenarios where RL infrastructure is unavailable.

###  2.5 Comparative Analysis

Existing methods each possess strengths in individual aspects, but none comprehensively covers the capability combination addressed in this work. GraphRAG [Edge et al., 2024] and LightRAG [Guo et al., 2025] perform well in batch construction and multi-hop reasoning but lack incremental update capability. iText2KG [Lairgi et al., 2024] supports incremental construction but follows a fixed pipeline with limited entity disambiguation capability. AriGraph [Anokhin et al., 2024] and KG-Agent [Jiang et al., 2025] adopt agent-driven approaches, but the former lacks vector integration and disambiguation mechanisms while the latter only supports read operations. The EDC framework [Zhang and Soh, 2024] performs well in entity canonicalization but its batch processing design cannot accommodate streaming construction. KAG [Liang et al., 2025] has domain-specific advantages but incurs high adaptation costs.

Recent concurrent and complementary works partially address subsets of these capabilities. PIKE-RAG [Wang et al., 2025] provides multi-layer heterogeneous graph construction with rationale augmentation but lacks autonomous CRUD tool operations and evidence-anchored provenance. StructRAG [Li et al., 2025] enables inference-time structure selection but does not support incremental KG lifecycle management. HybGRAG [Lee et al., 2025] achieves strong hybrid retrieval through an agentic retriever–critic design but does not address construction-level KG-vector synchronization. Graph-R1 [Luo et al., 2025] introduces RL-based multi-turn agentic retrieval over hypergraphs but relies on reinforcement learning training rather than prompt-engineered cognitive constraints. However, no existing method simultaneously provides all four capabilities: (1) full CRUD write-loop capabilities for autonomous KG lifecycle management; (2) real-time KG-vector consistency with failure compensation; (3) evidence-anchored provenance linking every knowledge entry to its source text; and (4) an auditable agent execution paradigm with explicit cognitive phase constraints. The proposed RAGA framework addresses all four requirements in a unified architecture.

##  3 Method

This section presents the RAGA methodological framework, covering problem formalization (Sec. 3.1), four-layer system architecture (Sec. 3.2), agent core design (Sec. 3.3), memory architecture (Sec. 3.4), KG-vector synchronization (Sec. 3.5), and three-layer fusion retrieval (Sec. 3.6).

###  3.1 Problem Formalization

Document Collection and Knowledge Graph Formalization. Given a document collection 𝒟={d1,d2,…,dn}\mathcal{D}=\\{d_{1},d_{2},\ldots,d_{n}\\}, each document did_{i} is segmented by a chunker 𝒞\mathcal{C} into an ordered sequence of text chunks 𝒞​(di)={ci,1,ci,2,…,ci,mi}\mathcal{C}(d_{i})=\\{c_{i,1},c_{i,2},\ldots,c_{i,m_{i}}\\}. The chunker 𝒞\mathcal{C} supports four strategies: fixed-size chunking (FIXED_SIZE), semantic sentence-boundary chunking (SEMANTIC), paragraph-level chunking (PARAGRAPH), and deep structure-aware chunking (STRUCTURAL). Section 3.2 will justify structure-aware chunking as the default strategy. Each text chunk cc carries a tuple (pos,struct,src)(\text{pos},\text{struct},\text{src}), respectively denoting position offset in the source document, structural label (e.g., heading, body, list item), and source document identifier.

The system’s core objective is to construct and maintain a dynamically evolving knowledge graph 𝒢t=(𝒱t,ℰt)\mathcal{G}_{t}=(\mathcal{V}_{t},\mathcal{E}_{t}), where tt denotes processing timestamp. The vertex set 𝒱t\mathcal{V}_{t} contains two node types: (1) standard entity nodes v∈𝒱tstdv\in\mathcal{V}_{t}^{\text{std}}, carrying attribute label label​(v)\text{label}(v) and unique identifier id​(v)\text{id}(v); (2) HyperNodes h∈𝒱thyperh\in\mathcal{V}_{t}^{\text{hyper}}, used to aggregate semantically equivalent or highly related entity clusters, reducing graph redundancy [Edge et al., 2024]. The edge set ℰt⊆𝒱t×𝒱t×ℛ\mathcal{E}_{t}\subseteq\mathcal{V}_{t}\times\mathcal{V}_{t}\times\mathcal{R} consists of directed edges with typed relations, where ℛ\mathcal{R} is a dynamically extendable relation type set. The system maintains an attribute map 𝒫t:𝒱t∪ℰt→2𝒜\mathcal{P}_{t}:\mathcal{V}_{t}\cup\mathcal{E}_{t}\rightarrow 2^{\mathcal{A}} over 𝒢t\mathcal{G}_{t}, assigning multi-valued attribute sets to nodes or edges, with 𝒜\mathcal{A} as the key-value pair domain.

Aligned Vector Index and HyperNode Formalization. To support semantic retrieval, the system maintains an aligned vector index ℳt\mathcal{M}_{t} mapping each graph object x∈𝒱t∪ℰtx\in\mathcal{V}_{t}\cup\mathcal{E}_{t} into dense vector space ℝd\mathbb{R}^{d}:

| ℳt:𝒱t∪ℰt→ℝd\mathcal{M}_{t}:\mathcal{V}_{t}\cup\mathcal{E}_{t}\rightarrow\mathbb{R}^{d} |  | (1)  
---|---|---|---  
  
The index ℳt\mathcal{M}_{t} is realized through a vector encoding service: text chunks, entities, and relations are encoded as vector representations upon writing, maintaining bidirectional references with object identifiers in the graph database. HyperNodes serve as bridges among chunks, documents, and related entities, with their vectors derivable from text chunk representations or member entity representations. Under member aggregation, the vector representation of HyperNode hh can be defined as the weighted centroid of member entity vector representations:

| 𝐦h=1|𝒮​(h)|​∑v∈𝒮​(h)w​(v)⋅ℳt​(v)\mathbf{m}_{h}=\frac{1}{|\mathcal{S}(h)|}\sum_{v\in\mathcal{S}(h)}w(v)\cdot\mathcal{M}_{t}(v) |  | (2)  
---|---|---|---  
  
where 𝒮​(h)\mathcal{S}(h) is the entity set associated with HyperNode hh, and w​(v)w(v) is the confidence weight of entity vv. In engineering implementation, chunk text vectors are concurrently preserved to enable direct evidence chunk recall from the vector store during retrieval.

Evaluation Dimensions. The system is designed around four evaluation dimensions:

  * •

Quality: Accuracy of extracted entities and relations, including entity denoising, relation confidence, and schema compliance;

  * •

Coverage: Efficiency of document information conversion into the KG, primarily tracking covered text chunk ratio and extracted entity density;

  * •

Retrieval: Precision and recall of fusion retrieval results;

  * •

Provenance: Coverage ratio of graph-object-to-source-chunk traceability chains.




In Section 4 experiments, retrieval efficacy is indirectly reflected through external metrics such as Evidence F1, Evidence Precision/Recall, and Recall@K; provenance completeness is manifested through evidence lists and document/paragraph identifiers in prediction records. Systematic evaluation of Quality and Coverage is deferred for future diagnostic toolchain support.

###  3.2 System Architecture

RAGA adopts a top-down four-layer architecture consisting of the Tool Layer, Reading Layer, Memory Layer, and Retrieval Layer. Each layer interacts through clearly defined interface contracts, with lower layers transparent to upper layers, enabling independent extension and replacement. The design references the classical perception–cognition–memory–action decomposition in modern agent architectures [Yao et al., 2023] while incorporating KG-specific structured storage requirements.

Tool Layer. The Tool Layer encapsulates atomic tools for KG construction and reading behaviors, providing a unified graph interaction interface for the upper-layer Agent. The toolset is functionally partitioned into six categories: reading and retrieval, entity operations, relation operations, review and deferred tasks, progress queries, and implicit provenance recording. The tool adaptation layer automatically invokes provenance storage on write operations for recording source text, operation type, and confidence. All tools interact with Agent state through the ToolState bridge class.

Reading Layer. The Reading Layer forms the cognitive center of the system, primarily responsible for having the Agent read documents paragraph by paragraph, extract structured knowledge, and perform graph operations and decisions. It uses LangGraph’s StateGraph as the orchestration paradigm, modeling the Agent’s cognitive loop as a directed state machine. The PromptAssembler module performs dynamic prompt assembly within the Reading Layer, mitigating context dilution issues during long-document processing.

Memory Layer. The Memory Layer employs a heterogeneous storage architecture comprising semantic memory, episodic and progress memory, working memory, and vector memory. The design is informed by the multi-memory co-architecture in AriGraph [Anokhin et al., 2024] but instantiates it concretely for KG-specific scenarios. Neo4j stores entities, relations, HyperNodes, and document evidence edges. MongoDB stores original documents including chunking, progress projections, and provenance records. Redis is primarily used for schema caching and short-term chain state, while Milvus stores chunk and entity vectors. The Reading Layer accesses each memory tier through dependency injection on demand.

Retrieval Layer. The Retrieval Layer implements a three-layer fusion retrieval pipeline combining vector similarity search with graph topology expansion. The processing flow has three stages: (1) semantic candidate chunks are recalled from the vector store; (2) entity anchors associated with candidate chunks are extracted, and multi-hop expansion and HyperNode evidence chunk lookback are executed through Neo4j; (3) the RRF (Reciprocal Rank Fusion) algorithm fuses vector candidates and graph candidates into a unified ranking. This fusion strategy draws on the multi-path retrieval complementarity ideas in HybridRAG [Sarmah et al., 2024] and LightRAG [Guo et al., 2025]. A timeout fallback path is also engineered: when the KG layer times out or fails, the system degrades gracefully and returns vector-layer results.

###  3.3 Agent Core Design

####  3.3.1 LangGraph-Based State Machine Architecture

The Agent’s cognitive loop uses LangGraph StateGraph as the orchestration foundation. The outer state machine primarily handles cognitive behavior, situated in the react_loop node, which contains multi-round ReAct tool loops and constitutes the main working node of the system. Other nodes handle record-keeping and flow management. The LLM system prompt includes the Read–Search–Verify–Construct four-phase cognitive constraint; the LLM autonomously follows these constraints, with the system using prompt engineering and tool contracts for guidance rather than hard-coded phase node enforcement.

The state machine topology is: the starting node bootstrap_schema performs domain detection and schema initialization, outputting an initial schema definition Σ0\Sigma_{0} and domain label τ\tau. read_paragraph then loads the current paragraph ccurrc_{\text{curr}} from the document queue. react_loop completes multiple LLM-output tool calls, tool execution, and observation result backfill rounds within a single node until the LLM returns a paragraph completion summary or reaches the maximum round limit. next_paragraph then advances the reading pointer and determines whether the document end has been reached. If unfinished cross-paragraph todo items exist, they are processed through the handle_todos node. The process finally reaches the finish node for session cleanup and statistics aggregation.

The Checkpointer component handles state persistence. The system defaults to in-memory snapshots but can switch to a Postgres persistent backend. Additionally, the reading node projects critical state to file checkpoints and MongoDB progress collections. This approach ensures fault tolerance during long-document processing. If a process crash or LLM call timeout occurs, the system prioritizes LangGraph checkpoint recovery. When in-memory snapshots are unavailable, it attempts file checkpoint recovery.

####  3.3.2 LLM-Driven ReAct Cognitive Loop

The Agent’s cognitive process on paragraphs is not composed of four fixed code nodes but rather an LLM-driven multi-round ReAct tool loop. The system prompt treats Read, Search, Verify, and Construct as workflow constraints. The LLM first identifies reusable knowledge objects in the current paragraph, then invokes retrieval tools to confirm whether identical or similar entities already exist in the KG. Based on tool-returned results, it decides to create, update, merge, delete, review, or defer each candidate, and finally submits graph operations through structured tool calls. This design inherits the reasoning–action–observation closed loop from the ReAct framework [Yao et al., 2023] while constraining the action space to safe tools required for KG construction.

Read. The system receives input comprising the current paragraph ccurrc_{\text{curr}}, adjacent context, current schema definition Σ\Sigma, and recent observation records. The LLM identifies durable knowledge objects such as methods, algorithms, datasets, metrics, systems, theories, problems, formal definitions, and named entities, while explicitly ignoring table of contents, formulaic language, transient narratives, and PDF parsing noise. If the paragraph contains no reusable knowledge, the LLM may directly return chunk_complete with a summary, without performing write operations.

Search. The LLM prioritizes using search_kg to check existing entities for each candidate concept, invokes browse_context when necessary to view adjacent paragraphs, or calls explore_fusion for combined vector and graph fusion retrieval. The primary goal of this phase is to prevent duplicate entity creation while providing necessary evidence for cross-paragraph relations.

Verify. After reading tool-returned results, the LLM makes decisions for each candidate: create an entity or relation when evidence is clear and no duplicates exist; update when existing objects need supplementation; merge when retrieval results indicate duplicates; call mark_for_review when uncertain; and call create_todo when current context is insufficient. Entity quality gatekeeping modules perform hard filtering at write-tool time, intercepting obvious sentence fragments, code, mathematical formulas, table-of-contents headings, OCR artifacts, or PDF garbage.

Construct. The LLM first submits entity operations, then relation operations dependent on those entities. Each tool result is backfilled as a role=tool message into the dialogue context; if a write fails, the LLM may initiate a search, change to an update, create a todo, or abandon the candidate based on failure information. The loop ends when the LLM returns valid paragraph-completion JSON, after which the outer LangGraph advances to the next paragraph.

Algorithm 1 ReAct Tool Decision Reference Flow

0: Current paragraph cc, toolset 𝒯\mathcal{T}, existing graph GtG_{t}, vector index MtM_{t}

0: Action sequence AA

1: A←∅A\leftarrow\emptyset

2: V←LLM.ReadDurableConcepts​(c)V\leftarrow\mathrm{LLM.ReadDurableConcepts}(c)

3: for v∈Vv\in V do

4: C←search_kg(v.name,fuzzy)∪explore_fusion(v.name)C\leftarrow\mathrm{search\\_kg}(v.\mathrm{name},\mathrm{fuzzy})\cup\mathrm{explore\\_fusion}(v.\mathrm{name})

5: if synonymous entity exists in CC then

6: A←A∪merge​_​entity/update​_​entityA\leftarrow A\cup\mathrm{merge\\_entity/update\\_entity}

7: else if vv has sufficient evidence and passes quality gate then

8: A←A∪create​_​entityA\leftarrow A\cup\mathrm{create\\_entity}

9: else

10: A←A∪mark​_​for​_​review/create​_​todoA\leftarrow A\cup\mathrm{mark\\_for\\_review/create\\_todo}

11: end if

12: end for

13: E←LLM.ProposeRelations​(c,V,C)E\leftarrow\mathrm{LLM.ProposeRelations}(c,V,C)

14: for e=(h​e​a​d,r​e​l,t​a​i​l)∈Ee=(head,rel,tail)\in E do

15: if h​e​a​dhead and t​a​i​ltail are resolved and evidence sufficient then

16: A←A∪create​_​relation/update​_​relationA\leftarrow A\cup\mathrm{create\\_relation/update\\_relation}

17: else

18: A←A∪create​_​todo/mark​_​for​_​reviewA\leftarrow A\cup\mathrm{create\\_todo/mark\\_for\\_review}

19: end if

20: end for

21: return AA

The above flow describes the reference pattern by which the LLM executes knowledge extraction and decision-making within the ReAct loop. In actual execution, the model autonomously determines the tool-calling sequence (including call order, parameters, and whether to skip certain steps) according to cognitive constraints in the system prompt, rather than executing line by line along fixed branches. This flow is embedded in the ReAct multi-round tool loop described in Section 3.3.2, where each paragraph may generate multiple rounds of “reasoning–tool call–observation” iterations.

####  3.3.3 Reading Progress State Machine and Error Handling

Each text chunk progresses through four lifecycle states in the Reading Layer: PENDING (awaiting processing), READING (currently active), VERIFIED (processed by the ReAct loop), and ARCHIVED (archived). State transitions are driven by the Reading Layer: when the Agent begins processing a paragraph, its state transitions from PENDING to READING. After the ReAct loop concludes and the LLM returns summary text, the state transitions to VERIFIED. Progress projection then records the ARCHIVED state in checkpoints. This state machine primarily serves recovery, observability, and scheduling purposes, not representing strong transactional commits across all underlying stores.

Error handling uses a two-category classification: Transient Errors (LLM API timeouts, rate limits, vector database connection interruptions) are automatically handled by retry decorators and LLM fallback clients, remaining transparent to the ReAct main loop. Permanent Errors (schema violations, entity ID conflicts, unrecoverable tool-call failures) cause the Agent to enter an explicit error handling branch, writing error information to the AgentState error history field. The PromptAssembler injects prompts in the next cycle to guide the Agent toward corrective action.

####  3.3.4 Knowledge Operation Toolset

The Tool Layer bridges the Agent and underlying storage systems, encapsulating complex KG operations and converting them into type-safe atomic tools. Each tool category uses JSON Schema to strictly constrain input parameter structure and output format; the Agent must comply with this contract when invoking tools within the ReAct loop.

Read tools include four tools serving evidence collection and context awareness phases. read_paragraph receives paragraph index, document identifier, and reading purpose description, returning the paragraph’s complete text, current index, and total paragraph count. search_kg receives a query keyword, search type (entity, relation, or fuzzy), and return limit, returning a structured list of matching entities or relations. browse_context browses local context with mode controlling scope (local for adjacent paragraphs, kg_neighbors for KG neighbor nodes, document_overview for document structure overview), returning a list of context snippets. explore_fusion performs joint retrieval across vector store and KG, with mode selecting fusion strategy (parallel retrieval then fusion, vector-first then graph expansion, graph-first then vector supplementation), returning an RRF-fused ranked candidate list.

Create tools include three tools supporting incremental graph object writing. batch_kg_operations executes combined search, create, update, merge, and delete operations in a single tool call to reduce tool-call round trips, used as the KG construction backend in Section 4 experiments. create_entity includes name, entity type, description, aliases, properties, source chunk binding, supporting evidence text, and certainty level. It returns a unique assigned entity ID or reuses an existing entity. create_relation uses head and tail entity names to designate endpoints, with relation type specification and evidence attributes similar to entity creation. create_todo creates deferred processing tasks with types including disambiguate, verify, attribute completion, or follow-up.

Update, merge, and delete tools support incremental graph correction and version evolution. update_entity receives target entity name, attribute update dictionary, update reason, and source chunk ID. update_relation accepts relation ID, evidence, confidence, and source chunk for relation evidence supplementation. merge_entity and merge_relation perform entity or relation resolution operations, recording merge basis and migrating evidence. delete_relation performs soft deletion to maintain historical version traceability. delete_entity performs hard deletion of entities.

Review, deferred task, and progress tools include mark_for_review, create_todo, and get_progress. When the LLM cannot determine facts, entity boundaries, or relation directions, it creates review items via mark_for_review to prevent uncertain knowledge from contaminating the graph. create_todo records cross-paragraph pending items. get_progress returns current paragraph position, entity/relation counts, merge count, todo count, and review queue length.

Table 1: Core Semantics of the RAGA Toolset Tool Name |  Category |  Core Parameters |  Return Value |  Phase  
---|---|---|---|---  
read_paragraph |  Read |  paragraph_idx, doc_id, purpose |  paragraph text, status |  Read  
browse_context |  Read |  query, mode, radius, chunk_id |  context snippets |  Search  
search_kg |  Search |  query, search_type, limit |  matched entities/relations |  Search  
explore_fusion |  Search |  query, mode, top_k |  fused retrieval results |  Search  
create_entity |  Create |  name, entity_type, evidence, certainty |  entity id / reused |  Construct  
create_relation |  Create |  head, relation_type, tail, evidence |  relation id |  Construct  
batch_kg_ops |  Batch |  searches, creates, updates, merges, deletes |  operation counters |  R/S/V/C  
update_entity |  Update |  entity_name, updates, reason |  update status |  V/C  
update_relation |  Update |  relation_id, evidence, confidence |  update status |  V/C  
merge_entity |  Merge |  target_name, source_name |  merge status |  V/C  
merge_relation |  Merge |  target_id, source_id |  merge status |  V/C  
delete_entity |  Delete |  entity_name, reason |  deletion status |  V/C  
delete_relation |  Delete |  relation_id, reason, soft |  soft deletion status |  V/C  
mark_for_review |  Review |  subject, reason, priority |  review item |  Verify  
create_todo |  Deferred |  task, todo_type, related_entity |  todo item |  Verify  
get_progress |  Progress |  none |  progress statistics |  R/S  
  
####  3.3.5 System Prompt Engineering

The Agent’s cognitive behavior is guided by system prompts dynamically assembled by the PromptAssembler module and injected into the LLM context before each react_loop iteration. Prompt engineering follows three principles: instruction layering, constraint explicitness, and context locality.

Explicit instructions for the cognitive loop. The system prompt describes four cognitive constraints in structured paragraphs: READ requires the LLM to identify durable knowledge objects while ignoring low-value chunks. SEARCH requires invoking search_kg, browse_context, or explore_fusion before creation. VERIFY requires the LLM to choose among create, update, merge, review, or todo based on tool-returned results. CONSTRUCT requires entity operations before relation operations, with the system correcting subsequent actions based on tool feedback upon failure. This prompt preserves ReAct’s free decision-making capability [Yao et al., 2023] while compressing the LLM’s action space to compliant tool sets, reducing hallucinatory tool calls.

Entity quality gatekeeping rules. Quality gatekeeping is embedded in two locations: semantic constraints in the ReAct system prompt, and structural filters inside the create_entity tool. The former requires concise, well-formed entity names and excludes code, formulas, PDF garbage, and generic headings. The latter checks length upper bound, printable character ratio, garbled text markers, punctuation ratio, code/formula patterns, and table-of-contents heading exclusion. Together, pre-constraint and post-filtering reduce the probability of noisy nodes entering the graph. The eight heuristic rules, evaluated from low to high computational cost, cover: name length limit (60 Unicode characters), printable character ratio (70%+), sentence fragment detection, code keyword detection, mathematical formula detection, punctuation flooding detection, PDF garbled marker detection, and generic heading exclusion. Entities intercepted are relayed to the LLM with a prompt to keep observations, create todos, or abandon the candidate.

Cross-chunk reasoning prompt strategy. The system prompt directs the Agent to query the existing graph before creation and to invoke browse_context or create_todo when context is insufficient. Todo items are processed by priority between paragraphs through the handle_todos node, primarily handling disambiguation, verification, or attribute completion.

Schema-guided prompt injection. The <active_schema> block dynamically injects current schema definitions including domain labels, relation types, entity labels, and attribute constraints. bootstrap_schema initializes this block, and schema evolution triggers incremental updates. The LLM can follow domain specifications during extraction, reducing invalid relation generation.

Working memory context maintenance. PromptAssembler maintains transient summaries in dynamic context, including current paragraph processing progress, recently known entities, recent tool observations, todo queue summaries, and necessary schema cues. This block’s compact design ensures context awareness while avoiding historical information drowning during long-document processing.

####  3.3.6 Structure-Aware Chunking

The chunker 𝒞\mathcal{C}’s design affects knowledge extraction granularity and structural integrity. The system implements four chunking strategies. FIXED_SIZE segments documents by fixed character length with configurable chunk_size (default 800 characters). Simple and uniform but may truncate mid-sentence, disrupt semantic continuity, and fail to recognize document structure boundaries such as section headings. SEMANTIC uses sentence-ending punctuation (period, question mark, exclamation mark) as boundaries, finding the nearest sentence boundary within chunk_size constraints to reduce sentence-level truncation. Does not handle cross-sentence semantic paragraph boundaries. PARAGRAPH treats double newlines (\n\n) as boundaries, partitioning each logical paragraph as one chunk, preserving paragraph-level integrity. Overly long paragraphs may produce chunks exceeding the LLM context window, while overly short paragraphs such as list items become fragmented. STRUCTURAL (default) parses document hierarchical structure including heading levels, list nesting, and code block boundaries to generate semantically cohesive chunk sequences. It maintains a stack structure to track the current heading level, placing chunk boundaries preferentially at structural transition points (before level-2 headings, code block boundaries, table starts). For same-level paragraphs, secondary segmentation applies the chunk_size threshold with boundaries at sentence or paragraph ends.

####  3.3.7 Schema Auto-Discovery

On cold start, the system does not assume a fixed domain schema. Instead, a Schema Orchestrator analyzes the document corpus to induce domain-specific entity types, relation types, and attribute constraints, thereby improving cross-domain adaptability. The discovery flow proceeds through four phases: Domain Detection (Phase 0) extracts sample text from the document prefix and determines via embedding similarity and LLM whether an existing domain can be reused. On detection failure or timeout, a bootstrap domain is constructed for rapid startup. Schema Discovery (Phase 1) selects representative samples (default up to 3 documents, each with the first 2000 characters), forms a discovery prompt submitted to the LLM, requiring analysis of sample content to induce domain-specific relation types, entity label types, and attribute patterns, outputting parseable JSON. Schema Validation (Phase 2) checks relation naming for UPPER_SNAKE_CASE compliance, filters low quality scores, identifies semantic duplicates with existing relation types, and verifies domain/range resolvability. Failed items are removed, merged, or retained as candidates. Schema Activation (Phase 3) writes validated schemas to the schema graph for storage, forming configurations cacheable in Redis as the active session schema. Schema Evolution (Phase 4) handles new relation patterns not covered by the active schema during processing; the orchestrator first searches for semantically similar existing relations for reuse, and if none exist, the LLM generates new relation definitions registered in PROPOSED state pending subsequent validation or human review.

###  3.4 Memory Architecture

The Memory Layer employs heterogeneous storage where each tier complements others in storage format, access patterns, and lifecycle. The design references multiple memory systems theory from cognitive science, applying human long-term semantic memory, episodic memory, and working memory to engineering implementation [Anokhin et al., 2024].

Semantic Memory uses Neo4j graph database to store the KG’s ontological structure—the long-term representation of entities, relations, and their attributes. The storage schema adopts the property graph model: node labels distinguish entity types (Person, Organization, Concept), with node attributes including name, description, aliases, and provenance record references. Relation types encode WORKS_AT, MENTIONS, IS_A, etc., with relation attributes recording confidence and extraction timestamps. Access patterns support two query modes: exact lookup by node identifier (entity ID or name), and pattern matching using labels and attributes (finding entities of a type or neighbors via specific relations). Neo4j’s Cypher query engine provides native support for multi-hop graph expansion operations described in Section 3.6.

Episodic Memory uses MongoDB document database to store agent execution trajectories during document processing, progress projections, and provenance records. Records include fields such as session identifier, paragraph index, AgentState snapshot subsets, tool call summaries, error history, source chunks, and evidence text. Episodic memory supports trace queries by document, chunk, entity, relation, and operation type.

Working Memory is managed through LangGraph AgentState, Checkpointer, and Redis cache. AgentState holds current paragraph content, recent observations, todo task queues, tool usage records, and related statistics. Redis primarily caches SchemaProfile, short-term states in HyperNode chains, and supports cross-node reads with expiration cleanup. Redis cache uses prefixed key names and TTL policies.

Vector Memory uses Milvus vector database to store dense vector representations of chunks and entities, supporting approximate nearest neighbor search. Current collection schema distinguishes chunk and entity collections: chunk collections contain chunk_id, embedding, tenant_id, run_id, dataset, document_id for isolation and lookup; entity collections contain entity_id, embedding, isolation fields, entity name, entity type, and KG node ID. Vector dimensionality is controlled by EMBEDDING_DIM in run configuration. Access mode is primarily ANN search: given query vector 𝐪\mathbf{q}, Milvus returns top-k neighbors with distance scores, supporting filtering by tenant, run, dataset, and document for isolated retrieval during evaluation and production.

Four-Layer Memory Coordination. The four memory layers are provided to the Reading Layer through dependency injection. During a single react_loop iteration in the Reading Layer, the access sequence typically proceeds as: (1) loading current context through AgentState and recent episodic records; (2) searching for similar entities and evidence chunks via graph or fusion retrieval; (3) performing topological queries via Neo4j to verify relation validity; (4) write tools committing entities, relations, and provenance records to corresponding backends; (5) the synchronization layer supplementing vector indices for newly created entities or chunks and connecting HyperNode evidence bridges. Layers maintain loose coupling, with configurable dependencies allowing substitution of real backends or test doubles.

###  3.5 KG-Vector Synchronization

Maintaining consistency between the KG and vector index is a primary engineering challenge. Since Neo4j and Milvus are independent systems that cannot share transaction boundaries, the current engineering approach employs sequential writes with a failure compensation strategy: first writing Neo4j structural objects, then writing Milvus vector records, and finally writing back vector references in the graph. If vector writing fails, the system compensates by deleting or marking already-written graph objects.

The synchronization process comprises three phases. Phase 1 (Graph Write): The coordinator writes entities, relations, or HyperNodes into Neo4j with tenant, run, dataset, and document isolation fields. For chunk synchronization, the system first creates document nodes and HyperNodes, then establishes evidence bridge edges such as HAS_EVIDENCE, MENTIONS_ENTITY, and EVIDENCED_BY. Phase 2 (Vector Write): The coordinator invokes the embedding service to generate chunk or entity vectors and writes them into the corresponding Milvus collection. Vector records carry isolation fields consistent with Neo4j, plus chunk, entity, or HyperNode identifiers for subsequent lookup and filtering during fusion retrieval. Phase 3 (Reference Write-Back and Compensation): After successful vector writing, the system writes back the vector ID or KG node ID into Neo4j object attributes. If Milvus writing fails, the system attempts to delete or roll back created Neo4j objects and logs the synchronization failure; if reference write-back fails, the primary write result is preserved and an alert is recorded, with subsequent repair possible through a consistency check interface.

Algorithm 2 KG-Vector Sequential Synchronization with Compensation

0: Graph object xx, semantic store SS, vector store VV, embedding service EE

0: Synchronization status and object identifiers 

1: k​g​_​i​d←S.write​(x)kg\\_id\leftarrow S.\mathrm{write}(x)

2: 𝐞←E.embed(x.text)\mathbf{e}\leftarrow E.\mathrm{embed}(x.\mathrm{text})

3: v​e​c​_​i​d←NILvec\\_id\leftarrow\mathrm{NIL}

4: if 𝐞\mathbf{e} generation succeeded then

5: vec_id←V.insert(kg_id,𝐞,x.scope)vec\\_id\leftarrow V.\mathrm{insert}(kg\\_id,\mathbf{e},x.\mathrm{scope})

6: else

7: S.compensate​(k​g​_​i​d)S.\mathrm{compensate}(kg\\_id)

8: return (FAILED,k​g​_​i​d,∅)(\mathrm{FAILED},kg\\_id,\emptyset)

9: end if

10: if v​e​c​_​i​d=NILvec\\_id=\mathrm{NIL} then

11: S.compensate​(k​g​_​i​d)S.\mathrm{compensate}(kg\\_id)

12: return (FAILED,k​g​_​i​d,∅)(\mathrm{FAILED},kg\\_id,\emptyset)

13: end if

14: S.set​_​embedding​_​ref​(k​g​_​i​d,v​e​c​_​i​d)S.\mathrm{set\\_embedding\\_ref}(kg\\_id,vec\\_id)

15: return (SUCCESS,k​g​_​i​d,v​e​c​_​i​d)(\mathrm{SUCCESS},kg\\_id,vec\\_id)

Algorithm 2 characterizes the core-path engineering consistency strategy: sequential writes with observable compensation. This strategy is simple to implement with clear fault boundaries but does not provide strict distributed transaction guarantees; compensation failures rely on logs and consistency check interfaces for subsequent repair.

###  3.6 Three-Layer Fusion Retrieval

The Retrieval Layer implements a “vector recall →\rightarrow graph expansion →\rightarrow fusion ranking” three-step pipeline, synthesizing semantic similarity and structural relevance to return answer context in response to user queries. The system supports four retrieval modes:

  * •

vector: Only performs vector recall, without triggering graph expansion;

  * •

kg: Only performs Neo4j-based graph multi-hop expansion, without recalling vector candidates;

  * •

fusion: Parallel execution of vector recall and graph expansion, fusing both ranking streams via RRF;

  * •

deep (default): Based on LLM query analysis, forces chained graph navigation via HyperNode bridging, ultimately merging vector candidates with navigation results rather than using the RRF formula.




Below, the fusion mode’s three-step flow is used to illustrate the retrieval mechanism; deep mode specifics are described in the Section 4 experimental setup.

####  3.6.1 Vector Recall (Step 1)

Given a user query qq, the encoder Enc first generates a query vector 𝐪=Enc​(q)\mathbf{q}=\text{Enc}(q). Milvus performs ANN search returning a candidate set:

| 𝒞vec={(xi,svec​(i))}i=1k1,svec​(i)=11+‖𝐪−ℳt​(xi)‖2\mathcal{C}_{\text{vec}}=\\{(x_{i},s_{\text{vec}}(i))\\}_{i=1}^{k_{1}},\quad s_{\text{vec}}(i)=\frac{1}{1+\|\mathbf{q}-\mathcal{M}_{t}(x_{i})\|_{2}} |  | (3)  
---|---|---|---  
  
where svec​(i)s_{\text{vec}}(i) is the normalized similarity score and k1k_{1} is the recall count (default 100). Searches may attach scalar filter conditions, e.g., restricting node_type=ENTITY\text{node\\_type}=\text{ENTITY} to exclude relation vectors.

####  3.6.2 KG Multi-Hop Expansion (Step 2)

For each entity xix_{i} in the vector recall result 𝒞vec\mathcal{C}_{\text{vec}}, a breadth-first search (BFS) in Neo4j fetches its hh-hop neighbors:

| 𝒩h​(xi)={y:dist𝒢t​(xi,y)≤h}\mathcal{N}_{h}(x_{i})=\\{y:\text{dist}_{\mathcal{G}_{t}}(x_{i},y)\leq h\\} |  | (4)  
---|---|---|---  
  
where dist𝒢t\text{dist}_{\mathcal{G}_{t}} is the shortest-path distance in the graph (measured in relation hops). Neighbor nodes pass through lightweight semantic filtering before joining the candidate set, forming the expanded candidate set 𝒞kg\mathcal{C}_{\text{kg}}. This multi-hop expansion mechanism is inspired by the “retrieval-reasoning” paradigm in Think-on-Graph [Sun et al., 2024] and RoG [Luo et al., 2024], but replaces LLM-driven reasoning path generation with deterministic BFS to reduce latency and ensure reproducibility.

####  3.6.3 RRF Fusion Ranking (Step 3)

The vector candidate set and graph candidate set each produce independent rankings. The system applies Reciprocal Rank Fusion (RRF) to fuse both ranking streams [Guo et al., 2025]. Let rankvec​(x)\text{rank}_{\text{vec}}(x) be the rank of xx in the vector candidate set (∞\infty if missed), and rankkg​(x)\text{rank}_{\text{kg}}(x) be the rank in the graph candidate set (∞\infty if missed). The fusion score is defined as:

| RRF​(x)=∑m=1M1k+rankm​(x)\text{RRF}(x)=\sum_{m=1}^{M}\frac{1}{k+\text{rank}_{m}(x)} |  | (5)  
---|---|---|---  
  
where M=2M=2 denotes the two source streams (vector retrieval and graph retrieval), rankm​(x)\text{rank}_{m}(x) is the rank of object xx in the mm-th candidate stream (∞\infty if missed), and kk is a smoothing constant, empirically set to 60. The final output is the fused candidate set 𝒞fused\mathcal{C}_{\text{fused}} sorted by descending RRF​(x)\text{RRF}(x). RRF requires no trainable parameters and is insensitive to the scoring scales of the two ranking streams, proving robust and effective in heterogeneous retrieval fusion scenarios [Guo et al., 2025, Sarmah et al., 2024].

####  3.6.4 Timeout Fallback and Result Truncation

After fusion ranking, the system truncates the candidate set according to the user-requested top-k count and returns it. The KG expansion layer has an independent timeout budget. When Neo4j multi-hop expansion times out or fails, the system records an alert and degrades to using only the vector candidate RRF result. This strategy prevents graph layer anomalies from blocking the QA main chain. In Section 4, Evidence F1, Evidence Precision/Recall, and Recall@K are used for external evaluation of retrieval quality.

##  4 Experiments

Due to computational resource constraints, this section evaluates RAGA on a small-batch subset of the QASPER scientific literature QA dataset, with a focus on qualitative comparison of retrieval modes, chunking strategies, and batch processing tools. We compare against published zero-shot and supervised methods including LED, GraphRAG, and a No-KG Control baseline. All reported metrics should be interpreted as preliminary evidence under this limited evaluation scale.

###  4.1 Experimental Setup

####  4.1.1 Evaluation Dataset

This experiment uses QASPER (Question Answering over Scientific Papers Evidence Retrieval) as the evaluation benchmark [Dasigi et al., 2021]. Published by Dasigi et al. at NAACL 2021, QASPER is one of the most representative evaluation benchmarks in scientific literature QA.

QASPER contains 1,585 NLP papers covering top venues such as ACL, EMNLP, and NAACL, with 5,049 annotated questions. All questions were posed by NLP practitioners who only read paper titles and abstracts, simulating real research scenarios where readers ask in-depth questions about paper content. QASPER questions span extractive, abstractive, yes/no, and unanswerable types, many requiring multi-paragraph evidence support [Dasigi et al., 2021].

This experiment selects a small-batch paper subset from the QASPER test set for evaluation. All experiments run under a zero-shot protocol in a local environment: no training set labels are used for supervised fine-tuning; each paper independently executes the complete KG construction, vector index building, and Agent inference pipeline.

To examine the impact of chunking strategies on retrieval performance, three chunking configurations are set: C1200 (chunk_size=1200, chunk_overlap=120), C1500 (chunk_size=1500, chunk_overlap=120), and C6000 (chunk_size=6000, chunk_overlap=120). C1200 and C1500 employ medium-granularity chunking; C6000 simulates large-granularity chunking to examine retrieval coverage sensitivity to chunk granularity.

####  4.1.2 Comparison Baselines

Multi-level comparison baselines are established, covering published supervised and zero-shot methods:

Published Supervised Methods: (1) LED-base / LED-large w/ evidence scaffold: Supervised fine-tuned models based on the Longformer-Encoder-Decoder architecture, fine-tuned using QASPER training set gold evidence paragraphs as supervision signals, serving as the official QASPER dataset baseline [Dasigi et al., 2021]. (2) Human lower bound: QASPER official inter-annotator agreement lower bound, reflecting the F1 level of human annotators on the same questions. This value is not an upper bound but a soft lower bound of multi-annotator agreement [Dasigi et al., 2021].

Zero-Shot Retrieval Baselines: (3) BM25: Classic probabilistic retrieval model computing relevance scores based on term frequency and document length normalization [Robertson et al., 1995]. Uses lightweight lexical retrieval with top-8 paragraphs as evidence. (4) TF-IDF: Classic sparse vector retrieval method, computing similarity via term frequency and inverse document frequency weighting [Sparck Jones, 1972]. (5) FAISS Vector RAG: Dense retrieval baseline based on FAISS vector index, performing ANN search on query and chunk embeddings without using KG signals. (6) No-KG Control: Naive vector retrieval using the same LLM backend (deepseek-v4-flash). Builds paragraph-level FAISS index within each paper, retrieves top-8 paragraphs via question vector, then has the LLM generate answers. Does not construct a KG or use Milvus/Neo4j/MongoDB. This baseline quantifies the net contribution of KG. (7) GraphRAG [Edge et al., 2024]: Microsoft’s official GraphRAG implementation (microsoft/graphrag), using the standard pipeline of “text chunking →\rightarrow entity/relation extraction →\rightarrow graph construction →\rightarrow Leiden community detection →\rightarrow community summarization →\rightarrow Local Search QA.” GraphRAG uses the same LLM backend and embedding model as RAGA.

Our Method: (8) RAGA: The complete method proposed in this paper, supporting multiple retrieval modes—Vector, KG, Fusion, and Deep, as defined in Section 3.6. Uses batch KG operation tools as the KG construction backend.

####  4.1.3 Evaluation Metrics

This experiment adopts the QASPER official evaluation protocol [Dasigi et al., 2021], performing quantitative evaluation on both answer quality and evidence quality dimensions, using the max-over-annotators scoring method (taking the best score among multiple annotators for each question).

Answer Quality. Answer F1 measures the token-level overlap between generated and reference answers. Let the reference answer be Ag​o​l​dA_{gold} and the model-generated answer be Ap​r​e​dA_{pred}, tokenized to Tg​o​l​dT_{gold} and Tp​r​e​dT_{pred} respectively:

| Pa​n​s=|Tp​r​e​d∩Tg​o​l​d||Tp​r​e​d|,Ra​n​s=|Tp​r​e​d∩Tg​o​l​d||Tg​o​l​d|P_{ans}=\frac{|T_{pred}\cap T_{gold}|}{|T_{pred}|},\quad R_{ans}=\frac{|T_{pred}\cap T_{gold}|}{|T_{gold}|} |  | (6)  
---|---|---|---  
  
| Answer F1=1N​∑i=1Nmaxj⁡2⋅Pa​n​s(i,j)⋅Ra​n​s(i,j)Pa​n​s(i,j)+Ra​n​s(i,j)\text{Answer F1}=\frac{1}{N}\sum_{i=1}^{N}\max_{j}\frac{2\cdot P_{ans}^{(i,j)}\cdot R_{ans}^{(i,j)}}{P_{ans}^{(i,j)}+R_{ans}^{(i,j)}} |  | (7)  
---|---|---|---  
  
Evidence Quality. Evidence F1 measures overlap between the paragraph set returned by the retrieval system and the human-annotated evidence paragraph set:

| Pe​v​i=|Ep​r​e​d∩Eg​o​l​d||Ep​r​e​d|,Re​v​i=|Ep​r​e​d∩Eg​o​l​d||Eg​o​l​d|P_{evi}=\frac{|E_{pred}\cap E_{gold}|}{|E_{pred}|},\quad R_{evi}=\frac{|E_{pred}\cap E_{gold}|}{|E_{gold}|} |  | (8)  
---|---|---|---  
  
| Evidence F1=1N​∑i=1Nmaxj⁡2⋅Pe​v​i(i,j)⋅Re​v​i(i,j)Pe​v​i(i,j)+Re​v​i(i,j)\text{Evidence F1}=\frac{1}{N}\sum_{i=1}^{N}\max_{j}\frac{2\cdot P_{evi}^{(i,j)}\cdot R_{evi}^{(i,j)}}{P_{evi}^{(i,j)}+R_{evi}^{(i,j)}} |  | (9)  
---|---|---|---  
  
Additionally, Retrieved Evidence F1 measures overlap between the raw retrieval-stage results and annotated evidence (without evidence re-ranking post-processing), reflecting the retrieval system’s raw recall capability.

####  4.1.4 Experimental Environment

Core configuration: LLM inference uses deepseek-v4-flash under zero-shot protocol without supervised fine-tuning. Text generation uses temperature=0.7; tool calling and JSON generation use temperature=0.3. Embedding model is qwen3-embedding-8b with output dimension 2,048. KG storage uses Neo4j; vector index uses Milvus; metadata and raw document content use MongoDB; schema caching and short-term state use Redis. RAGA Fusion mode uses RRF (k=60k=60) to fuse vector and graph candidates. Timeout configuration: LLM_TIMEOUT=120, LLM_TOTAL_TIMEOUT=240; each paper runs with an independent run_id and \--require-clean-backends to prevent cross-document contamination.

###  4.2 Main Experimental Results

####  4.2.1 Retrieval Mode Comparison

Table 2 shows RAGA’s retrieval and QA performance across four retrieval modes under the C1200 configuration with the new batch processing tool.

Table 2: Retrieval Mode Comparison (C1200, New Batch Tool) Retrieval Mode | Answer F1 | Evidence F1 | Retrieved Evidence F1  
---|---|---|---  
KG (Graph-Only) | 0.526 | 0.339 | 0.094  
Vector (Vector-Only) | 0.587 | 0.363 | 0.196  
Deep (HyperNode Bridge) | 0.523 | 0.295 | 0.199  
Fusion (Graph+Vector) | 0.615 | 0.411 | 0.188  
  
Table 2 shows: (1) Fusion mode achieves the best results on both Answer F1 (0.615) and Evidence F1 (0.411), primarily because RRF fusion ranking effectively combines vector retrieval’s semantic coverage advantage with graph retrieval’s structural precision. (2) Pure vector retrieval serves as a stable baseline with Answer F1=0.587 and Evidence F1=0.363, only 0.028 below Fusion in Answer F1, indicating that in this configuration pure semantic retrieval already possesses strong answer-localization capability; Fusion’s significance mainly manifests in improved evidence precision. (3) KG mode has lower Answer F1 (0.526) but still competitive Evidence F1 (0.339). The new batch tool extracts entities more conservatively, making graph signals leaner and more effective. (4) Deep mode leads marginally in Retrieved Evidence F1 (0.199), indicating HyperNode bridging’s advantage in precise evidence localization, but its Answer F1 (0.523) and Evidence F1 (0.295) fall below Fusion, primarily because graph navigation results used as retrieval context lack sufficient coverage to support answer generation. Overall, Fusion mode achieves the best balance between answer quality and evidence precision through RRF fusion of graph and vector signals.

####  4.2.2 Comparison with Published Work

Table 3 reports Answer F1 and Evidence F1 for RAGA against published QASPER baselines, the No-KG Control, and GraphRAG. RAGA Fusion (C1200) achieves Answer F1=61.5 and Evidence F1=41.1 under zero-shot protocol.

Table 3: Comparison with Published Work and Ablation Baselines Method | Training | Answer F1 | Evidence F1  
---|---|---|---  
LED-large w/ evidence scaffold [Dasigi et al., 2021] | Supervised | <20.0<20.0 | –  
LED-base w/ evidence scaffold [Dasigi et al., 2021] | Supervised | 33.6 | 39.4  
No-KG Control (FAISS) | Zero-shot | 55.4 | 35.9  
GraphRAG [Edge et al., 2024] | Zero-shot | 31.6 | 47.2  
RAGA KG (C1500) | Zero-shot | 60.5 | 38.8  
RAGA Fusion (C1200) | Zero-shot | 61.5 | 41.1  
Human lower bound [Dasigi et al., 2021] | – | 60.9 | 71.6  
  
Table 3 shows the following.

(1) RAGA Fusion achieves Answer F1=61.5 under zero-shot protocol on this evaluation sample, approaching the human inter-annotator agreement level of 60.9. No-KG Control achieves Answer F1=55.4, substantially higher than LED-base’s 33.6, indicating that the LLM’s reading comprehension capability is the dominant performance factor. RAGA Fusion gains an additional 6.1pp through KG integration. GraphRAG’s Answer F1 of 31.6 is markedly lower than No-KG Control, indicating that its community summarization approach, while beneficial for global information aggregation, loses fine-grained evidence details and reduces answer generation quality.

(2) In evidence retrieval, GraphRAG achieves the highest Evidence F1 of 47.2, compared to RAGA Fusion’s 41.1 and No-KG Control’s 35.9. However, GraphRAG’s high evidence recall does not translate into high answer quality (Answer F1 only 31.6), revealing a gap between evidence recall and answer generation: community summaries lose the precise wording of original text, whereas RAGA maintains original-chunk evidence fidelity through Agent-driven direct chunk access.

(3) RAGA Fusion’s Answer F1 approaches human inter-annotator agreement and its Evidence F1 exceeds No-KG Control by 5.2pp, with a stronger evidence-to-answer quality ratio than GraphRAG. Given the limited evaluation sample, these comparisons provide preliminary evidence for the framework’s effectiveness.

####  4.2.3 Chunk Size Ablation

Table 4 shows Answer F1 comparison across three chunking configurations and four retrieval modes (all using the new batch tool).

Table 4: Chunk Size Ablation Results (Answer F1) Retrieval Mode | C1200 | C1500 | C6000  
---|---|---|---  
KG | 0.526 | 0.605 | 0.468  
Vector | 0.587 | 0.602 | 0.364  
Deep | 0.523 | 0.511 | 0.315  
Fusion | 0.615 | 0.560 | 0.346  
  
Results show: (1) C1200 is the overall optimal configuration—Vector and Fusion achieve best Answer F1 (0.587 and 0.615 respectively) under C1200, indicating that medium-granularity chunking is beneficial for answer generation across different retrieval strategies. (2) KG mode performs prominently under C1500 (Answer F1=0.605 vs. 0.526 under C1200), likely because slightly larger chunks make entity-to-chunk mappings more concentrated, enabling graph traversal to cover more relevant information within a single chunk. (3) C6000 causes significant performance degradation across all modes—Fusion drops from 0.615 to 0.346, Vector from 0.587 to 0.364. Large-granularity chunking causes each chunk to contain excessive information, making retrieval-returned paragraph granularity insufficient to precisely match QASPER evidence paragraph requirements, reducing both evidence localization precision and answer quality. (4) Vector mode shows stability between C1500 (0.602) and C1200 (0.587), suggesting semantic retrieval has lower sensitivity to chunk granularity compared to KG and Deep modes that depend on entity-to-chunk mappings.

####  4.2.4 Batch Tool Comparison

Table 5 compares the old tool with the batch processing tool under C1200 configuration.

Table 5: Old vs. New Batch KG Construction Tool Comparison (C1200) Retrieval Mode | Old Tool Answer F1 | New Batch Tool Answer F1 | Change  
---|---|---|---  
KG | 0.650 | 0.526 |  −-0.124  
Vector | 0.582 | 0.587 | +0.005  
Deep | 0.543 | 0.523 |  −-0.020  
Fusion | 0.526 | 0.615 | +0.089  
  
The comparison shows: (1) Fusion mode improves notably—the new batch tool raises Fusion Answer F1 from 0.526 to 0.615 (+17%), as leaner entity extraction reduces graph noise and improves RRF fusion signal quality. (2) KG retrieval declines from 0.650 to 0.526; the old tool’s over-extraction increased coverage at the cost of noise, while the new tool favors signal purity. Since Fusion is the recommended mode, this trade-off is acceptable. (3) Construction efficiency improves—on the tested paper (41 chunks, 4 questions), the batch tool reduces KG construction time from 77 to 54 minutes (30% reduction). (4) Vector retrieval is unaffected—it relies solely on chunk embeddings and is stable across tool versions (0.587 vs. 0.582).

###  4.3 Discussion and Limitations

Design implications. Fusion mode is recommended as the default retrieval strategy, performing best across configurations. C1200 (chunk_size=1200) is the recommended default, robust across retrieval modes. The new batch tool improves Fusion quality by 17% and reduces construction time by 30%, supporting a “quality over quantity” KG construction philosophy.

Limitations. Given the limited evaluation sample, all results should be interpreted as preliminary. The following limitations are acknowledged: (1) Evaluation scale: This experiment was conducted on a small-batch subset of QASPER; larger-scale evaluation and cross-dataset validation remain for future work. (2) Retrieval recall: Retrieved Evidence F1 remains low (0.188), indicating room for improvement in initial retrieval recall. (3) Computational efficiency: End-to-end per-paper processing incurs high time cost, though the one-time construction cost can be amortized across more questions. Future optimization directions include parallel construction, incremental update mechanisms, and adaptive retrieval path selection. (4) Chunking generalizability: Optimal chunking granularity may vary across domains and document types.

##  5 Conclusion

This paper proposed RAGA, an LLM-based autonomous knowledge graph construction and retrieval fusion framework that addresses three structural deficiencies of existing KG construction methods: cross-chunk semantic relation loss, entity redundancy and insufficient disambiguation, and construction process uninterpretability.

RAGA’s core contributions span four dimensions. (1) Autonomous toolset: 16 atomic tools enabling full KG lifecycle management with batch operations reducing construction time by 30% over per-operation tools. (2) Cognitive loop: The Read–Search–Verify–Construct constraint embedded in a ReAct tool loop, supported by a reading-progress state machine for fault-tolerant long-document processing. (3) KG-vector synchronization: A sequential-write-with-compensation strategy maintaining repairable consistency between symbolic (Neo4j) and vector (Milvus) layers, enabling RRF fusion retrieval. (4) Evidence-anchored verification: Structured provenance records linking knowledge entries to original text chunks with source, evidence, operation type, and confidence metadata.

On a QASPER subset, RAGA Fusion achieved Answer F1=0.615 and Evidence F1=0.411 under C1200, outperforming GraphRAG (Answer F1=31.6%) and No-KG Control (Answer F1=55.4%). KG fusion contributed +2.8pp to Answer F1 and +4.8pp to Evidence F1. GraphRAG achieved the highest Evidence F1 (47.2%) but its community summarization sacrificed fine-grained text fidelity; RAGA preserved original-chunk evidence fidelity through Agent-driven direct chunk access, better translating evidence recall into answer quality. Results are directional given the limited evaluation sample.

Future work includes: (1) full test set evaluation with rigorous component-wise ablation; (2) cross-modal KG construction incorporating figures, tables, and pseudocode; (3) incremental online learning for continuous KG evolution; (4) human-in-the-loop feedback optimization for domain-expert-guided quality improvement.

## References

  * P. Anokhin, N. Semenov, A. Sorokin, D. Evseev, A. Kravchenko, M. Burtsev, and E. Burnaev (2024) AriGraph: learning knowledge graph world models with episodic memory for LLM agents.  External Links: 2407.04363 Cited by: §1, §1, §2.1, §2.5, §3.2, §3.4. 
  * A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi (2024) Self-RAG: learning to retrieve, generate, and critique through self-reflection.  In The Twelfth International Conference on Learning Representations (ICLR),  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2310.11511) Cited by: §2.3. 
  * P. Dasigi, K. Lo, I. Beltagy, A. Cohan, N. A. Smith, and M. Gardner (2021) A dataset of information-seeking questions and answers anchored in research papers.  In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies,  pp. 4599–4610.  External Links: [Document](https://dx.doi.org/10.18653/v1/2021.naacl-main.365) Cited by: §1, §1, §4.1.1, §4.1.1, §4.1.2, §4.1.3, Table 3, Table 3, Table 3. 
  * G. Dong, J. Jin, X. Li, Y. Zhu, Z. Dou, and J. Wen (2025) RAG-critic: leveraging automated critic-guided agentic workflow for retrieval augmented generation.  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),  pp. 3551–3578.  External Links: [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.179) Cited by: §2.4. 
  * D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, D. Metropolitansky, R. O. Ness, and J. Larson (2024) From local to global: a graph RAG approach to query-focused summarization.  External Links: 2404.16130 Cited by: §1, §1, §2.2, §2.5, §3.1, §4.1.2, Table 3. 
  * Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang, and H. Wang (2023) Retrieval-augmented generation for large language models: a survey.  External Links: 2312.10997 Cited by: §1. 
  * Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang (2025) LightRAG: simple and fast retrieval-augmented generation.  In Findings of the Association for Computational Linguistics: EMNLP 2025,  pp. 10746–10761.  External Links: [Document](https://dx.doi.org/10.18653/v1/2025.findings-emnlp.568) Cited by: §1, §2.2, §2.5, §3.2, §3.6.3, §3.6.3. 
  * B. J. Gutierrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su (2024) HippoRAG: neurobiologically inspired long-term memory for large language models.  In Advances in Neural Information Processing Systems 38 (NeurIPS 2024),  pp. 59532–59569.  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2405.14831) Cited by: §2.4. 
  * X. He, Y. Tian, Y. Sun, N. V. Chawla, T. Laurent, Y. LeCun, X. Bresson, and B. Hooi (2024) G-retriever: retrieval-augmented generation for textual graph understanding and question answering.  In Advances in Neural Information Processing Systems 37 (NeurIPS 2024),  Vol. 37,  pp. 132876–132907.  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2402.07630) Cited by: §2.4. 
  * Y. Huo, Y. Lu, Z. Zhang, H. Chen, and Y. Lin (2026) AtomMem: learnable dynamic agentic memory with atomic memory operation.  External Links: 2601.08323 Cited by: §2.3. 
  * S. Jeong, J. Baek, S. Cho, S. J. Hwang, and J. Park (2024) Adaptive-RAG: learning to adapt retrieval-augmented large language models through question complexity.  In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),  pp. 7036–7050.  External Links: [Document](https://dx.doi.org/10.18653/v1/2024.naacl-long.389) Cited by: §2.3. 
  * J. Jiang, K. Zhou, W. X. Zhao, Y. Song, C. Zhu, H. Zhu, and J. Wen (2025) KG-Agent: an efficient autonomous agent framework for complex reasoning over knowledge graph.  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),  pp. 9505–9523.  External Links: [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.468) Cited by: §1, §2.1, §2.5. 
  * Y. Lairgi, L. Moncla, R. Cazabet, K. Benabdeslem, and P. Cléau (2024) iText2KG: incremental knowledge graphs construction using large language models.  In Web Information Systems Engineering — WISE 2024,  pp. 214–229.  External Links: [Document](https://dx.doi.org/10.1007/978-981-96-0573-6%5F16) Cited by: §1, §1, §1, §2.2, §2.5. 
  * M. Lee, Q. Zhu, C. Mavromatis, Z. Han, S. Adeshina, V. N. Ioannidis, H. Rangwala, and C. Faloutsos (2025) HybGRAG: hybrid retrieval-augmented generation on textual and relational knowledge bases.  In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),  pp. 879–893.  External Links: [Document](https://dx.doi.org/10.18653/v1/2025.acl-long.43) Cited by: §2.4, §2.5. 
  * P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela (2020) Retrieval-augmented generation for knowledge-intensive NLP tasks.  In Advances in Neural Information Processing Systems 33 (NeurIPS 2020),  pp. 9459–9474.  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2005.11401) Cited by: §1. 
  * Z. Li, X. Chen, H. Yu, H. Lin, Y. Lu, Q. Tang, F. Huang, X. Han, L. Sun, and Y. Li (2025) StructRAG: boosting knowledge intensive reasoning of LLMs via inference-time hybrid information structurization.  In The Thirteenth International Conference on Learning Representations (ICLR),  Cited by: §2.2, §2.5. 
  * L. Liang, Z. Bo, Z. Gui, Z. Zhu, L. Zhong, P. Zhao, M. Sun, Z. Zhang, J. Zhou, W. Chen, W. Zhang, and H. Chen (2025) KAG: boosting LLMs in professional domains via knowledge augmented generation.  In Companion Proceedings of the ACM on Web Conference 2025 (WWW 2025 Companion),  pp. 334–343.  External Links: [Document](https://dx.doi.org/10.1145/3701716.3715240) Cited by: §1, §2.2, §2.5. 
  * H. Luo, H. E, G. Chen, Q. Lin, Y. Guo, F. Xu, Z. Kuang, M. Song, X. Wu, Y. Zhu, and L. A. Tuan (2025) Graph-R1: towards agentic graphRAG framework via end-to-end reinforcement learning.  External Links: 2507.21892 Cited by: §2.4, §2.5. 
  * L. Luo, Y. Li, G. Haffari, and S. Pan (2024) Reasoning on graphs: faithful and interpretable large language model reasoning.  In The Twelfth International Conference on Learning Representations (ICLR),  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2310.01061) Cited by: §2.4, §3.6.2. 
  * C. Lv, H. Chang, Y. Guo, S. Tao, and S. Zhou (2026) All-Mem: agentic lifelong memory via dynamic topology evolution.  External Links: 2603.19595 Cited by: §2.3. 
  * C. Mavromatis and G. Karypis (2025) GNN-RAG: graph neural retrieval for efficient large language model reasoning on knowledge graphs.  In Findings of the Association for Computational Linguistics: ACL 2025,  pp. 16682–16699.  External Links: [Document](https://dx.doi.org/10.18653/v1/2025.findings-acl.856) Cited by: §2.4. 
  * Y. Ning and H. Liu (2024) UrbanKGent: a unified large language model agent framework for urban knowledge graph construction.  In Advances in Neural Information Processing Systems 37 (NeurIPS 2024),  Vol. 37,  pp. 123127–123154.  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2402.06861) Cited by: §2.1. 
  * C. Packer, V. Fang, S. G. Patil, K. Lin, S. Wooders, I. Stoica, and J. E. Gonzalez (2023) MemGPT: towards LLMs as operating systems.  External Links: 2310.08560 Cited by: §2.1. 
  * S. E. Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, and M. Gatford (1995) Okapi at TREC-3.  In Overview of the Third Text REtrieval Conference (TREC-3),  NIST Special Publication,  pp. 109–126.  Cited by: §4.1.2. 
  * B. Sarmah, D. Mehta, B. Hall, R. Rao, S. Patel, and S. Pasquali (2024) HybridRAG: integrating knowledge graphs and vector retrieval augmented generation for efficient information extraction.  In Proceedings of the 5th ACM International Conference on AI in Finance (ICAIF 2024),  pp. 608–616.  External Links: [Document](https://dx.doi.org/10.1145/3677052.3698671) Cited by: §1, §2.4, §3.2, §3.6.3. 
  * P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C. D. Manning (2024) RAPTOR: recursive abstractive processing for tree-organized retrieval.  In The Twelfth International Conference on Learning Representations (ICLR),  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2401.18059) Cited by: §1, §2.2. 
  * K. Sparck Jones (1972) A statistical interpretation of term specificity and its application in retrieval.  Journal of Documentation 28 (1),  pp. 11–21.  External Links: [Document](https://dx.doi.org/10.1108/eb026526) Cited by: §4.1.2. 
  * J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y. Gong, L. M. Ni, H. Shum, and J. Guo (2024) Think-on-graph: deep and responsible reasoning of large language model on knowledge graph.  In The Twelfth International Conference on Learning Representations (ICLR),  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2307.07697) Cited by: §2.4, §3.6.2. 
  * H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal (2023) Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.  In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),  pp. 10014–10037.  External Links: [Document](https://dx.doi.org/10.18653/v1/2023.acl-long.557) Cited by: §2.3. 
  * J. Wang, J. Fu, R. Wang, L. Song, and J. Bian (2025) PIKE-RAG: specialized knowledge and rationale augmented generation.  Note: Microsoft Research Asia External Links: 2501.11551 Cited by: §2.2, §2.5. 
  * W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang (2025) A-MEM: agentic memory for LLM agents.  External Links: 2502.12110 Cited by: §2.3. 
  * S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao (2023) ReAct: synergizing reasoning and acting in language models.  In The Eleventh International Conference on Learning Representations (ICLR),  External Links: [Document](https://dx.doi.org/10.48550/arXiv.2210.03629) Cited by: §1, §2.3, §3.2, §3.3.2, §3.3.5. 
  * B. Zhang and H. Soh (2024) Extract, define, canonicalize: an LLM-based framework for knowledge graph construction.  In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,  pp. 9820–9836.  External Links: [Document](https://dx.doi.org/10.18653/v1/2024.emnlp-main.548) Cited by: §1, §1, §1, §2.2, §2.5. 
  * W. Zhong, L. Guo, Q. Gao, H. Ye, and Y. Wang (2024) MemoryBank: enhancing large language models with long-term memory.  In Proceedings of the AAAI Conference on Artificial Intelligence,  Vol. 38,  pp. 19724–19731.  External Links: [Document](https://dx.doi.org/10.1609/aaai.v38i17.29946) Cited by: §2.1. 
  * Y. Zhu, X. Wang, J. Chen, S. Qiao, Y. Ou, Y. Yao, S. Deng, H. Chen, and N. Zhang (2024) LLMs for knowledge graph construction and reasoning: recent capabilities and future opportunities.  World Wide Web 27 (5),  pp. 58.  External Links: [Document](https://dx.doi.org/10.1007/s11280-024-01297-w) Cited by: §1. 



[◄](/html/2605.17071) [](/) [Feeling  
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion  
report](/log/2605.17072) [Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2605.17072) [View original  
on arXiv](https://arxiv.org/abs/2605.17072)[►](/html/2605.17073)

[](javascript:toggleColorScheme\(\) "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Jun 6 01:08:36 2026 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
