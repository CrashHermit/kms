# EvoRAG: Making Knowledge Graph-based RAG Automatically Evolve through Feedback-driven Backpropagation

Zhenbo Fu1, Yuanzhe Zhang1, Qiange Wang1, Hao Yuan1, Yuehao Xu1,   
Enze Yi2, Yanfeng Zhang1, Ge Yu1 Affiliation: 1School of Computer Science and Engineering, Northeastern University, Shenyang 110819, China; , 2Northeast Electric Power Research Institute of State Grid Liaoning Electric Power Supply Co, Ltd, Shenyang, Liaoning, 110004, China email:[fuzhenbo,zhangyz,yuanhao@stumail.neu.edu.cn,thecoldxyh@gmail.com](mailto:fuzhenbo,zhangyz,yuanhao@stumail.neu.edu.cn,thecoldxyh@gmail.com) email:[wangqg,zhangyf, yuge@mail.neu.edu.cn, yienze˙cs@163.com](mailto:wangqg,zhangyf,%20yuge@mail.neu.edu.cn,%20yienze%CB%99cs@163.com)

###### Abstract.

Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) has emerged as a promising paradigm for enhancing LLM reasoning by retrieving multi-hop paths from KGs. However, existing KG-RAG frameworks often underperform in real-world scenarios because the pre-captured knowledge dependencies are not tailored to the downstream task or its evolving requirements. These frameworks struggle to adapt to task-specific requirements and lack mechanisms to filter low-contribution knowledge during generation. We observe that feedback on generated responses offers effective supervision for improving KG quality, as it directly reflects user expectations and provides insights into the correctness and usefulness of the output. However, a key challenge lies in effectively linking response-level feedback to triplet-level contribution evaluation and knowledge updates in the KG.

In this work, we propose EvoRAG, a self-evolving KG-RAG framework that leverages the feedback over generated responses to continuously refine the KG and enhance reasoning accuracy. EvoRAG introduces a feedback-driven backpropagation mechanism that attributes feedback to retrieved paths by measuring their utility for response and propagates this utility back to individual triplets, supporting fine-grained KG refinements towards more adaptive and accurate reasoning. Through EvoRAG, we establish a closed loop that couples feedback, LLM, and graph data, continuously enhancing the performance and robustness in real-world scenarios. Experimental results show that EvoRAG improves reasoning accuracy by 7.34%7.34\% over state-of-the-art KG-RAG frameworks. The source code has been made available at <https://github.com/iDC-NEU/EvoRAG>.

##  1\. Introduction

Figure 1.  Comparison of conventional KG-RAG and EvoRAG. EvoRAG introduces a backpropagation mechanism that propagates response-level feedback to individual triplets, enabling continuous KG refinements. 

Retrieval-Augmented Generation (RAG) (RAGSurvey_arxiv23; RAGNLP_NIPS20; RAGAISurvey_arxiv24) empowers Large Language Models (LLMs) to improve response quality by leveraging external knowledge, and has been widely adopted in various domains (RAGForLegalQA_ICCR24; RAGForHealth_ACL24; RAGForFinacial_ICAIF23). Among RAG paradigms, Knowledge Graph-based RAG (KG-RAG) (GraphRAGSurvey_arxiv25; GraphRAGsurvey_arxiv_2024; GraphRAGBench_arxiv25) has gained increasing attention for its ability to transform the textual corpus into structured knowledge graphs (KGs) (GraphRAGBench_arxiv25), capturing rich semantic information and entity-level relations. Given a query qq, the core idea of KG-RAG is to retrieve a relevant knowledge subgraph (KSG) from the KG and feed it together with qq to the LLM for response generation. The retrieved KSG is typically organized into a sequence of reasoning paths, ordered chains of triplets that capture multi-hop semantic connections, where each triplet follows the form ⟨​h​e​a​d​e​n​t​i​t​y,r​e​l​a​t​i​o​n,t​a​i​l​e​n​t​i​t​y​⟩\textlangle head\ entity,relation,tail\ entity\textrangle.

Although KG-RAG frameworks have demonstrated promising results by leveraging the structural information, they often underperform in real-world scenarios due to a mismatch between the pre-captured dependencies in KG and the task-specific requirements (GraphRAGSurvey_arxiv25; GraphRAGBench_arxiv25). This mismatch primarily stems from two structural limitations of KG-RAG frameworks: insufficient adaptability to downstream reasoning tasks and limited dynamicity in handling knowledge freshness and reliability. These factors are overlooked in recent KG refinement studies (KGRefine_survey_2016; mlkgr_survey_2024), which mainly focus on detecting factual errors and semantic inconsistencies. Firstly, adaptability refers to the ability to organize and retrieve knowledge in a manner that supports real-time queries. Naively applying RAG with existing KG frameworks often retrieves information that is semantically correct but contributes little to the query. For example, they may retrieve irrelevant triplets or overlook long-range dependencies required for complex reasoning (RuleRAG_arxiv24; Knowgpt_NIPS24; Gretriever_NIPS24). This leads to ineffective reasoning and poor knowledge utilization. Secondly, dynamicity refers to the ability to detect and eliminate outdated or erroneous information. Existing KG-RAG frameworks are generally built on static KGs. They lack mechanisms for continuously detecting and eliminating invalid knowledge, such as outdated relations or KSGs that are no longer required by the downstream applications (TCROF_arxiv25; EraRAG_arxiv25), leading to degraded reasoning performance. These limitations reveal a data management problem: how to continuously refine KG to improve adaptability and support dynamic evolution.

We observe that feedback derived from generated contexts, such as signals on correctness, consistency, and user satisfaction, provides a valuable data source for continuously improving the KG. In LLM-based systems, such feedback is commonly obtained by leveraging LLMs to evaluate generated responses (selfrefine2023nips; reflexion2023nips; critiquellm2024acl; G_eval2023acl; ye2024beyond; trivedi2024self; peng2023check; RLAIFvsRLHF_ICML24). Similar feedback may also come from human judgments (RLTHF_ICLM; RRHF_NIPS24; KGT_arxiv24; KGV_IPM25; cleangraph_arxiv24) or comparisons with available ground truth (ACE2025arxiv; han2025self; wang2025reinforcement). Such feedback directly reflects reasoning effectiveness and can serve as a guidance signal for continuously refining the KG to improve task performance.

However, a mismatch in granularity between feedback and KG refinement makes such integration difficult: feedback assesses overall response quality over multiple retrieved paths, whereas KG refinement operates at the level of individual triplets, leading to a mismatch as feedback reflects the combined effect of multiple interacting triplets rather than localized signals for each discrete and reusable knowledge units. As a result, feedback needs to be correctly attributed to reusable triplets across multiple reasoning paths and queries. Unlike traditional machine learning systems that explicitly construct a computation graph during the forward pass and rely on backpropagation to propagate gradients to model parameters, feedback-driven KG refinement lacks an inherent mechanism to propagate feedback to underlying triplets, requiring carefully designed algorithms to establish this association.

In this work, we present EvoRAG, a self-evolving KG-RAG framework that utilizes feedback to continuously refine the KG, enhancing the accuracy and relevance of the reasoning process, as illustrated in Figure 1. To realize this, EvoRAG introduces a feedback-driven backpropagation mechanism that establishes a connection between response-level feedback and triplet-level knowledge updates through two key stages. Firstly, EvoRAG attributes the feedback to individual reasoning paths by measuring their utility, i.e., how much each path contributes to the response reflected in the feedback. Reasoning paths are used as intermediates because they directly guide the LLM in generating the response, while also providing a natural bridge to the underlying triplets from which they are constructed. Secondly, EvoRAG propagates the path utility to update the contribution score of each involved triplet. These scores are maintained as learnable parameters and continuously updated across interactions, allowing EvoRAG to prioritize task-relevant knowledge. Through EvoRAG, we construct a closed-loop mechanism that tightly couples feedback, LLM reasoning, and triplet-level KG updates, continuously improving its accuracy in real-world queries.

In summary, our primary contributions are as follows:

  * •

We propose a feedback-driven backpropagation mechanism that utilizes feedback to drive the evolution of the KG-RAG framework, enhancing adaptability and improving overall accuracy.

  * •

We introduce an effective mechanism that resolves the challenge of mapping response-level feedback to triplet-level updates, enabling fine-grained KG refinement and retrieval improvement.

  * •

We develop EvoRAG, a self-evolving KG-RAG framework that adapts to dynamic tasks and requirements. Experimental results demonstrate that EvoRAG improves accuracy by 7.34%7.34\% compared to the state-of-the-art KG-RAG frameworks.




##  2\. Preliminary

###  2.1. KG-RAG

RAG.  Large language models (LLMs) (GPT4_2023; ChatGPT; llama_arxiv2023; Gemma_arxiv2024; GLM_arxiv2022) have recently demonstrated remarkable potential in handling complex tasks (LLM_QA_TACL23; LLM_QA_EMNLP2019; LLM_Trans_ICML23; LLM_code_ICML23; LLM_code_ICSE2023; LLM_recommendation_arxiv2023). However, they are susceptible to hallucinations when generating answers for queries that require information beyond their knowledge (zhao2023beyond; ge2023openagi; llm_hallucination_arxiv23; LLM_hallucination_ACMCS2023; llm_outdated_arxiv24; LLM_domain_arxiv2023; LLM_domain_arxiv2024; LLM_domain_nature2023). To address these limitations, Retrieval-Augmented Generation (RAG) (RAGSurvey_arxiv23; RAG_NIPS20; GraphRAGBench_arxiv25; GraphRAGSurvey_arxiv25; neutronrag_demo_sigmodemo25) has emerged as a promising approach, enhancing LLMs with external knowledge by retrieving relevant information, improving the accuracy and relevance of the generated responses.

Among existing RAG methods, KG-RAG (MicrosoftGraphRAG_arxiv_2024; lightrag_arxiv_2024; kag_arxiv24; PathRAG_arxiv25; DALK_EMNLP24; RuleRAG_arxiv24; GNNRAG_arxiv_2024; Graph_CoT_ACL24; Gretriever_NIPS24; Structrag_arxiv24; GraphReader_EMNLP24; StructureGuided_arxiv24; yuan2025depcache) has attracted considerable attention due to its ability to leverage structured knowledge (wang2022neutronstar; fu2025neutrontask; chen2023neutronstream). KG-RAG organizes external corpus in a KG, enabling a more comprehensive understanding by effectively leveraging interconnections between pieces of knowledge, making it well-suited for complex, reasoning-intensive tasks.

KG.  Knowledge Graph (KG) encodes a wide range of knowledge in the form of triplets: G={(e1,r,e2)|e1,e2∈E,r∈R}G=\\{(e_{1},r,e_{2})|e_{1},e_{2}\in E,r\in R\\}, where EE denotes the set of nodes (entities) and RR denotes the set of directed edges that signify relations between those entities. The triplet (e1,r,e2)(e_{1},r,e_{2}) is the fundamental building unit of knowledge in a KG, where e1e_{1} denotes the subject entity, rr denotes the predicate, and e2e_{2} denotes the object entity.

Reasoning path refers to the sequence of entities and relations traversed from a starting entity: L=e0→r1e1→r2⋯→rieiL=e_{0}\xrightarrow{r_{1}}e_{1}\xrightarrow{r_{2}}\cdots\xrightarrow{r_{i}}e_{i}, where e0e_{0} is the starting entity, ei∈Ee_{i}\in E is the ii-th entity, and ri∈Rr_{i}\in R is the ii-th relation. Such paths reveal higher-order dependencies and provide interpretable evidence for downstream tasks, e.g., link prediction, question answering, and recommendation.

Figure 2. The overall workflow of KG-RAG. 

The workflow of KG-RAG.  As illustrated in Figure 2, the KG-RAG frameworks typically consist of three key components: KG construction, KSG retrieval, and response generation.

(➊) KG construction. KG construction refers to the process of mining structured knowledge from large-scale external corpora, which can be performed using specialized information extraction tools, such as OpenIE (OpenIE_ICCL18), or by leveraging LLMs. Specifically, the external corpus is divided into multiple text chunks, each of which undergoes entity recognition and relation extraction to construct triplets, such as (T​o​m,W​o​r​k​s​A​t,G​o​o​g​l​e)(Tom,WorksAt,Google). These extracted triplets are then inserted into a graph database to form a complete KG, which in turn supports subsequent online RAG services.

(➋) KSG retrieval. KSG retrieval refers to the process of extracting relevant reasoning paths from a KG for a given query, which typically consists of three steps: Query Entity Recognition, Subgraph Extraction, and Path Retrieval (lego-graphrag_vldb25). Firstly, Query Entity Recognition operates on the input query to identify query entities and align them with corresponding nodes in the KG. This step typically employs a semantic similarity model to extract entities whose surface forms or contextual embeddings closely match the query expressions. Secondly, Subgraph Extraction operates on the entire KG to extract a subgraph related to the query entities. The goal of this step is to reduce the search space and improve the effectiveness and efficiency of retrieval. Thirdly, Path Retrieval operates on the extracted subgraphs to collect the top-MM reasoning paths starting from the query entities and connecting them to potential target answers (e.g., one of the reasoning paths for the entity “E​v​aEva” is E​v​a→H​a​s​B​r​o​t​h​e​rB​o​b→W​o​r​k​s​A​tZ​e​l​oEva\xrightarrow{HasBrother}Bob\xrightarrow{WorksAt}Zelo). Both Subgraph Extraction and Path Retrieval can be implemented using graph algorithms (e.g., neighbor expansion, personalized PageRank) or semantically enhanced models (e.g., embedding models, LLMs). By integrating these approaches, existing KG-RAG implementations can be broadly covered within a unified formulation (kag_arxiv24; PathRAG_arxiv25; DALK_EMNLP24; RuleRAG_arxiv24; GNNRAG_arxiv_2024; Graph_CoT_ACL24; Gretriever_NIPS24; Structrag_arxiv24; GraphReader_EMNLP24; StructureGuided_arxiv24; TOG_ICLR24; Knowgpt_NIPS24; GNNRAG_ISCEIC24).

(➌) Response generation. Response generation refers to the process of producing the response conditioned on both the user query and the reasoning paths. Specifically, the retrieved paths are combined with the query to form an augmented prompt, which is subsequently fed into an LLM for generation. Unlike traditional RAG, which ranks individual text segments by semantic similarity, KG-RAG links entities and relations to compose multi-hop evidence spanning multiple passages, which enables cross-passage inference and yields more coherent and better grounded responses.

###  2.2. Motivation

Figure 3. Proportion of error types in KRAG and EvoRAG. #IF, #LP, and #OI represent three error types: irrelevant facts, long reasoning paths, and outdated information, respectively. 

KG-RAG has shown great potential in enhancing generation quality by leveraging the structural information of external knowledge graphs. However, in real-world applications, there exists a notable gap between the predefined dependencies encoded in the KG and the adaptive and dynamic requirements of user queries. This gap results in two fundamental limitations: lack of adaptability to downstream reasoning tasks and lack of dynamicity in maintaining knowledge validity. Firstly, the underlying KGs are inherently query-agnostic, as they are constructed offline without considering the specific query intent. This often leads to two common issues: (1) Irrelevant facts introduce noise. As shown in Figure 2, when the user asks “Where does Eva’s brother work?”, the retriever returns unrelated facts such as (B​o​b,L​i​v​e​s​I​n,N​i​v​a)(Bob,LivesIn,Niva), which do not contribute to the answer. (2) Failing to capture long-range dependencies. For example, answering that Eva’s brother works at Google requires a 33-hop reasoning path: E​v​a→H​a​s​B​r​o​t​h​e​rB​o​b→C​o​l​l​e​a​g​u​eT​o​m→W​o​r​k​s​A​tG​o​o​g​l​eEva\xrightarrow{HasBrother}Bob\xrightarrow{Colleague}Tom\xrightarrow{WorksAt}Google. However, pre-defined retrieval strategy (e.g., 22-hop retrieval) fails to capture such multi-hop dependencies, causing critical information to be omitted. Secondly, existing KG-RAG systems typically rely on static KGs and lack mechanisms for timely updates (kag_arxiv24; PathRAG_arxiv25; DALK_EMNLP24; GNNRAG_arxiv_2024; Gretriever_NIPS24). As a result, outdated or invalid information may persist and mislead the LLM. For example, the KG may still contain (B​o​b,W​o​r​k​s​A​t,Z​e​l​o)(Bob,WorksAt,Zelo), even though the company ceased operations in 20202020.

In real-world deployments, a KG-RAG system typically serves a large number of concurrent users who interact with a shared knowledge graph. Multiple users often issue queries focused on the same regions of the graph, such as popular entities, ongoing events, or trending topics. Consequently, the system frequently receives highly similar or even identical queries within the same local subgraph. This concentrated access pattern amplifies the aforementioned issues: irrelevant facts and missed long-range dependencies repeatedly affect many users, and outdated triplets continue to mislead multiple responses.

To systematically quantify the impact of the above limitations on KG-RAG reasoning, we analyze the erroneous responses in the RGB (RGB_benchmark_AAAI24) and MultiHop (Multihoprag_arxiv24) datasets. We categorize the errors into four types: (1) irrelevant facts (IF), (2) long reasoning paths (LP), (3) outdated information (OI), and (4) other errors caused by missing knowledge in the KG or hallucinations by the LLM (not the focus of this work). To provide a baseline, we adopt a state-of-the-art instance from prior work (lego-graphrag_vldb25), referred to as KRAG. Firstly, KRAG constructs the retrieval subgraph by expanding from the query entities to include all entities within two hops. Then, a semantic model (BAAI/bge-large-en-v1.5 (bge_arxiv23)) is employed to select reasoning paths within the subgraph that are most similar to the query as the retried results. As shown in Figure 3, IF, LP, and OI together account for over half of all errors, with average proportions of 17.9%17.9\%, 20.7%20.7\%, and 11.9%11.9\%, respectively, making them the primary bottlenecks that fundamentally limit the effectiveness of KG-RAG in real-world reasoning tasks. To address these critical limitations, we propose EvoRAG, which incorporates a feedback-driven backpropagation mechanism to explicitly target and mitigate these error sources, resulting in substantial improvements in reasoning accuracy. In the following sections, we present the design and implementation of EvoRAG in detail.

##  3\. System Overview

Figure 4. EvoRAG system overview. 

We propose EvoRAG, a self-evolving KG-RAG framework that continuously improves reasoning effectiveness through fine-grained knowledge refinement and retrieval optimization. As illustrated in Figure 4, the system operates in an iterative loop where users continuously submit queries. For each query, EvoRAG retrieves multiple reasoning paths from the KG as contextual knowledge, which are then fed into an LLM to generate responses. To further enhance reasoning quality, EvoRAG introduces a feedback-driven backpropagation mechanism (see Section 4) that transforms coarse-grained feedback into fine-grained supervision over individual data triplets. This mechanism enables the system to iteratively refine both the underlying knowledge and the retrieval process. We implement this mechanism in two stages.

In the first stage, we propose a feedback-based path evaluation module that propagates response-level feedback to the retrieved reasoning paths. The output is a path-level utility score that quantifies its overall quality and the unique contribution to the generated response. Instead of directly evaluating individual triplets, we adopt reasoning paths as the intermediate unit. This design is motivated by the observation that triplets do not function in isolation. During inference, responses are generated through multi-hop reasoning over paths composed of multiple triplets, where the contribution of each triplet is inherently context-dependent, shaped by other triplets within the same path as well as the semantics of the input query. Consequently, directly assigning utility to individual triplets may lead to misalignment, as a triplet that is beneficial in one path can be misleading in another. Reasoning paths therefore provide a more appropriate intermediate abstraction, bridging high-level feedback and fine-grained knowledge units.

In the second stage, we propose a gradient backpropagation module that further propagates path-level utility to the constituent triplets. Specifically, the module computes the gradients of the utility expectation over paths with respect to path utility and distributes them to triplets based on their roles within each path. To enable consistent supervision across queries, we associate each triplet with a learnable contribution score, which serves as the foundation for fine-grained reasoning supervision and allows the KG to gradually evolve toward task-specific relevance. This score is defined as a dimensionless and non-negative coefficient that estimates the expected influence of a triplet on response quality. In practice, all scores are initialized to 100100 and dynamically updated over time based on accumulated feedback signals.

With the updated contribution scores, EvoRAG provides feedback-guided KG management, enabling both KG evolution and retrieval optimization (see Section 5). Specifically, EvoRAG implements relation-centric KG evolution, which includes relation fusion and suppression. Relation fusion introduces shortcut edges between frequently co-occurring high-utility triplets to enhance long-range reasoning effectiveness, while relation suppression downweights low-utility triplets, ensuring structural coherence and knowledge quality. Then, EvoRAG adopts a hybrid priority-based retrieval strategy that integrates semantic relevance with feedback-derived contribution scores. This hybrid mechanism balances query-dependent relevance and query-independent reliability, guiding retrieval toward both contextually appropriate and historically effective knowledge.

##  4\. Feedback-driven Backpropagation

In this section, we first introduce the sources of feedback, followed by the feedback-based path evaluation for computing path-level utility, and finally introduce the gradient backpropagation mechanism that propagates this utility to individual triplets to update the contribution scores.

###  4.1. Feedback

Overview of feedback and its sources.  In interactive systems, responses are often accompanied by signals that indicate their quality with respect to task objectives, such as factual correctness, reasoning soundness, or relevance to user intent. We refer to such signals as feedback. In modern LLM-based systems, feedback is commonly obtained by using LLMs to evaluate generated responses and produce quality assessments (selfrefine2023nips; reflexion2023nips; critiquellm2024acl; G_eval2023acl; ye2024beyond; trivedi2024self; peng2023check; RLAIFvsRLHF_ICML24). More generally, similar feedback may also arise from other sources. For example, human assessments may appear as ratings, binary judgments, or survey responses (RLTHF_ICLM; RRHF_NIPS24; KGT_arxiv24; KGV_IPM25; cleangraph_arxiv24); and when ground truth is available, comparison with reference answers provides an objective measure of correctness (ACE2025arxiv; han2025self; wang2025reinforcement).

In this work, we primarily rely on feedback derived from LLM-based evaluation, following prior studies that use LLMs to assess response quality (selfrefine2023nips; reflexion2023nips; critiquellm2024acl; G_eval2023acl). Specifically, given the ground-truth answers available in our datasets, the evaluator LLM is prompted to assess each generated response and assign a scalar satisfaction score from 1 to 5, providing a reliable and objective assessment. A score of 1 indicates complete dissatisfaction (e.g., irrelevant or factually incorrect responses), while a score of 5 indicates full satisfaction (e.g., responses that are correct, relevant, and well aligned with the query intent), with intermediate scores (22–44) reflecting varying levels of partial adequacy. To demonstrate the generality of EvoRAG, we further evaluate it using other kinds of feedback, including human judgments and ground-truth based feedback using F1 scores, as shown in Section 6.4.

Challenge.  Although unified feedback scoring provides a consistent evaluation signal, a fundamental challenge lies in its granularity mismatch. Feedback is only available at the response level, offering coarse-grained assessments of overall quality, whereas effective KG refinement requires fine-grained identification of the specific triplets that influence reasoning and retrieval. This mismatch limits precise knowledge adjustment and constrains overall system performance.

To address this challenge, our framework translates coarse response-level feedback into fine-grained supervision through a two-step decomposition process. Feedback is first transformed into path-level utility via a path evaluation module, which captures the contribution of entire reasoning trajectories. The utility is then propagated to constituent triplets through gradient-based backpropagation process to update their contribution scores.

###  4.2. Feedback-based Path Evaluation

Figure 5. Feedback-driven Backpropagation. We first use an evaluation function to compute the utility of each reasoning path based on feedback, and then transform it into gradients that are propagated to individual triplets.

To enable effective learning from response-level feedback, we define a utility evaluation function ff that assigns utility to each reasoning path by analyzing its contribution to the generated response. The evaluation considers the semantics of the input query, the retrieved paths, the generated response, and the feedback score. Specifically, when feedback indicates a correct response, ff rewards paths that semantically support the answer; when the response is incorrect, it attributes negative utility to paths that may have misled generation. Instead of assigning binary judgments, the evaluation function produces continuous utility in a bounded range (i.e., [−1,1][-1,1]), enabling nuanced assessment of each path’s contribution. We formalize the path utility evaluation as:

(1) |  | U​{L}=f​(q,Rq,L,F​S),\displaystyle\small U\\{L\\}=f(q,R_{q},L,FS), |   
---|---|---|---  
  
where qq is the user query, RqR_{q} is the response, LL is the set of retrieved reasoning paths, and F​SFS is the feedback score. In practice, ff can be instantiated by a range of semantic models. In this work, we use an LLM as a constrained path-level scoring function. Given the retrieved paths, the query, the generated response, and the feedback score, the LLM assigns a utility score to each path based on its relevance, following the LLM-as-a-judge paradigm (TextGrad_nature25; LLM_as_a_judge_NIPS23). This process is analogous to RAG guided by the external knowledge: the LLM evaluates paths using the provided context instead of producing new responses, and is therefore minimally affected by hallucination.

To capture the multi-faceted nature of path quality under feedback, we decompose path evaluation into three complementary dimensions. Supportiveness measures whether a path provides evidence that supports or undermines the generated response, conditioned on response correctness as determined by feedback. Higher Supportiveness increases path utility, while lower values decrease it. Fidelity and Conflict serve as auxiliary metrics for cross-validation. Fidelity measures how much a path contributes to the response, with higher values meaning greater contribution. Conflict measures whether a path contradicts the response, with lower values meaning less contradiction. The LLM assigns a score in [−1,1][-1,1] to each dimension based on the query, retrieved paths, generated response, and feedback. Importantly, a path’s utility is updated based on Supportiveness only when Fidelity is high and Conflict is low; otherwise, the update is suppressed. This design prevents erroneous utility updates caused by spurious grounding or contradictory evidence, ensuring reliable path-level attribution.

###  4.3. Gradient Backpropagation

Once the utility of each reasoning path is evaluated, the next step is to propagate this utility back to individual triplets, refining their contribution scores so that future retrievals are more likely to select high-utility paths. To derive backpropagation, we first need to introduce how the contribution score influences the path selection strategy during forward computation.

####  4.3.1. Forward Computation

In the forward retrieval phase, each query retrieves multiple reasoning paths from the KG, which is composed of a sequence of triplets. Each triplet tt is associated with a selection probability (P​(t)∈[0,1]P(t)\in[0,1]), which reflects its semantic relevance to the query and the contribution score. Formally, the P​(t)P(t) can be defined as:

(2) |  | P​(t)=(1−α)​Sr​(t)+α​Sc​(t),\displaystyle\small P(t)=(1-\alpha)S_{r}(t)+\alpha S_{c}(t), |   
---|---|---|---  
  
where Sr​(t)∈(0,1]S_{r}(t)\in(0,1] measures the semantic similarity between the triplet tt and the query qq, providing a query-specific score that aligns path selection with the current query. In contrast, Sc​(t)∈[0,1]S_{c}(t)\in[0,1] is the learnable contribution score by normalization, which accumulates feedback across multiple queries and rounds, thereby decoupling triplet assessment from any single query. The parameter α∈[0,1]\alpha\in[0,1] is a learnable trade-off between SrS_{r} and ScS_{c}, allowing the system to balance the current query’s guidance with accumulated knowledge from previous feedback.

Based on these probabilities, the priority of a reasoning path (Li∈LL_{i}\in L) can be defined as follows:

(3) |  | P​(Li)=exp⁡(1|Li|​∑t∈Lilog⁡P​(t))∑Lj∈Lexp⁡(1|Lj|​∑t∈Ljlog⁡P​(t)),\small P(L_{i})=\frac{\exp\Big(\tfrac{1}{|L_{i}|}\sum_{t\in L_{i}}\log P(t)\Big)}{\sum_{L_{j}\in L}\exp\Big(\tfrac{1}{|L_{j}|}\sum_{t\in L_{j}}\log P(t)\Big)}, |   
---|---|---|---  
  
where LiL_{i} denotes a reasoning path and |Li||L_{i}| is its hop length (i.e., the number of triplets it contains). The length normalization takes the log-average of triplet probabilities to eliminate the inherent bias of the multiplicative formulation toward shorter paths, enabling fair comparison across reasoning paths of different hop lengths.

The Formula 3 not only ensures relevance to the current query, but also enables generalization to unseen queries. When a new query arrives, the similarity score SrS_{r} is freshly calculated at query time, ensuring specificity to the user’s intent and filtering out irrelevant knowledge, where the contribution score ScS_{c} transfers the accumulated feedback on triplet quality to guide the selection of the reasoning paths. In this way, the model adapts to unseen queries by letting SrS_{r} provide query-time specificity and ScS_{c} provide cross-query transferability, yielding path priorities that balance immediate relevance with long-term utility.

Loss function.  We formulate an objective that maximizes the expected utility of the retrieved reasoning paths. Formally, the loss function is defined as follows:

(4) |  | ℒ\displaystyle\small\mathcal{L} | =−log⁡(𝔼​[U​(L)])=−log⁡(∑Li∈LP​(Li)​U​(Li)+12),\displaystyle=-\log(\mathbb{E}[U(L)])=-\log\left(\sum_{L_{i}\in L}P(L_{i})\frac{U(L_{i})+1}{2}\right), |   
---|---|---|---|---  
  
where LL denotes the set of all retrieved reasoning paths. Optimizing expected utility encourages the system to adapt this preference with observed effectiveness, guiding future retrievals toward reasoning paths that consistently yield high-quality responses.

####  4.3.2. Backward Computation

We compute the gradient of the loss ℒ\mathcal{L} with respect to the contribution score Sc​(t)S_{c}(t) as follows:

(5) |  | ∇Sc​(t)ℒ=−α2​𝔼​[U​(L)]​∑t∈Li∏g∈LiP​(g)P​(t)​Vi,\displaystyle\small\nabla_{S_{c}(t)}\mathcal{L}=-\frac{\alpha}{2\mathbb{E}[U(L)]}\sum_{{t\in L_{i}}}\frac{\prod_{g\in L_{i}}P(g)}{P(t)}V_{i}, |   
---|---|---|---  
(6) |  | Vi=P​(Li)|Li|​(U​(Li)−∑Lj∈LP​(Lj)​U​(Lj)),\displaystyle V_{i}=\frac{P(L_{i})}{|L_{i}|}\left(U(L_{i})-\sum_{L_{j}\in L}P(L_{j})U(L_{j})\right), |   
  
where ViV_{i} denotes the deviation of a path’s utility from the expected utility across all paths. The gradient ∇Sc​(t)ℒ\nabla_{S_{c}(t)}\mathcal{L} quantifies how the triplet tt contributes to reasoning paths with above- or below-average utility. If tt frequently appears in high-utility paths (i.e., paths where U​(Li)>∑Lj∈LP​(Lj)​U​(Lj)U(L_{i})>\sum_{L_{j}\in L}P(L_{j})U(L_{j})), the gradient becomes negative, reducing the loss and thereby increasing the contribution score Sc​(t)S_{c}(t). Conversely, if tt tends to appear in low-utility paths, the score will be reduced accordingly. Similarly, the gradient of α\alpha can be computed as follows:

(7) |  | ∇αℒ=∑t∈TSr​(t)−Sc​(t)2​𝔼​[U​(L)]​∑t∈Li∏g∈LiP​(g)P​(t)​Vi,\displaystyle\small\nabla_{\alpha}\mathcal{L}=\sum_{t\in T}\frac{S_{r}(t)-S_{c}(t)}{2\mathbb{E}[U(L)]}\sum_{{t\in L_{i}}}\frac{\prod_{g\in L_{i}}P(g)}{P(t)}V_{i}, |   
---|---|---|---  
  
where TT denotes the set of all triplets. Finally, we apply gradient descent to update both the contribution score and the parameter α\alpha:

(8) |  | Sc​(t)\displaystyle\small S_{c}(t) | =Sc​(t)−η​∇Sc​(t)ℒ,\displaystyle=S_{c}(t)-\eta\nabla_{S_{c}(t)}\mathcal{L}, |   
---|---|---|---|---  
(9) |  | α\displaystyle\alpha | =α−η​∇αℒ,\displaystyle=\alpha-\eta\nabla_{\alpha}\mathcal{L}, |   
  
where η\eta is the learning rate. This optimization process ensures that triplets associated with helpful reasoning paths become more prominent, while those consistently tied to unproductive reasoning are gradually suppressed.

####  4.3.3. Accuracy Analysis

To evaluate the effectiveness of our feedback-driven backpropagation, we analyze both its convergence and its behavior under noisy feedback. The convergence analysis provides theoretical guarantees for stable and correct updates, while the noise tolerance analysis considers the impact of occasional erroneous feedback on the updates.

Convergence analysis.  Since (U​(L)+1)/2\big({U(L)+1}\big)/2 is in the range of [0,1][0,1] and ∑Li∈LP​(Li)=1\sum_{L_{i}\in L}P(L_{i})=1, the expected utility 𝔼​[U​(L)]\mathbb{E}[U(L)] is bounded within (0,1](0,1], which ensures that the loss ℒ\mathcal{L} is non-negative and upper-bounded. This boundedness prevents gradient explosion and provides a stable optimization objective. Moreover, ℒ\mathcal{L} is convex with respect to the path distribution, implying that the objective function admits a unique global optimum over probability distributions (boyd2004convex). In our framework, the contribution score Sc​(t)S_{c}(t) only affects ℒ\mathcal{L} through the softmax-based path probability P​(Li)P(L_{i}) (Formula 3). The softmax mapping is smooth and strictly monotonic, which guarantees that the gradient ∇Sc​(t)ℒ\nabla_{S_{c}(t)}\mathcal{L} exists and is Lipschitz-continuous (nesterov2013introductory). This gradient updates the contribution score of each triplet according to the utility of the paths they participate in. By standard results in stochastic convex optimization (bottou2018optimization), with a properly chosen learning rate, the iterative updates of Sc​(t)S_{c}(t) converge to a stationary point. At convergence, the induced distribution over paths aligns with their utility: paths with consistently high utility receive a larger selection probability, while harmful or low-utility paths are gradually down-weighted. This ensures that the retrieval process adaptively emphasizes informative reasoning paths, reflecting the accumulated feedback across queries.

Noise tolerance.  Feedback-driven backpropagation is inherently noise-tolerant to erroneous feedback and LLM scoring errors in two folds. First, contribution scores are updated cumulatively across queries, allowing occasional errors to be averaged over time. Second, path utility is updated by supportiveness (the feedback-driven dimension) only when fidelity is high and conflict is low; otherwise, the update is suppressed. This cross-validation mechanism prevents spurious feedback or LLM misjudgments from distorting path evaluation, while reinforcing consistently supported paths.

####  4.3.4. Complexity Analysis

We analyze the time and space complexity of EvoRAG by considering two components. For feedback-driven backpropagation, contribution scores are propagated along each reasoning path and aggregated to update triplets. The computation scales with the number of paths PP, the average path length HH, and the number of triplets TT, yielding a time complexity of 𝒪​(2​P+H2​P+T)\mathcal{O}(2P+H^{2}P+T). The memory is dominated by storing path utilities and triplet-level contribution scores, yielding a space complexity of 𝒪​(P+T)\mathcal{O}(P+T). For path evaluation, all reasoning paths are processed in a single LLM call. Let CLLM​(L)C_{\text{LLM}}(L) denote a black-box cost function of an LLM call with input length LL, abstracting away model architecture and deployment details. Since the LL is proportional to the length of all paths, i.e., L=𝒪​(H​P)L=\mathcal{O}(HP), the overall time complexity of path evaluation is 𝒪​(CLLM​(H​P))\mathcal{O}(C_{\text{LLM}}(HP)), while the space overhead is dominated by the prompt and intermediate LLM states.

##  5\. Feedback-guided KG Management

This section presents the feedback-guided KG management mechanism of EvoRAG, which involves relation-centric KG evolution and hybrid priority-based retrieval involving similarity and contribution score for improving retrieval quality.

###  5.1. Relation-centric KG Evolution

We refine the KG based on feedback-derived contribution scores that reflect how each triplet supports reasoning. Instead of updating entities, EvoRAG adopts a relation-centric evolution strategy. Since entities are shared across multiple triplets, modifying them would introduce cascading and ambiguous effects. In contrast, each relation uniquely defines the semantic intent of a triplet and determines whether the connection between two entities is meaningful, which aligns precisely with what the contribution score measures.

The KG evolution is performed after each feedback iteration, which consists of multiple batches of user queries. After aggregating triplet-level scores, we calculate the global mean μ\mu and standard deviation σ\sigma, which define refinement thresholds: triplets with scores greater than τh​i​g​h=μ+σ\tau_{high}=\mu+\sigma are considered high quality, while those consistently below τl​o​w=μ−σ\tau_{low}=\mu-\sigma throughout iterations are considered low quality. These thresholds provide a data-driven basis, ensuring that relation fusion and pruning are guided by statistically significant differences rather than random fluctuations.

Guided by these thresholds, the KG is evolved through two operations: Relation Fusion and Relation Suppression. As illustrated in Figure 6, high-quality relations are reinforced by adding shortcut edges that connect the endpoints of multiple hops of high-quality paths, while persistently low-quality relations are suppressed via reduced contribution scores. Together, these operations balance adaptivity with stability, ensuring that the KG evolves in alignment with feedback while maintaining structural coherence.

Figure 6. Illustration of relation-centric KG evolution, where triplets with low contribution scores are progressively down-weighted in retrieval probability. 

  * •

Relation fusion. This operation strengthens high-quality relations by abstracting a shortcut edge that connects the endpoints of a multi-hop path, effectively reducing reasoning depth. For a path Li=(t1,t2,…,tk)L_{i}=(t_{1},t_{2},\dots,t_{k}) whose triplets satisfy Sc​(ti)>τh​i​g​hS_{c}(t_{i})>\tau_{high}, EvoRAG abstracts a shortcut edge r^\hat{r}:

(10) |  | (e1,r1,e2),(e2,ri,e3),…,(ek−1,rk,ek)⇒(e1,r^,ek),\displaystyle(e_{1},r_{1},e_{2}),(e_{2},r_{i},e_{3}),\dots,(e_{k-1},r_{k},e_{k})\Rightarrow(e_{1},\hat{r},e_{k}), |   
---|---|---|---  
  
where the r^\hat{r} is assigned a label and a score: the label is recommended by the LLM based on the semantics of the multi-hop path, and the score is set to the path’s average contribution.

  * •

Relation Suppression.  We progressively suppress low-quality triplets based on their long-term contribution patterns through two strategies. Low-contribution triplets are softly deprioritized during retrieval. As defined in Formula 2, retrieval is jointly guided by semantic similarity and contribution score, which reduces the retrieval probability of low-score triplets without immediately discarding them. Triplets that are semantically relevant to other queries may still be retrieved and can regain contribution once they prove useful again (Section 5.2).




Algorithm 1 summarizes the KG evolution process. Given the KG 𝒢(h)\mathcal{G}^{(h)}, we first compute the mean μ\mu and standard deviation σ\sigma of triplet contribution scores Sc(h)​(t)S_{c}^{(h)}(t) over all triplets t∈𝒢(h)t\in\mathcal{G}^{(h)} (line 1). Then, we start from triplets whose scores exceed the threshold (line 3), and perform parallel BFS triplets to identify candidate multi-hop paths (lines 4-17). A shortcut edge is created between the endpoints of a path if (i) the path’s average contribution score exceeds the threshold and (ii) no existing edge connects the endpoints in the KG (lines 12-14). To improve efficiency, neighbors are traversed in descending order of Sc(h)S_{c}^{(h)} (line 9), allowing early termination within each BFS layer when remaining neighbors fall below the threshold (line 16). This process ensures that the KG gradually refines its structure, strengthening consistently useful relations while suppressing noisy or unreliable ones.

It is worth noting that the current evolution module focuses on refining relations rather than adding or removing entities. Entity modification corresponds to factual correction, which requires external knowledge verification beyond the scope of feedback-based optimization. In contrast, our contribution score mechanism adjusts the structural importance of existing relations to improve reasoning behavior. Thus, entity-level updates and relation-centric evolution address orthogonal aspects, and future work will explore their integration for joint factual and behavioral adaptation.

Input: KG 𝒢(h)\mathcal{G}^{(h)} at iteration hh, contribution scores Sc(h)S_{c}^{(h)} for each triplet tt appearing in 𝒢\mathcal{G}, max hop HH

Output: Updated KG 𝒢(h+1)\mathcal{G}^{(h+1)}

1 Compute global mean μ\mu and std σ\sigma of Sc(h)S_{c}^{(h)};

2 Initialize S​h​o​r​t​c​u​t=[]Shortcut=[];

3 Ts​t​a​r​t←{t|t∈𝒢T_{start}\leftarrow\\{t|t\in\mathcal{G} and Sc(h)(t)≥μ+σ}S_{c}^{(h)}(t)\geq\mu+\sigma\\};

4 parallel for _t s​t​a​r​t∈Ts​t​a​r​tt_{start}\in T_{start}_ do

5 F​r​o​n​t​i​e​r←[ts​t​a​r​t]Frontier\leftarrow[t_{start}]; 

6 for _h ​o​p=1→Hhop=1\to H and F​r​o​n​t​i​e​r≠[]Frontier\neq[]_ do

7 N​e​x​t​F​r​o​n​t​i​e​r←[]NextFrontier\leftarrow[]; 

8 for _each tc​u​r​r∈F​r​o​n​t​i​e​rt_{curr}\in Frontier_ do

9 N←NbrTriplet​(tc​u​r​r)N\leftarrow\textsc{NbrTriplet}(t_{curr}); // sorted by Sc(h)S_{c}^{(h)} to prioritize high-contribution relations

10 for _each tn​b​r∈Nt_{nbr}\in N_ do

11 S¯←Score​(ts​t​a​r​t,tn​b​r)\bar{S}\leftarrow\textsc{Score}(t_{start},t_{nbr}); // avg. path score

12 if _S ¯≥μ+σ\bar{S}\geq\mu+\sigma and ∄(ts​t​a​r​t.head,r,tn​b​r.tail)∈KG\nexists(t_{start}.head,r,t_{nbr}.tail)\in KG_ then

13 S​h​o​r​t​c​u​tShortcut.push(((ts​t​a​r​t.head,r∗,tn​b​r.tail),S¯(t_{start}.head,r^{*},t_{nbr}.tail),\bar{S}));

14 N​e​x​t​F​r​o​n​t​i​e​r.push​(tn​b​r)NextFrontier.\textsc{push}(t_{nbr}); 

15

16 else

17 break; 

18

19

20

21 F​r​o​n​t​i​e​r←N​e​x​t​F​r​o​n​t​i​e​rFrontier\leftarrow NextFrontier; 

22

23

24

25G(h+1)←(G(h)∪S​h​o​r​t​c​u​t)G^{(h+1)}\leftarrow(G^{(h)}\cup Shortcut);

26 return T(h+1)T^{(h+1)}

Algorithm 1 Feedback-driven KG Evolution.

###  5.2. Hybrid Priority-based Retrieval

EvoRAG introduces a hybrid priority-based retrieval module that jointly leverages semantic relevance and feedback-derived triplet contribution. Building on existing KG-RAG retrieval methods (lego-graphrag_vldb25; lightrag_arxiv_2024; MicrosoftGraphRAG_arxiv_2024) that emphasize short-term semantic relevance, our approach incorporates long-term feedback to progressively adjust retrieval priorities according to user needs and knowledge reliability.

When a query qq arrives, EvoRAG first encodes it into an embedding vector and retrieves the top-NN relevant entities (EqE_{q}) as starting nodes based on cosine similarity. The kk-hop neighborhoods of these entities are traversed to construct a high-recall subgraph that preserves potentially useful multi-hop relations. This step ensures comprehensive coverage of potentially useful knowledge.

Within the extracted subgraph, EvoRAG performs a hybrid ranking of candidate reasoning paths. Each triplet tt is associated with two complementary scores: (1) a relevance score Sr​(t)S_{r}(t) that measures semantic similarity to qq, and (2) a contribution score Sc​(t)S_{c}(t) that captures its historical utility across iterations. These scores are integrated into a unified path priority P​(Li)P(L_{i}) (Eq. 3), which adaptively balances short-term relevance with long-term reliability. For each starting entity e∈Eqe\in E_{q}, EvoRAG retains the top-MM paths with the priorities, thereby suppressing those dominated by low-quality triplets and emphasizing reusable, high-confidence knowledge.

Importantly, EvoRAG remains compatible with various retrieval backends. While Sr​(t)S_{r}(t) currently follows a similarity-based formulation (lego-graphrag_vldb25; lightrag_arxiv_2024; MicrosoftGraphRAG_arxiv_2024), it can be replaced with other strategies such as ToG (TOG_ICLR24) or DALK (DALK_EMNLP24). The key innovation lies in the integration of Sc​(t)S_{c}(t), which injects feedback-driven triplet quality into the retrieval process, forming a flexible and adaptive hybrid retrieval paradigm.

Empirically, setting N=10N=10 and M=10M=10 yields the best performance, achieving high response accuracy. Larger values bring marginal gains while increasing retrieval cost (see Section 6.9). This hybrid retrieval design improves reasoning accuracy by filtering low-contribution triplets and reducing unnecessary context, resulting in more efficient and focused LLM input.

##  6\. EXPERIMENTAL EVALUATION

###  6.1. Experimental Setup

Environments.  The experiment is conducted on a GPU server equipped with 22 Intel(R) Xeon(R) Silver 43164316 CPUs, 503503GB DRAM, and 2×2\times NVIDIA RTX A6000 (4848 GB) GPUs. The server runs Ubuntu 20.0420.04 OS (Linux kernel 5.15.05.15.0) with GCC-9.4.09.4.0, CUDA 11.311.3 with driver version 560.35560.35, and PyTorch 1.13.01.13.0 backend.

Datasets.  We evaluate the effectiveness of EvoRAG on three real-world datasets, as summarized in Table 1. RGB (RGB_benchmark_AAAI24) is constructed from news articles to evaluate reasoning capabilities of RAG tasks. MultiHop (MTH) (Multihoprag_arxiv24) focuses on multi-hop queries that require integrating information from multiple news documents. HotpotQA (HPQ) (Hotpotqa_EMNLP18) is a multi-hop QA dataset derived from Wikipedia, where answering each query requires combining evidence from two paragraphs. We use all 300300 English-language reasoning queries from the RGB dataset and all 816816 reasoning queries from the MTH dataset. For HPQ, we follow prior work (TCROF_arxiv25) and randomly select 600600 queries for evaluation due to the high cost of processing the full dataset.

Training and test set construction.  To evaluate EvoRAG under realistic online-service conditions, we simulate a multi-user scenario where numerous queries concentrate on the same local regions of the knowledge graph. Based on this setting, we construct the training and test sets as follows. The test set directly uses the original queries listed in Table 1, representing user queries in these hotspot regions. The training set is constructed by generating additional query–answer pairs within the subgraphs retrieved based on the test queries. For each subgraph, we sample alternative reasoning paths (up to two hops) that differ from the paths used to form test queries, avoiding data leakage. The corresponding target entities are used as ground-truth answers, producing diverse yet localized queries that simulate multi-user access to the same hotspot regions. During training, the generated queries are issued over multiple iterations to simulate real-time user interactions, allowing the system to collect feedback and update the KG incrementally. The model’s performance is then evaluated on the test set.

Table 1. Dataset description and the corresponding sizes of constructed KGs. Dataset | Query | Type | Entity | Triplets  
---|---|---|---|---  
RGB (RGB_benchmark_AAAI24) | 300 | Single-hop | 54,544 | 74,394  
Multihop (MTH) (Multihoprag_arxiv24) | 816 | Multi-hop | 30,953 | 26,876  
HotpotQA (HPQ) (Hotpotqa_EMNLP18) | 600 | Multi-hop | 76,280 | 74,942  
Table 2.  Performance comparison with KG-RAG and KGR methods. MRAG, LRAG, and KRAG are KG-RAG baselines. TransE, RotatE, CAGED, and LLM_Sim are KGR methods applied to refine the KG, whose effects are evaluated using KRAG due to its operation at the triplet-level retrieval granularity. Best results are highlighted in bold, and worst results in red. ↑ indicates the accuracy improvement over the worst result.  Dataset | Metric | Existing KG-RAG frameworks | KRAG with various KGR methods | EvoRAG  
---|---|---|---|---  
MRAG | LRAG | KRAG | TransE | RotatE | CAGED | LLM_Sim  
RGB | #ACC | 75.67%75.67\% | 76.00%76.00\% | 71.00%71.00\% | 67.67%67.67\% | 74.33%74.33\% | 73.33%73.33\% | 67.33%67.33\% |  84.00% ↑\uparrow8.00-16.67  
#EM | 47.33%47.33\% | 47.00%47.00\% | 42.33%42.33\% | 42.67%42.67\% | 47.33%47.33\% | 45.67%45.67\% | 43.33%43.33\% |  56.67% ↑\uparrow9.34-14.34  
#F1 | 68.69%68.69\% | 64.99%64.99\% | 64.21%64.21\% | 60.61%60.61\% | 68.47%68.47\% | 67.02%67.02\% | 62.95%62.95\% |  75.40% ↑\uparrow6.71-14.79  
MTH | #ACC | 75.61%75.61\% | 76.20%76.20\% | 74.02%74.02\% | 50.12%50.12\% | 76.47%76.47\% | 76.72%76.72\% | 72.06%72.06\% |  80.26% ↑\uparrow3.54-30.14  
#EM | 69.61%69.61\% | 70.43%70.43\% | 71.94%71.94\% | 50.37%50.37\% | 72.55%72.55\% | 74.02%74.02\% | 71.45%71.45\% |  78.55% ↑\uparrow6.00-28.18  
#F1 | 75.86%75.86\% | 74.87%74.87\% | 74.87%74.87\% | 52.23%52.23\% | 77.08%77.08\% | 77.83%77.83\% | 73.27%73.27\% |  80.80% ↑\uparrow2.97-28.57  
HPQ | #ACC | 38.83%38.83\% | 44.83%44.83\% | 39.00%39.00\% | 27.67%27.67\% | 36.33%36.33\% | 29.17%29.17\% | 32.83%32.83\% |  48.16% ↑\uparrow3.33-20.49  
#EM | 25.50%25.50\% | 25.67%25.67\% | 24.83%24.83\% | 17.83%17.83\% | 24.50%24.50\% | 20.17%20.17\% | 21.50%21.50\% |  35.84% ↑\uparrow10.17-18.01  
#F1 | 37.36%37.36\% | 40.47%40.47\% | 41.32%41.32\% | 29.34%29.34\% | 38.51%38.51\% | 32.21%32.21\% | 36.07%36.07\% |  46.55% ↑\uparrow6.08-17.21  
  
Feedback generation.  By default, we use LLM-generated feedback produced by Qwen2.5-32B (Section 4.1) in our main experiments. Table 3 shows that the generated feedback achieves an agreement ratio of 93.38%93.38\% with the ground truth, indicating that it serves as a reliable proxy for evaluation. To demonstrate the robustness and applicability of EvoRAG, we also evaluate it under diverse feedback settings. We first introduce partially noisy feedback to assess its tolerance to unreliable signals (Section 6.5). We then consider alternative feedback sources, including human expert judgments to reflect practical scenarios without explicit ground truth, and ground-truth-based feedback (e.g., F1 score) as an ideal reference (Section 6.4). The results show that EvoRAG maintains stable effectiveness under various feedback reliability and availability settings.

Table 3. Alignment ratio between generated feedback and ground truth. Dataset | Total Feedback | Correct Feedback | Proportion  
---|---|---|---  
RGB | 300 | 288 | 96%  
MTH | 816 | 783 | 95.96%  
HPQ | 600 | 529 | 88.17%  
  
Baseline methods.  EvoRAG improves the generation quality of KG-RAG frameworks by dynamically refining the knowledge graph based on feedback. To evaluate effectiveness, we compare it with two categories of baselines.

  * •

KG-RAG framework. We compare EvoRAG with three representative KG-RAG baselines, including Microsoft GraphRAG (MRAG) (MicrosoftGraphRAG_arxiv_2024), LightRAG (LRAG) (lightrag_arxiv_2024), and KRAG. MRAG and LRAG are two variants that integrate textual chunks with KGs, providing fine-grained knowledge. MRAG applies community detection algorithms to summarize subgraph information, providing abstract representations. LRAG adopts a dual-level retrieval to combine entity-level and relation-level information. In addition, we adopt KRAG, the most effective KG-RAG instance identified in Lego-GraphRAG (lego-graphrag_vldb25) (see Section 2.2).

  * •

KG-RAG with KGR methods. Since EvoRAG refines the underlying KG, we compare it with KG-RAG frameworks augmented by KG refinement (KGR) methods. We integrate several representative KGR methods into KRAG, including deep learning-based methods such as TransE (TransE_NIPS13), RotatE (RotatR_ICLR19), and CAGED (CAGED_CIKM22), and an LLM-based method, LLM_sim (LLMKGR_GenAIK25). These methods aim to identify and correct noisy or irrelevant triplets.




KG construction.  Since the datasets contain only raw texts, we construct KGs following prior work (MicrosoftGraphRAG_arxiv_2024; lightrag_arxiv_2024). Texts are split into 512-token chunks, from which GPT-4o-mini extracts entities and relations using predefined prompts. Table 1 reports KG statistics. MRAG and LRAG further build higher-level textual representations, while KRAG and EvoRAG operate directly on triplet-based KGs.

Implementation details.  For each query, KRAG extracts N=10N=10 query entities and collects their 22-hop neighbors to form a candidate subgraph. From this subgraph, up to M=10M=10 reasoning paths per query entity are selected. EvoRAG builds upon KRAG by introducing a feedback-driven backpropagation mechanism that learns a contribution score for each triplet (with a learning rate of 0.50.5) during online operation to guide the retrieval over the progressively refined KG. For KGR methods, TransE and RotatE identify noisy triplets using the constraint ‖(h+r−t)‖<γ||(h+r-t)||<\gamma, where hh and tt denote the embeddings of head and tail entities, and rr is the relation embedding. We set γ=0.1\gamma=0.1 following (LLMKGR_GenAIK25). To ensure fair comparison, we adopt Qwen2.5-32B as the unified LLM backbone across all methods. It is used for path-level evaluation in EvoRAG and response generation in EvoRAG and all baselines.

###  6.2. Overall Comparison

We compare EvoRAG with various KG-RAG frameworks and with KRAG enhanced by different KGR methods, using accuracy (ACC), exact match (EM), and F1 as evaluation metrics. ACC measures the proportion of queries for which the response contains the ground-truth answer. EM indicates whether the response exactly matches the ground-truth answer. F1 evaluates the token-level overlap by computing the harmonic mean of accuracy and recall. EvoRAG is trained on the training set, enabling it to adaptively refine the KG. Table 2 reports the experimental results.

Compared to KG-RAG frameworks, EvoRAG achieves an average improvement of 7.34%7.34\% in ACC, 9.84%9.84\% in EM, and 7.29%7.29\% in F1 score across three datasets. These gains primarily stem from the ability of EvoRAG to adapt retrieval to downstream query and dynamically correct noisy or incomplete knowledge. Existing KG-RAG frameworks rely on static KGs, where irrelevant or noisy knowledge may suppress useful reasoning paths. This prevents them from retrieving high-quality context and ultimately limits performance. In contrast, EvoRAG leverages a feedback-driven backpropagation mechanism that propagates response-level feedback to individual knowledge triplets. This mechanism addresses cases where baselines miss correct answers. EvoRAG reweights suppressed but useful paths upward and diminishes the influence of misleading ones. Accumulated across queries, these adjustments enable the system to recover answers systematically overlooked by static methods. In addition, the KG evolution mechanism in EvoRAG enhances the KG’s adaptability to RAG tasks, allowing it to evolve continuously during online service by suppressing low-quality triplets and generating new relations.

Figure 7. The problematic triplets ratio and response accuracy across iterations. Each iteration involves answering all queries and propagating corresponding feedback.

Compared to KGR-based KGRAG methods, EvoRAG achieves an average improvement of 13.80%13.80\% in ACC, 12.74%12.74\% in EM, and 11.28%11.28\% in F1 score across three datasets. Traditional DL-based KGR methods, such as TransE, RotatE, and CAGED, rely on supervised learning to detect noisy triplets. These methods require fine-grained labels indicating the correctness of individual triplets, which are typically unavailable in KG-RAG scenarios. LLM_sim attempts to overcome this limitation by leveraging LLMs to evaluate and refine triplets based on internal knowledge. However, it suffers from hallucinations, especially when encountering knowledge beyond the model’s training data, and may introduce new noise or mistakenly modify correct facts. Therefore, traditional KGR methods are not suitable for the KG-RAG framework and may even significantly degrade accuracy. In contrast, EvoRAG directly leverages the real-time feedback, which provides more direct and task-relevant supervision. Although this feedback may occasionally be noisy and coarse-grained, EvoRAG leverages it through a feedback-driven backpropagation mechanism that aggregates feedback across queries and propagation the feedback into a reliable basis for knowledge refinement. In this way, EvoRAG continually adapts the knowledge graph to actual usage patterns, enabling sustained improvements in reasoning performance over time.

Figure 8. Accuracy across feedback sources. LF, GF, and HF denote LLM, ground-truth (F1), and human feedback.

Figure 9. Noise tolerance analysis. 10%10\% and 20%20\% erroneous feedback (EF) is injected during each iteration. 

Figure 10. Ablation study of feedback mechanism.

Figure 11. Ablation study on the use of Fidelity and Conflict for cross-validation (CV).

###  6.3. The Improvement of KG Quality

KG evolution is performed once per iteration, where each iteration processes a set of training queries (equal in size to the test set) in batches, with contribution scores updated after each batch. To evaluate its impact on retrieval quality, we track training accuracy and the number of problematic triplets retrieved per query. Problematic triplets defined in Section 2.2 as irrelevant, outdated, incorrect, or long-path connections, are manually annotated.

Figure 7 shows that as iterations increase, the ratio of problematic triplets steadily decreases while accuracy improves and stabilizes after approximately 66 iterations across all datasets. This indicates that EvoRAG can effectively incorporate useful feedback within a few iterations. On RGB, EvoRAG suppresses around 83.01%83.01\% of problematic triplets, achieving larger gains than HPQ due to the dataset’s high level of noisy facts. Performance improvements are most significant in early iterations, where EvoRAG can rapidly identify and down-weight misleading triplets. As iterations progress, the performance gradually converges as the system stabilizes.

###  6.4. Impact of Different Feedback Sources

To evaluate the applicability of EvoRAG, we compare its accuracy under different feedback sources. Specifically, we consider LLM-generated feedback, ground-truth based feedback (using F1 score), and human feedback, where users assign satisfaction scores to each response. As shown in Figure 11, EvoRAG achieves comparable performance across all settings, with an average difference of 0.75%0.75\%. This suggests that the performance gains mainly stem from the availability of feedback signals rather than their specific source.

###  6.5. Feedback Tolerance Analysis

To evaluate the impact of noisy feedback on EvoRAG, we simulate erroneous feedback by randomly selecting a fraction of the collected feedback and inverting the correctness. In the noise injection, we randomly select 10%10\% and 20%20\% of the feedback scores excluding neutral scores of 3, and flip them: scores 1−21-2 are converted to 4−54-5, and scores 4−54-5 are converted to 1−21-2. This process simulates imperfect or noisy feedback, reflecting realistic scenarios where evaluation judgments may be inaccurate or inconsistent.

The experimental results are reported in Figure 11. We observe that EvoRAG remains tolerant under noisy feedback. Even with 10%10\% and 20%20\% erroneous feedback, accuracy drops only slightly, by 1.15%1.15\% and 2.43%,2.43\%, respectively. This resilience stems from the cumulative update of contribution scores across multiple queries and reasoning paths. Since each triplet is evaluated from diverse contexts, occasional erroneous feedback has only a transient impact, while consistent signals dominate over time, enabling accurate convergence despite noise.

###  6.6. Ablation Study

####  6.6.1. Ablation study of Feedback Mechanism

To evaluate the contribution of the feedback mechanism, we compare four settings: LLM-only, LLM with static KG (SG), LLM+SG with feedback (FB), and LLM+SG+FB with graph evolution (i.e., EvoRAG). As shown in Figure 11, SG outperforms the LLM-only by 23.42%23.42\% in accuracy, demonstrating KG-RAG’s ability to supplement missing knowledge. FB further improves accuracy by 6.6%6.6\% by prioritizing reliable triplets via feedback, while EvoRAG adds 3.07%3.07\% by updating the KG to prune low-utility triplets and reinforce correct long-range reasoning. These results demonstrate the effectiveness of the feedback mechanism in improving reasoning accuracy.

####  6.6.2. Ablation Study of Cross-validation Mechanism

We evaluate path utility updates with and without Fidelity and Conflict. As shown in Figure 11, incorporating these cross-validation constraints improves the average accuracy by 2.24%2.24\% over the variant without cross-validation, confirming their effectiveness in filtering unreliable or contradictory paths.

###  6.7. Token Cost Comparison

Figure 12. Comparison of EvoRAG with KG-RAG frameworks in terms of prompt length and time-to-first-token (TTFT).

EvoRAG reduces prompt length by down-weighting noisy, outdated, or irrelevant triplets during retrieval, thereby lowering token cost and improving system efficiency, as token consumption directly determines inference latency and resource overhead. To quantify these benefits, we evaluate prompt token cost and time-to-first-token (TTFT), which captures the latency of LLM responses. As shown in Figure 12, compared to MRAG, LRAG, and KRAG, EvoRAG achieves an average reduction of 4.6×4.6\times in prompt length, resulting in consistently lower TTFT. MRAG and LRAG often retrieve not only reasoning paths but also large amounts of associated text chunks, significantly increasing prompt length and prefill latency. KRAG retrieves raw triplets but lacks mechanisms to suppress semantically redundant or low-quality triplets. In contrast, EvoRAG achieves higher accuracy with shorter prompt length, demonstrating that it retains essential knowledge while successfully filtering out redundant or outdated information.

###  6.8. Runtime Performance Analysis

Figure 13. Comparison of forward and backward propagation time of EvoRAG under different batch sizes.

The backpropagation mechanism in EvoRAG introduces additional overhead compared to traditional KG-RAG frameworks. To mitigate this overhead, we exploit a key observation: the prompts used in forward and backward propagation share a common prefix, including the same query and retrieved reasoning paths, and differ only in the task-specific suffix. Building on this insight, we adopt a KV-cache reuse strategy in vLLM (vllm_sosp23), enabling the model to reuse cached computations for the shared prefix, without compromising effectiveness. We evaluate the runtime of EvoRAG under varying batch sizes, measuring both forward and backward propagation. As shown in Figure 13, KV-cache reuse effectively reduces the overhead of backpropagation, with its relative cost decreasing as batch size increases. At a batch size of 2020, backpropagation accounts for only 23.90%23.90\% of the total runtime. This efficiency gain arises from improved utilization under larger batches, where shared-prefix reuse and system-level scheduling amortize the cost of repeated computations. Overall, these results demonstrate that the overhead introduced by EvoRAG is well-controlled, making it suitable for deployment in real-world online scenarios.

###  6.9. Sensitivity Analysis of Entity and Path Numbers

For KG-RAG frameworks, the number of entities (NN) and reasoning paths per entity (MM) directly impact the amount of knowledge retrieved and overall accuracy. We analyze the sensitivity of them by varying one parameter at a time while keeping the other fixed at 1010. The experimental results are shown in Figure 14. We observe that increasing either NN or MM generally improves accuracy, as more retrieved knowledge provides a richer context for generation. However, the gains diminish when NN and MM reach 10, where additional paths contribute marginally to performance, indicating that a limited number of paths are sufficient to enhance generation. Notably, accuracy is more influenced by the number of entities than by the number of paths. This is because adding entities introduces more diverse and potentially relevant information.

Figure 14. Impact of hyperparameters on accuracy. NN indicates the number of retrieved entities; MM indicates the number of retrieval paths per entity.

###  6.10. Scalability on Large Dataset

Figure 15. Evaluation on 5,000 queries with large KG (890,389 entities, 1,043,360 triplets). (a) Accuracy comparison. (b) Noise tolerance analysis. (c) Performance analysis.

We evaluate EvoRAG in a large-scale setting with 5,000 queries sampled from HotpotQA on a KG containing 890,389 entities and 1,043,360 triplets. Figure 15 summarizes the results. As shown in Figure 15(a), EvoRAG improves the average accuracy over MRAG, LRAG, and KRAG by 8.25%8.25\%, demonstrating that the feedback mechanism remains effective at scale. Figure 15(b) shows that each 10%10\% increase in erroneous feedback reduces accuracy by less than 1%1\%, indicating strong noise tolerance due to cumulative aggregation across queries. Figure 15(c) shows that while the larger KG increases overall runtime (mainly in forward propagation due to retrieval overhead), backpropagation remains lightweight, accounting for only 15.35%15.35\% of total runtime at batch size 20.

###  6.11. Effectiveness Analysis of Feedback Mechanism

In this section, we analyze how feedback-driven backpropagation affects the KG and retrieval results, and present a case study on representative queries.

Table 4. Proportions of added and removed triplets in retrieval results after KG evolution. Proportion of Triplets | RGB | MTP | HPQ  
---|---|---|---  
Added | 24.41% | 21.06% | 13.46%  
Removed | 28.97% | 38.23% | 17.31%  
  
####  6.11.1. Changes in Retrieval Results

We analyze the impact of KG evolution on retrieval by comparing the retrieved triplets before and after evolution. As shown in Figure 4, 28.17%28.17\% of low-contribution triplets are removed, while 19.64%19.64\% of high-contribution triplets are newly introduced. This shows that KG evolution shifts retrieval away from repeatedly selecting unhelpful relations and toward incorporating triplets that have proven useful.

####  6.11.2. Case study

Figure 16 shows a case study for a query from the HPQ dataset. In the original retrieval, path contains a clear factual error, but is ranked higher due to its high semantic similarity to the query, causing the correct path to Strange Interlude to be ranked lower. After feedback continuously indicates an incorrect answer in 8 iterations, EvoRAG suppresses this noisy triplet by reducing its contribution score and reinforces correct triplets. As a result, the correct path becomes more salient in retrieval, leading to a correct answer in subsequent queries.

Figure 16. Comparison of retrieved paths of a query over HPQ before and after 8 feedback iterations.

###  6.12. The Effectiveness of Feedback-driven Backpropagation in MRAG and LRAG

Figure 17. Accuracy comparison of MRAG and LRAG on two datasets with Feedback-driven Backpropagation (FB).

Since MRAG and LRAG retrieve text chunks and use graphs only as indices, we adapt our framework to operate at the chunk level to support feedback-driven backpropagation. Specifically, we define contribution scores for text chunks and incorporate them into retrieval ranking alongside semantic similarity. As shown in Figure 17, the integration improves the average accuracy of MRAG and LRAG by 2.83%2.83\% and 1.92%1.92\%, respectively. The gains are smaller than those achieved by the triplet-level framework, as each text chunk aggregates multiple facts and thus provides coarser-grained feedback signals.

###  6.13. Performance with Various LLM Backends

We evaluate the generality of EvoRAG across different LLM backends by replacing Qwen2.5-32B with Llama-3.1-70B (4-bit) and GPT-4o-mini for all LLM-dependent components in EvoRAG and the baselines. As shown in Figure 18, EvoRAG consistently outperforms existing RAG methods, improving average accuracy by 8.5%8.5\% with Llama-3.1-70B and by 7.6%7.6\% with GPT-4o-mini. These results indicate that the performance gains of EvoRAG are robust to the choice of LLM backend.

Figure 18. Accuracy comparison of different KG-RAG frameworks across different LLM backends.

##  7\. Related Work

KG-RAG framework.  KG-RAG frameworks can be broadly grouped into two categories. One line of work represents knowledge as textual chunks and uses graph structures mainly for indexing inter-chunk relations (Raptor_ICLR24; PG-RAG_arxiv24; TOG2_arxiv24; Graphcoder_arxiv24; HippoRAG_NIPS24; MicrosoftGraphRAG_arxiv_2024; lightrag_arxiv_2024). While effective for summarization, these methods are less suitable for complex reasoning. Another line explicitly constructs and leverages KGs to support multi-hop reasoning over structured triplets (ProLLM_bioRxiv24; TOG_ICLR24; KGRAG_arxiv24; Knowgpt_NIPS24; KELP_ACL24; kag_arxiv24; PathRAG_arxiv25; DALK_EMNLP24; RuleRAG_arxiv24; GNNRAG_arxiv_2024; Graph_CoT_ACL24; Gretriever_NIPS24; Structrag_arxiv24; GraphReader_EMNLP24; StructureGuided_arxiv24). However, most of these approaches emphasize retrieval and prompt design, without fully exploiting the KG’s capacity for complex reasoning, which is critical for accurate responses (GraphRAGBench_arxiv25).

KG refinement.  KG refinement (KGR) focuses on enhancing the factual accuracy and utility of KGs by removing redundant triplets, correcting incorrect facts, and adding missing information. Rule-based methods (Probabilistic_KG_noise_detection_CVPR19; Probabilistic_KG_noise_detection_CIKM19; KGRefine_survey_2016; rule_base_KG_noise_detection_PRICAI21) rely on logical constraints but scale poorly, while DL-based approaches (mlkgr_survey_2024; embed_rulelearn_KGR_TKDE19; KGRsurvey_2021; TransE_NIPS13; TransT_PKDD17; TransG_ACL16; RotatR_ICLR19) typically operate offline and require high-quality training data. Recently, LLM-based methods (LLMKGR_GenAIK25; KICGPT_EMNLP23; TCROF_arxiv25) assess semantic plausibility or generate facts, but remain detached from downstream tasks.

KG validation.  KG validation (KGV) (KGT_arxiv24; KGV_IPM25; cleangraph_arxiv24) assesses triplet correctness by relying on external evidence and expert-curated annotations, which often require task-specific pipelines and substantial expert involvement, thereby limiting scalability.

Feedback-driven model optimization.  These approaches (ouyang2022training; yu2025self; zhang2024cppo) optimize model outputs in a feedback-driven manner via fine-tuning or reinforcement learning, yet operate solely at the model level without updating the KG.

##  8\. Conclusion

We propose EvoRAG, a self-evolving KG-RAG framework that leverages real-time feedback to continuously refine the KG and improve reasoning accuracy. EvoRAG introduces a feedback-driven backpropagation mechanism that connects response-level feedback and triplet-level knowledge updates, which attributes feedback to individual reasoning paths and propagates it to adjust the scores of involved triplets. This establishes a closed loop, where reasoning feedback drives KG refinement, and the evolving KG improves future reasoning. The experimental results demonstrate that EvoRAG is both effective and robust, offering a scalable solution for adaptive and scalable KG maintenance in real-world applications.

[◄](/html/2604.15675) [](/) [Feeling  
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion  
report](/log/2604.15676) [Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2604.15676) [View original  
on arXiv](https://arxiv.org/abs/2604.15676)[►](/html/2604.15677)

[](javascript:toggleColorScheme\(\) "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Tue May 5 19:33:09 2026 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
