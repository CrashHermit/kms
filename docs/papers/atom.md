# ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs

Yassir LAIRGI1,2, Ludovic MONCLA1, Khalid BENABDESLEM1,   
Rémy CAZABET1 Pierre CLÉAU2   
1LIRIS, INSA Lyon, Université Claude Bernard Lyon 1, France   
2GAUC, Lyon, France   
{ludovic.moncla, khalid.benabdeslem, remy.cazabet}@liris.cnrs.fr   
{yassir.lairgi, pierre.cleau}@auvalie.com

###### Abstract

In today’s rapidly expanding data landscape, knowledge extraction from unstructured text is vital for real-time analytics, temporal inference, and dynamic memory frameworks. However, traditional static knowledge graph (KG) construction often overlooks the dynamic and time-sensitive nature of real-world data, limiting adaptability to continuous changes. Moreover, recent zero- or few-shot approaches that avoid domain-specific fine-tuning or reliance on prebuilt ontologies often suffer from instability across multiple runs, as well as incomplete coverage of key facts. To address these challenges, we introduce ATOM111The code, prompts, and dataset are available at <https://github.com/AuvaLab/itext2kg>. ATOM is available as an open-source Python library. (AdapTive and OptiMized), a few-shot and scalable approach that builds and continuously updates Temporal Knowledge Graphs (TKGs) from unstructured texts. ATOM splits input documents into minimal, self-contained “atomic” facts, improving extraction exhaustivity and stability. Then, it constructs atomic TKGs from these facts while employing a dual-time modeling that distinguishes when information is observed from when it is valid. The resulting atomic TKGs are subsequently merged in parallel. Empirical evaluations demonstrate that ATOM achieves ∼18%\sim 18\% higher exhaustivity, ∼17%\sim 17\% better stability, and over 90% latency reduction compared to baseline methods, demonstrating a strong scalability potential for dynamic TKG construction.

ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs

Yassir LAIRGI1,2, Ludovic MONCLA1, Khalid BENABDESLEM1, Rémy CAZABET1, and Pierre CLÉAU2 1LIRIS, INSA Lyon, Université Claude Bernard Lyon 1, France 2GAUC, Lyon, France {ludovic.moncla, khalid.benabdeslem, remy.cazabet}@liris.cnrs.fr {yassir.lairgi, pierre.cleau}@auvalie.com

##  1 Introduction

Unstructured data is expanding at an unprecedented rate (Dresp-Langley et al., 2019), and given that the majority of big data is inherently unstructured (Trugenberger, 2015), there is an urgent need for robust information extraction and data modeling techniques to unlock its potential and derive insights across a broad spectrum of applications (Cetera et al., 2022). A prominent model for converting this unstructured data into structured, actionable knowledge is the Knowledge Graph (KG) (Zhong et al., 2023).

KG construction involves identifying entities, relationships, and attributes from diverse data sources to create structured knowledge representations. Traditionally, many approaches have focused on static Knowledge Graphs (KGs), which provide snapshots of knowledge without incorporating temporal dynamics. However, as real-world phenomena are inherently dynamic, static KGs, rarely or never updated, struggle to remain relevant and accurate (Jiang et al., 2023). In contrast, Temporal Knowledge Graphs (TKGs) integrate time dimensions by associating timestamps or time intervals with facts (e.g., (Einstein, was awarded, the Nobel Prize, in 1921)), making them particularly well-suited for analyzing changes, trends, and enabling temporal reasoning.

GraphRAG (Edge et al., 2024) and agent-based architectures (Xi et al., 2023) have demonstrated the potential of TKGs in retrieving and modeling dynamic information (Wu et al., 2024). Additionally, TKGs have been effectively used to model the memory of agents within agentic systems (Anokhin et al., 2024), highlighting their role in capturing the evolving nature of knowledge for adaptive and responsive systems.

Traditional methods for KG construction, often reliant on entity recognition and relation extraction, face several limitations. They typically depend on predefined ontologies and supervised learning techniques that require extensive human annotation (Al-Moslmi et al., 2020). Recent advances in Large Language Models (LLMs) (Jin et al., 2023) and zero- or few-shot techniques (Zhang et al., 2024; Carta et al., 2023; Hu et al., 2023) have paved the way for more flexible KG construction approaches that reduce dependency on extensive training datasets.

Despite these advances, current zero- or few-shot methods for KG construction often suffer from several limitations. They can be non-exhaustive, omitting key relationships, and prone to instability, where multiple construction runs on the same text yield different results. Moreover, many of these approaches overlook the temporal dimension of the input data and struggle to adapt to real-life scenarios with dynamic, evolving data, leading to false positives and a lack of scalability (Cai et al., 2024).

In this paper, we propose ATOM (AdapTive and OptiMized), a few-shot and scalable dynamic TKG construction approach from unstructured text, ensuring stability and exhaustivity. ATOM introduces a strategy that decomposes unstructured text into atomic facts. Rather than processing these atomic facts sequentially, ATOM proposes an architecture with parallel 5-tuple extraction, followed by a parallel atomic merging mechanism. In the rest of the paper, we present related work in Section 2, our proposed approach ATOM in Section 3, experimental evaluation in Section 4, conclusion in Section 5, and present the limitations in Section 6.

##  2 Related work

Current zero- and few-shot approaches to KG construction, such as AttacKG+ (Zhang et al., 2024), iterative LLM prompting pipelines (Carta et al., 2023), LLM-Tikg (Hu et al., 2023), LLM Builder 222<https://llm-graph-builder.neo4jlabs.com/>, and LLM Graph Transformer333<https://python.langchain.com/docs/how_to/graph_constructing/> aim to build KGs without requiring task-specific training. However, these methods suffer from inconsistencies such as unresolved entities and relations.

That is why iText2KG Lairgi et al. (2024) introduces an incremental, zero-shot architecture that constructs KGs iteratively by comparing newly extracted entities and relations with existing ones using embeddings and cosine similarity, achieving performance gains over some state-of-the-art LLM-based methods. However, it produces non-exhaustive and non-stable KGs due to the stochastic nature of LLMs Atil et al. (2024). Moreover, it fails to incorporate the temporal dimension, and scalability remains a significant challenge when applying it to real-world scenarios due to its incremental nature.

Graphiti (Rasmussen et al., 2025) proposed a dynamic TKG construction approach for agents’ memory with an exclusively LLM-based entity/relation and temporal resolution framework. A key limitation is that it relies solely on prompting the LLM across all its modules, making it heavily dependent on LLM calls. The system prompts the LLM with all previous entities for entity resolution, which becomes impractical as the graph scales to millions of nodes. Similarly, time conflicts are resolved exclusively through LLM calls, resulting in high computational costs and scalability challenges when applied to large-scale datasets. Moreover, they do not handle the exhaustivity and stability of the constructed TKGs.

AriGraph (Anokhin et al., 2024) integrated semantic and episodic memories to support reasoning, planning, and decision-making in LLM agents. However, their entity resolution method leads to semantic drift in the temporal KG, where, for example, a reference to “Apple” might ambiguously denote either the company or the fruit. Furthermore, scalability becomes problematic as the volume of unstructured data increases.

Despite these advances, current zero- and few-shot TKG construction methods face three key limitations: (1) they struggle to maintain exhaustive fact coverage when processing longer texts, (2) they often produce unstable TKGs across multiple runs, and (3) they lack scalable architectures for dynamic temporal updates. To address these challenges, we propose ATOM, a framework that combines atomic fact decomposition for exhaustive and stable extraction and parallel merging for scalability.

##  3 Proposed approach: ATOM

In this section, we first present some notations and definitions used throughout the paper and then introduce the formulation of our proposed framework.

Figure 1:  ATOM’s architecture, running in parallel, ensuring scalability, speed, and continuous updates. Unstructured texts observed at time tt are denoted by DtD_{t}, the ii-th temporal atomic fact observed at time tt is denoted by ft,if_{t,i}, the ii-th atomic TKG observed at time tt is denoted by 𝒢it\mathcal{G}^{t}_{i}, the TKG snapshot observed at time tt is denoted by 𝒢st\mathcal{G}_{s}^{t}, and the updated TKG at time tt is denoted by 𝒢t\mathcal{G}^{t}. 

###  3.1 Problem statement

ATOM incorporates dual-time modeling, differentiating between when facts are observed and the temporal information conveyed by the facts themselves, which is characterized by a validity period. This approach better reflects real-world data (Rasmussen et al., 2025; Chekol and Stuckenschmidt, 2018; Meijer, 2022). This separation ensures TKG dynamism and proper inference of relative times by providing the observation time as context to the LLM (eg, ’month ago’).

######  Definition 1 (Dynamic-Temporal KG with Dual-Time Modeling).

Let 𝒯obs\mathcal{T}_{\text{obs}} be an ordered set of _observation timestamps_ at which the KG is updated, and let 𝒯start\mathcal{T}_{\text{start}}, 𝒯end\mathcal{T}_{\text{end}} be sets of timestamps used to label inherent validity period of facts defined by their start and end times, respectively. For each observation time t∈𝒯obst\in\mathcal{T}_{\text{obs}}, a _TKG_ snapshot is defined as:

| 𝒢st=(ℰt,ℛt,𝒯startt,𝒯endt,ℱt)\mathcal{G}_{s}^{t}\;=\;\Bigl(\mathcal{E}^{t},\;\mathcal{R}^{t},\;\mathcal{T}_{\text{start}}^{t},\;\mathcal{T}_{\text{end}}^{t},\;\mathcal{F}^{t}\Bigr) |  | (1)  
---|---|---|---  
  
where:

  * •

ℰt\mathcal{E}^{t} is the set of entities known at _observation time_ tt,

  * •

ℛt\mathcal{R}^{t} is the set of relations known at _observation time_ tt,

  * •

𝒯startt⊆𝒯start\mathcal{T}_{\text{start}}^{t}\subseteq\mathcal{T}_{\text{start}} is the set of _validity start times_ referenced by facts in this snapshot,

  * •

𝒯endt⊆𝒯end\mathcal{T}_{\text{end}}^{t}\subseteq\mathcal{T}_{\text{end}} is the set of _validity end times_ referenced by facts in this snapshot,

  * •

ℱt⊆ℰt×ℛt×ℰt×𝒯startt×𝒯endt\mathcal{F}^{t}\;\subseteq\;\mathcal{E}^{t}\times\mathcal{R}^{t}\times\mathcal{E}^{t}\times{\mathcal{T}_{\text{start}}^{t}}\times{\mathcal{T}_{\text{end}}^{t}} is the set of temporal facts (5-tuples) observed in the snapshot at _observation time_ tt.




A fact in this snapshot is a 5-tuple (quintuple ) (es,rp,eo,ts​t​a​r​t,te​n​d)(e_{s},r_{p},e_{o},t_{start},t_{end}) indicating the relation rp∈ℛtr_{p}\in\mathcal{R}^{t} holds between the subject entity es∈ℰte_{s}\in\mathcal{E}^{t} and the object entity eo∈ℰte_{o}\in\mathcal{E}^{t}. Technically, ts​t​a​r​tt_{start} and te​n​dt_{end} are chosen to be lists to aggregate start and end validity timestamps to track the history of the same fact. Validity start and end timestamps can be unknown, in which case their respective lists are empty (denoted as [.][.]).

A _Dynamic Temporal Knowledge Graph (DTKG)_ , updated at t, is defined as the parallel pairwise merge of these TKG snapshots via the merge operator ⊕\oplus (as described later in Section 3.2.3).

| 𝒢t=⨁t′∈𝒯o​b​s={…,t−1,t}𝒢st′=𝒢t−1⊕𝒢st\mathcal{G}^{t}=\bigoplus_{t^{\prime}\in\mathcal{T}_{obs}=\\{...,t-1,t\\}}\mathcal{G}_{s}^{t^{\prime}}=\mathcal{G}^{t-1}\oplus\mathcal{G}_{s}^{t} |  | (2)  
---|---|---|---  
  
######  Definition 2 (Temporal Atomic Fact with Dual-Time Modeling).

Let 𝒯obs\mathcal{T}_{\mathrm{obs}} be a set of _observation timestamps_ at which new data is ingested, and let 𝒯start\mathcal{T}_{\mathrm{start}} and 𝒯end\mathcal{T}_{\mathrm{end}} be the sets of validity periods mentioned within the data. For each observation time t∈𝒯obst\in\mathcal{T}_{\mathrm{obs}}, let DtD_{t} be an unstructured text that becomes available at tt. A _temporal atomic fact_ ft,if_{t,i} is a short, self-contained snippet derived from DtD_{t} that conveys exactly one piece of information, extracted by the LLM. Depending on the content of the snippet, an atomic fact may or may not explicitly contain a validity period. Formally,

| 𝖤𝗑𝗍𝗋𝖺𝖼𝗍𝖠𝖥𝖺𝖼𝗍𝗌𝖫𝖫𝖬​(Dt)={ft,1,…,ft,mt}\mathsf{ExtractAFacts_{LLM}}(D_{t})=\\{f_{t,1},\dots,f_{t,m_{t}}\\} |  | (3)  
---|---|---|---  
  
An example is provided in Section B in the Appendices. In what follows, the term atomic fact is used for conciseness.

######  Definition 3 (Atomic Temporal KG).

Given an atomic fact ft,if_{t,i} observed at time tt, its _atomic temporal KG_ 𝒢it\mathcal{G}^{t}_{i} is the set of 5-tuples extracted by the LLM:

| 𝒢it\displaystyle\mathcal{G}^{t}_{i} | =𝖤𝗑𝗍𝗋𝖺𝖼𝗍𝖰𝗎𝗂𝗇𝗍𝗎𝗉𝗅𝖾𝗌𝖫𝖫𝖬​(ft,i)\displaystyle=\mathsf{ExtractQuintuples_{LLM}}(f_{t,i}) |  | (4)  
---|---|---|---|---  
|  | ⊆𝒫​(ℰt×ℛt×ℰt×𝒯startt×𝒯endt)\displaystyle\subseteq\mathcal{P}\bigl(\mathcal{E}^{t}\times\mathcal{R}^{t}\times\mathcal{E}^{t}\times{\mathcal{T}_{\text{start}}^{t}}\times{\mathcal{T}_{\text{end}}^{t}}\bigr) |   
  
Concretely, 𝒢it\mathcal{G}^{t}_{i} is the set of 5-tuples (es,rp,eo,ts​t​a​r​t,te​n​d)(e_{s},r_{p},e_{o},t_{start},t_{end}) derived from a single atomic fact ft,if_{t,i} at observation time tt.

Given the definitions 1 and 3, the DTKG, updated at tt:

| 𝒢t=⨁t′∈𝒯o​b​s𝒢st′=⨁t′∈𝒯o​b​s(⨁i∈⟦1,mt′⟧𝒢it′)\mathcal{G}^{t}=\bigoplus_{t^{\prime}\in\mathcal{T}_{obs}}\mathcal{G}_{s}^{t^{\prime}}=\bigoplus_{t^{\prime}\in\mathcal{T}_{obs}}(\bigoplus_{i\in\llbracket 1,m_{t^{\prime}}\rrbracket}\mathcal{G}_{i}^{t^{\prime}}) |  | (5)  
---|---|---|---  
  
Figure F.1 in the Appendices illustrates a detailed example of ATOM’s pipeline.

###  3.2 ATOM’s Framework

Given a continuous stream of unstructured texts, our goal is to construct and maintain a consistent and dynamic TKG, ensuring for each t∈𝒯o​b​st\in\mathcal{T}_{obs}:

  1. (C1)

Exhaustivity: the constructed TKG snapshot 𝒢st\mathcal{G}_{s}^{t} ideally captures every 5-tuple that is semantically present in DtD_{t}.

  2. (C2)

Stability across multiple runs: when the identical 5-tuple extraction prompt is executed repeatedly on the same input text using the same LLM, the resulting TKG snapshots should be nearly identical.




In the following, we detail the different modules of ATOM’s architecture (Figure 1).

####  3.2.1 Module-1: Atomic fact decomposition

ATOM does not construct TKGs directly from raw input documents but first decomposes them into atomic facts (Figure 1). This decomposition addresses a fundamental limitation of LLMs: the "forgetting effect" where models prioritize salient information in longer contexts while omitting key relationships, leading to incomplete knowledge extraction Liu et al. (2023). Following (Hosseini et al., 2024; Chen et al., 2023; Raina and Gales, 2024), ATOM uses LLM-based prompting for decomposition with an optimal chunk size to maintain high exhaustivity (determined experimentally in Section 4.3). However, while prior work focused on information retrieval applications, ATOM applies atomic decomposition specifically for TKG construction. This strategy addresses both conditions: it ensures exhaustivity (C1) by preventing information loss that occurs when LLMs process complex, multi-fact paragraphs, and enhances stability (C2) by providing clear, unambiguous contexts that reduce output variance across multiple runs. Each atomic fact is related to an observation time, and it is necessary to encapsulate the relative validity period presented in the context. The primary computational challenge of this approach is scale: a single document can yield hundreds or thousands of atomic facts. Sequential processing of each fact for 5-tuple extraction, followed by entity/relation and temporal resolution, becomes time-consuming. To address this challenge, ATOM employs a parallel architecture for both extraction and merging phases, as detailed in the subsequent modules.

####  3.2.2 Module-2: Atomic TKGs construction

5-tuples are extracted from each atomic fact in parallel using an LLM, producing atomic TKGs 𝒢it\mathcal{G}^{t}_{i} while embedding nodes and relations following (Lairgi et al., 2024). To facilitate temporal resolution in Module-3, ATOM preprocesses 5-tuples during their extraction. It prevents separate quintuples describing the same temporal fact from coexisting in the same TKG such as (John_Doe, is_ceo, X, [01-01-2025], [.]) and (John_Doe, is_no_longer_ceo, X, [01-01-2026], [.]), which should be resolved into (John_Doe, is_ceo, X, [01-01-2025], [01-01-2026]). During the extraction, few-shot examples are provided as context to the LLM to transform end validity facts into affirmative counterparts while modifying only the tendt_{\text{end}} time. For instance, the statement John Doe is no longer the CEO of X on 01-01-2026 is converted into the 5-tuple (John_Doe, is_ceo, X, [.], [01-01-2026]), ensuring direct matching with the corresponding validity start time 5-tuple during the merge. For relative temporal expressions (e.g., ’a month ago’), the observation time is provided as context, enabling the LLM to infer the validity period.

####  3.2.3 Module-3: Parallel atomic merge of TKGs and DTKG update

ATOM then employs the binary merge algorithm (Algorithm A.1 in the Appendices) to merge pairs of atomic TKGs. The algorithm proceeds in three phases: first, entity resolution searches for exact matches between 𝒢it\mathcal{G}^{t}_{i} and 𝒢i+1t\mathcal{G}^{t}_{i+1} based on name and label. When no exact match exists, cosine similarity is computed, merging entities if similarity exceeds θE\theta_{E}. Second, relation resolution merges relation names regardless of endpoints and timestamps (e.g., owns ⟷\longleftrightarrow possesses ⟷\longleftrightarrow has) using threshold θR\theta_{R}. Third, temporal resolution merges observation and validity time sets for relations with similar (es,rp,eo)(e_{s},r_{p},e_{o}), detecting and aligning end-action facts with their corresponding beginning facts. Unlike Graphiti, ATOM avoids LLM calls during merging, improving scalability and preventing context overflow as the graph expands. The preprocessing of end-actions during extraction enables this LLM-independent merging approach. Subsequently, the binary merge function is extended to handle the entire set of atomic TKGs through iterative pairwise merging in parallel until a single consolidated TKG is obtained (Algorithm A.2 in the Appendices). Atomic TKGs are organized into pairs, with each pair merged in parallel. If the number of TKGs is odd, the remaining TKG carries forward to the next iteration. This process continues iteratively, reducing the number of TKGs at each step, until convergence to a single merged TKG. This parallel strategy scales with the number of available threads and addresses the computational challenge from Module-1, enabling ATOM to maintain low latency while preserving the exhaustivity and stability benefits of atomic decomposition. After the merge of all atomic TKGs, the snapshot 𝒢st\mathcal{G}^{t}_{s} is obtained, and it is merged with the previous DTKG 𝒢t−1\mathcal{G}^{t-1} using Algorithm A.1 to yield the DTKG updated at tt, 𝒢t\mathcal{G}^{t}.

##  4 Experiments

Our evaluation addresses the following research questions:

  1. RQ1:

How does exhaustivity deteriorate as the LLM context increases, and what degree of information loss could occur?

  2. RQ2:

How does ATOM’s atomic fact decomposition ensure stability, exhaustivity, and improve the quality of the 5-tuples?

  3. RQ3:

How does ATOM scale with the number of atomic facts provided as input, and what is its time complexity compared to baseline methods?

  4. RQ4:

How does ATOM perform on DTKG construction consistency compared to baseline methods?




###  4.1 Metrics

To assess (C1), we adopt the metrics Hallucination (HALL), Omission (OM), and Match (MATCH) introduced by Ghanem and Cruz (2024). Given a gold-standard KG, MATCH denotes correctly extracted triplets, OM refers to triplets that exist in the source but are missing from the extracted KG, and HALL represents unsupported triplets. These definitions are extended from factual triplets (es,rp,eo)(e_{s},r_{p},e_{o}) to temporal 5-tuples (es,rp,eo,ts​t​a​r​t,te​n​d)(e_{s},r_{p},e_{o},t_{start},t_{end}). A 5-tuple may exhibit correctness at the factual level while introducing temporal hallucination or omission. Hence, for 5-tuples whose factual components match the gold standard, the following definitions are established:

  * •

MATCHt\text{MATCH}_{t}: 5-tuples whose (ts​t​a​r​t,te​n​d)(t_{start},t_{end}) is also correct.

  * •

HALLt\text{HALL}_{t}: 5-tuples containing temporal values not present in the gold standard.

  * •

OMt\text{OM}_{t}: 5-tuples whose the (ts​t​a​r​t,te​n​d)(t_{start},t_{end}) are in the gold standard but are not reproduced in the extraction.




##### Exhaustivity (RQ1 & RQ2).

The exhaustivity of the factual component is measured as:

| RMATCH=|MATCH||MATCH|+|OM|R_{\text{MATCH}}=\frac{|\text{MATCH}|}{|\text{MATCH}|+|\text{OM}|} |  | (6)  
---|---|---|---  
  
The exhaustivity of the temporal component RMATCHtR_{\text{MATCH}_{t}} is computed analogously with |MATCHt||\text{MATCH}_{t}| in the numerator.

##### 5-tuples Quality (RQ2).

It is assessed through the factual/temporal hallucination and omission rates:

| RHALL=|HALL||MATCH|+|HALL|R_{\text{HALL}}=\frac{|\text{HALL}|}{|\text{MATCH}|+|\text{HALL}|} |  | (7)  
---|---|---|---  
| ROM=|OM||MATCH|+|OM|R_{\text{OM}}=\frac{|\text{OM}|}{|\text{MATCH}|+|\text{OM}|} |  | (8)  
---|---|---|---  
  
The ROMtR_{\text{OM}_{t}} is computed similarly to ROMR_{\text{OM}} with |OMt||\text{OM}_{t}| in the numerator. And RHALLtR_{\text{HALL}_{t}} is simply RMATCHR_{\text{MATCH}} \- RMATCHtR_{\text{MATCH}_{t}} \- ROMtR_{\text{OM}_{t}}.

##### Stability (RQ2).

It is measured using the cosine similarity between the centroid embeddings of 5-tuple sets obtained across independent runs. Let 𝒄(1)\boldsymbol{c}^{(1)} denote the centroid vector obtained during the baseline run (RUN 1) and let 𝒄(r)\boldsymbol{c}^{(r)} denote the centroid obtained at repetition rr. Formally, the stability score is computed as:

| Sr=cos⁡(𝒄(1),𝒄(r))=⟨𝒄(1),𝒄(r)⟩∥𝒄(1)∥​∥𝒄(r)∥S_{r}=\cos\bigl(\boldsymbol{c}^{(1)},\,\boldsymbol{c}^{(r)}\bigr)=\frac{\bigl\langle\boldsymbol{c}^{(1)},\,\boldsymbol{c}^{(r)}\bigr\rangle}{\lVert\boldsymbol{c}^{(1)}\rVert\,\lVert\boldsymbol{c}^{(r)}\rVert} |  | (9)  
---|---|---|---  
  
This score is computed for r=2,3r=2,3.

##### Time complexity (RQ3).

Time complexity is evaluated by progressively increasing the number of atomic facts provided as input and measuring the total wall-clock latency required to construct the complete DTKG.

##### DTKG consistency (RQ4).

For entity/relation resolution, the false discovery rate (1-precision) is defined in (Lairgi et al., 2024). Since they overlook recall and F1-score, we extend the evaluation to include precision, recall, and F1-score for both entity resolution (ER) and relation resolution (RR), denoted as Metric-ER and Metric-RR, respectively. For temporal resolution, qualitative comparison is provided.

###  4.2 Datasets and baseline methods

DocRed (Yao et al., 2019) is unsuitable for temporal extraction due to inconsistencies (Tan et al., 2022), and TempDocRed (Zhu et al., 2025), though temporally enriched, focuses only on event start dates and named entities. CS-GS and Music-GS (Kabal et al., 2024), also used in iText2KG evaluations, lack temporal data. Therefore, the NYT News dynamic and temporal dataset (Singh, 2023), containing lead paragraphs from two million news articles since 2000, is adopted. From this, the 2020-COVID-NYT subset comprising 1,076 articles that focus on COVID-19 dynamics during 2020 is extracted. This subset is enriched with human-verified atomic facts and 5-tuples. Publication dates are used as observation dates (details on observation time modeling in Section D in the Appendices). More information about this dataset is provided in Table T.1 in the Appendices. To the best of our knowledge, an approach for resolving duplicate entities and relations while maintaining KG consistency among the SOTA methods for zero- and few-shot KG construction is supported only by iText2KG and Graphiti. The temporal aspect is handled by Graphiti only. Hence, it is the primary comparator of ATOM. In all experiments, the temperature is set to 0 to reduce hallucinations. text-embedding-large-3444<https://platform.openai.com/docs/models/text-embedding-3-large> is used for embeddings. θE=0.8\theta_{E}=0.8 and θR=0.7\theta_{R}=0.7 are estimated following Lairgi et al. (2024) (details are in Section C in the Appendices).

###  4.3 Exhaustivity deterioration in longer contexts

Exhaustivity is evaluated by iteratively concatenating lead paragraphs (increasing context size) and testing five SOTA LLMs claude-sonnet-4-2025-01-31555<https://docs.claude.com/en/docs/about-claude/models/overview>, gpt-4o-2024-11-20666<https://platform.openai.com/docs/models/gpt-4o-2024-11-20>, mistral-large-2411777<https://docs.mistral.ai/getting-started/models/models_overview/>, gpt-4.1-2025-04-14888<https://openai.com/index/gpt-4-1/>, o3-mini-2025-01-31999<https://openai.com/index/openai-o3-mini/>. Figure 2 shows a clear "forgetting effect". A decreased factual and temporal exhaustivity as token count increases across all models except claude-Sonnet-4-2025-01-31, which maintains the highest exhaustivity for atomic facts but degrades for 5-tuples. This indicates that LLMs prioritize salient facts in longer texts. Moreover, all models show higher exhaustivity for atomic fact decomposition than 5-tuple extraction. Atomic facts require surface-level parsing, while 5-tuples demand deeper semantic understanding of entities, relationship identification, and temporal extraction (ts​t​a​r​tt_{start}, te​n​dt_{end}). This added complexity causes greater information loss. To mitigate information loss in the atomic fact decomposition, we empirically determine the optimal chunk size at <400<400 tokens to keep the exhaustivity >0.8>0.8. For subsequent evaluations, we use claude-sonnet-4-2025-01-31 for atomic fact decomposition based on its superior performance. For 5-tuple extraction, both gpt-4.1-2025-04-14 and claude-sonnet-4-2025-01-31 perform comparably, hence gpt-4.1-2025-04-14 is used due to its lower cost. Section 4.5 evaluates exhaustivity gains from using atomic facts as input.

Figure 2:  Exhaustivity vs. token count as context for (a) the atomic fact decomposition (b) 5-tuples extraction.

###  4.4 TKG stability

Rerunning the same prompt multiple times leads to variations in output. To evaluate the stability score, 5-tuples are initially extracted from both atomic facts (denoted as (F) in Table 1) and lead paragraphs individually, establishing a baseline named Run 1. This extraction process is subsequently repeated twice more without altering any parameters. As shown in Table 1, centroids of the extracted 5-tuples’ embeddings from atomic facts remain nearly constant across runs 1 and 2, exhibiting very low standard deviations and high stability scores. In contrast, the centroids of the extracted 5-tuples’ embeddings from the lead paragraphs demonstrate greater variability. This addresses (C2) and reflects the effect of atomic facts in maintaining a stable construction of TKGs with a gain of ∼17%\sim 17\%.

Table 1: Stability SrS_{r} evaluated by rerunning the 5-tuples extraction process multiple times without any modifications, using gpt-4.1-2025-04-14 with Run 1 as a baseline. The extraction is performed on (F) atomic facts and (L) lead paragraphs. Dataset | Run 2 | Run 3  
---|---|---  
2020-COVID-NYT (F) | 0.944 ±\pm 0.024 | 0.943 ±\pm 0.025  
2020-COVID-NYT (L) | 0.776 ±\pm 0.214 | 0.773 ±\pm 0.214  
  
###  4.5 The exhaustivity and quality of the 5-tuples

5-tuples are extracted from both atomic facts (denoted as (F) in Table 2) and lead paragraphs (L). As shown in Table 2, higher values are maintained by the RMATCHR_{\text{MATCH}} and RMATCHtR_{\text{MATCH}_{t}} rates when atomic facts are provided as input compared to when lead paragraphs are used, with a gain of ∼18%\sim 18\% achieved on temporal exhaustivity and ∼31%\sim 31\% on factual exhaustivity, along with a gain of approximately ∼31%\sim 31\% on factual omission. Hence, atomic fact decomposition improves temporal and factual exhaustivity, and this addresses (C1). However, an increase in RHALLR_{\text{HALL}} by ∼9%\sim 9\% is observed, which is attributed to the addition of ’inferred’ atomic facts from the lead paragraphs by the LLM during the atomic fact decomposition. Consequently, the LLM extracts 5-tuples that do not exist in the gold standard. Furthermore, a higher ROMtR_{\text{OM}_{t}} is observed, which is attributed to imperfections that may occur during atomic fact decomposition, where temporal information may not be assigned to atomic facts by the LLM. This error propagates to 5-tuples extraction in ROMtR_{\text{OM}_{t}}. It is a trade-off that is discussed further in Section 6.

Table 2: 5-tuple quality metrics. The extraction is performed on (F) atomic facts and (L) lead paragraphs.

Metric | 2020-COVID-NYT (L) | 2020-COVID-NYT (F)  
---|---|---  
RMATCHR_{\text{MATCH}} | 0.405 ±\pm 0.150 | 0.720 ±\pm 0.143  
RMATCHtR_{\text{MATCH}_{t}} | 0.176 ±\pm 0.123 | 0.354 ±\pm 0.165  
ROMtR_{\text{OM}_{t}} | 0.229 ±\pm 0.131 | 0.366 ±\pm 0.158  
RHALLtR_{\text{HALL}_{t}} | 0.000 ±\pm 0.000 | 0.000 ±\pm 0.000  
ROMR_{\text{OM}} | 0.595 ±\pm 0.150 | 0.280 ±\pm 0.143  
RHALLR_{\text{HALL}} | 0.333 ±\pm 0.172 | 0.428 ±\pm 0.128  
  
###  4.6 ATOM’s time complexity

Given the demonstrated benefits of atomic fact decomposition in improving exhaustivity and stability, all subsequent experiments utilize atomic facts as input rather than lead paragraphs. All baseline methods are run using gpt-4.1-2025-04-14. ATOM employs 8 threads and a batch size of 40 atomic facts for 5-tuples extraction, which respects OpenAI rate limits. iText2KG and Graphiti separate entity and relation extraction, increasing latency. Graphiti’s incremental entity/relation resolution, which relies on the LLM, limits parallel requests and significantly increases latency as the graph expands. Similarly, iText2KG is incremental, restricting parallel requests. Although iText2KG uses a distance metric for resolution, reducing LLM dependency, its separate extraction steps double the number of LLM calls and induce isolated entities that require further LLM iterations. Conversely, ATOM’s architecture facilitates (1) parallel LLM calls, (2) parallel merge of atomic TKGs, (3) LLM-independent merging, and (4) temporal resolution. This design reduces latency by 93.8% compared to Graphiti and 95.3% compared to iText2KG (Figure 3). ATOM’s Module-3 accounts for only 13% of its total latency, with the remainder attributed to API calls, which can be further minimized through either increasing the batch size (by upgrading the API tier) or scaling hardware for local LLM deployment.

Figure 3:  Latency comparison of the baseline methods as a function of the number of atomic facts as input.

###  4.7 ATOM’s DTKG construction

Table 3 shows that ATOM and iText2KG demonstrate comparable entity and relation resolution performance, as both employ a distance metric for merging. ATOM shows an improvement over Graphiti, whose incremental, LLM-based entity and relation resolution degrades with graph expansion (increasing context size), which is consistent with findings in Section 4.3. Beyond entity and relation resolution, temporal resolution reveals more significant differences between ATOM and Graphiti. The examples in Figures F.3 and F.4 in the Appendices illustrate atomic facts observed at different times that refer to the same information but with different validity periods. In these examples, Graphiti creates separate relations for 5-tuples with different validity periods (ts​t​a​r​tt_{start}, te​n​dt_{end}), while ATOM detects these similar relations and extends their validity period history. This temporal resolution serves two functions: tracking events that naturally appear and disappear over time, and matching relations with only ts​t​a​r​tt_{start} or only te​n​dt_{end} to complete their validity periods as additional information becomes available. Additionally, Graphiti incorporates validity periods only and does not allow for observation time modeling, treating observation time as ts​t​a​r​tt_{start}. This can induce errors. For example, if a news article observed on "January 23, 2020" states "The mysterious respiratory virus spread to at least 10 other countries," Graphiti would set ts​t​a​r​t=23​-​01​-​2020t_{start}=23\text{-}01\text{-}2020, while the statement does not specify a validity period and the true validity could be weeks or days before its publication. In contrast, ATOM separately models observation time and validity periods, allowing it to recognize atomic facts without a validity period and avoid incorrect temporal assignments (Example in Figure F.2 in the Appendices).

Table 3: Performance on DTKG construction. Metric | ATOM | Graphiti | iText2KG  
---|---|---|---  
Precision-ER | 0.994 | 0.967 | 0.974  
Recall-ER | 0.993 | 0.952 | 0.980  
F1-Score-ER | 0.994 | 0.959 | 0.977  
Precision-RR | 1 | 0.917 | 0.991  
Recall-RR | 1 | 0.888 | 0.988  
F1-Score-RR | 1 | 0.902 | 0.989  
  
##  5 Conclusion

In this paper, we presented ATOM, a few-shot and scalable approach for constructing and dynamically updating TKGs from unstructured texts. Experimental results indicate that ATOM’s atomic fact decomposition effectively addresses the exhaustivity and stability challenges often observed in LLM-based TKG construction methods. Its parallel architecture accelerates TKG construction and enables scalability for larger unstructured texts. Potential directions for future work include quantitative evaluation of temporal resolution and fine-tuning an LLM specifically for refining atomic fact decomposition. In summary, ATOM enables fast and continuous updates of TKGs.

##  6 Limitations

ATOM has some limitations that warrant consideration. First, the atomic fact decomposition can introduce error propagation: the LLM may generate inferred facts not present in the source text, leading to increased hallucination rates, and may fail to properly assign temporal information to atomic facts, resulting in temporal omissions in the extracted 5-tuples (Section 4.5). A potential improvement consists of fine-tuning an LLM model specifically for this task. Second, the distance-metric-based merging approach, while scalable and efficient, can occasionally merge semantically distinct named entities that exhibit high similarity (e.g., gpt-5:model and gpt-3.5:model). A supervised entity/relation resolution classifier trained on labeled entity pairs could replace the threshold-based approach.

## References

  * Al-Moslmi et al. (2020) Tareq Al-Moslmi, Marc Gallofré Ocaña, Andreas L. Opdahl, and Csaba Veres. 2020.  [Named entity extraction for knowledge graphs: A literature overview](https://doi.org/10.1109/ACCESS.2020.2973928).  _IEEE Access_ , 8:32862–32881. 
  * Anokhin et al. (2024) Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry Evseev, Mikhail Burtsev, and Evgeny Burnaev. 2024.  Arigraph: Learning knowledge graph world models with episodic memory for llm agents.  _arXiv preprint arXiv:2407.04363_. 
  * Atil et al. (2024) Berk Atil, Sarp Aykent, Alexa Chittams, Lisheng Fu, Rebecca J Passonneau, Evan Radcliffe, Guru Rajan Rajagopal, Adam Sloan, Tomasz Tudrej, Ferhan Ture, and 1 others. 2024.  Non-determinism of" deterministic" llm settings.  _arXiv preprint arXiv:2408.04667_. 
  * Cai et al. (2024) Li Cai, Xin Mao, Yuhao Zhou, Zhaoguang Long, Changxu Wu, and Man Lan. 2024.  A survey on temporal knowledge graph: Representation learning and applications.  _arXiv preprint arXiv:2403.04782_. 
  * Carta et al. (2023) Salvatore Carta, Alessandro Giuliani, Leonardo Piano, Alessandro Sebastian Podda, Livio Pompianu, and Sandro Gabriele Tiddia. 2023.  Iterative zero-shot LLM prompting for knowledge graph construction.  _arXiv preprint arXiv:2307.01128_. 
  * Cetera et al. (2022) Wiesław Cetera, Włodzimierz Gogołek, Aleksander Żołnierski, and Dariusz Jaruga. 2022.  Potential for the use of large unstructured data resources by public innovation support institutions.  _Journal of Big Data_ , 9(1):46. 
  * Chekol and Stuckenschmidt (2018) Melisachew Wudage Chekol and Heiner Stuckenschmidt. 2018.  Towards probabilistic bitemporal knowledge graphs.  In _Companion Proceedings of the The Web Conference 2018_ , pages 1757–1762. 
  * Chen et al. (2023) Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2023.  Dense x retrieval: What retrieval granularity should we use?  _arXiv preprint arXiv:2312.06648_. 
  * Dresp-Langley et al. (2019) Birgitta Dresp-Langley, Ole Kristian Ekseth, Jan Fesl, Seiichi Gohshi, Marc Kurz, and Hans-Werner Sehring. 2019.  Occam’s razor for big data? on detecting quality in large unstructured datasets.  _Applied Sciences_ , 9(15):3065. 
  * Edge et al. (2024) Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. 2024.  From local to global: A graph rag approach to query-focused summarization.  _arXiv preprint arXiv:2404.16130_. 
  * Ghanem and Cruz (2024) Hussam Ghanem and Christophe Cruz. 2024.  Enhancing knowledge graph construction: Evaluating with emphasis on hallucination, omission, and graph similarity metrics.  In _International Knowledge Graph and Semantic Web Conference_ , pages 32–46. Springer. 
  * Hosseini et al. (2024) Mohammad Javad Hosseini, Yang Gao, Tim Baumgärtner, Alex Fabrikant, and Reinald Kim Amplayo. 2024.  Scalable and domain-general abstractive proposition segmentation.  _arXiv preprint arXiv:2406.19803_. 
  * Hu et al. (2023) Yuelin Hu, Futai Zou, Jiajia Han, Xin Sun, and Yilei Wang. 2023.  LLM-Tikg: Threat intelligence knowledge graph construction utilizing large language model.  _Available at SSRN 4671345_. 
  * Jiang et al. (2023) Xuhui Jiang, Chengjin Xu, Yinghan Shen, Xun Sun, Lumingyuan Tang, Saizhuo Wang, Zhongwu Chen, Yuanzhuo Wang, and Jian Guo. 2023.  On the evolution of knowledge graphs: A survey and perspective.  _arXiv preprint arXiv:2310.04835_. 
  * Jin et al. (2023) Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. 2023.  Large language models on graphs: A comprehensive survey.  _arXiv preprint arXiv:2312.02783_. 
  * Kabal et al. (2024) Othmane Kabal, Mounira Harzallah, Fabrice Guillet, and Ryutaro Ichise. 2024.  Enhancing domain-independent knowledge graph construction through openie cleaning and llms validation.  _Procedia Computer Science_ , 246:2617–2626. 
  * Lairgi et al. (2024) Yassir Lairgi, Ludovic Moncla, Rémy Cazabet, Khalid Benabdeslem, and Pierre Cléau. 2024.  itext2kg: Incremental knowledge graphs construction using large language models.  In _International Conference on Web Information Systems Engineering_ , pages 214–229. Springer. 
  * Liu et al. (2023) Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023.  Lost in the middle: How language models use long contexts.  _arXiv preprint arXiv:2307.03172_. 
  * Meijer (2022) Lisa Meijer. 2022.  Bi-vaks: Bi-temporal versioning approach for knowledge graphs.  _Delft University of Technology_. 
  * Raina and Gales (2024) Vatsal Raina and Mark Gales. 2024.  Question-based retrieval using atomic units for enterprise rag.  _arXiv preprint arXiv:2405.12363_. 
  * Rasmussen et al. (2025) Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. 2025.  Zep: A temporal knowledge graph architecture for agent memory.  _arXiv preprint arXiv:2501.13956_. 
  * Singh (2023) Aryan Singh. 2023.  NYT Articles: 2.1M+ (2000-Present) Daily Updated.  <https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present>.  Accessed: 2025-06-01. 
  * Tan et al. (2022) Qingyu Tan, Lu Xu, Lidong Bing, Hwee Tou Ng, and Sharifah Mahani Aljunied. 2022.  Revisiting docred–addressing the false negative problem in relation extraction.  _arXiv preprint arXiv:2205.12696_. 
  * Trugenberger (2015) Carlo A Trugenberger. 2015.  Scientific discovery by machine intelligence: A new avenue for drug research.  _arXiv preprint arXiv:1506.07116_. 
  * Wu et al. (2024) Yuxia Wu, Yuan Fang, and Lizi Liao. 2024.  Retrieval augmented generation for dynamic graph modeling.  _arXiv preprint arXiv:2408.14523_. 
  * Xi et al. (2023) Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, and 1 others. 2023.  The rise and potential of large language model based agents: A survey.  _arXiv preprint arXiv:2309.07864_. 
  * Yao et al. (2019) Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin, Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou, and Maosong Sun. 2019.  Docred: A large-scale document-level relation extraction dataset.  _arXiv preprint arXiv:1906.06127_. 
  * Zhang et al. (2024) Yongheng Zhang, Tingwen Du, Yunshan Ma, Xiang Wang, Yi Xie, Guozheng Yang, Yuliang Lu, and Ee-Chien Chang. 2024.  AttacKG+: Boosting attack knowledge graph construction with large language models.  _arXiv preprint arXiv:2405.04753_. 
  * Zhong et al. (2023) Lingfeng Zhong, Jia Wu, Qian Li, Hao Peng, and Xindong Wu. 2023.  A comprehensive survey on automatic knowledge graph construction.  _ACM Computing Surveys_ , 56(4):1–62. 
  * Zhu et al. (2025) Jun Zhu, Yan Fu, Junlin Zhou, and Duanbing Chen. 2025.  A temporal knowledge graph generation dataset supervised distantly by large language models.  _Scientific Data_ , 12(1):734. 



##  Appendix A ATOM’s Algorithms

ATOM’s framework is based on two main algorithms, presented below: Algorithm A.1 for merging pairs of TKGs and Algorithm A.2 for parallelizing the merge of lists of TKGs.

Algorithm A.1 Binary Merge of TKGs

1:function BinaryMerge(T​K​G1=(ℰ1,ℛ1)TKG_{1}=(\mathcal{E}_{1},\mathcal{R}_{1}), T​K​G2=(ℰ2,ℛ2)TKG_{2}=(\mathcal{E}_{2},\mathcal{R}_{2}), θE\theta_{E}, θR\theta_{R}) 

2: // —— Entity Resolution —— //

3: Initialize mapping M←∅M\leftarrow\emptyset

4: for all entity e∈ℰ1e\in\mathcal{E}_{1} do

5: if there exists e′∈ℰ2e^{\prime}\in\mathcal{E}_{2} such that e.name=e′.namee.\text{name}=e^{\prime}.\text{name} and e.label=e′.labele.\text{label}=e^{\prime}.\text{label} then

6: M​(e)←e′M(e)\leftarrow e^{\prime}

7: else

8: Compute s←maxe′∈ℰ2cos(e.𝐯,e′.𝐯)s\leftarrow\max\limits_{e^{\prime}\in\mathcal{E}_{2}}\cos\Big(e.\mathbf{v},e^{\prime}.\mathbf{v}\Big) ⊳\triangleright Cosine similarity of e and e’ embeddings 

9: Let e∗←argmaxe′∈ℰ2cos(e.𝐯,e′.𝐯)e^{*}\leftarrow\arg\max\limits_{e^{\prime}\in\mathcal{E}_{2}}\cos\Big(e.\mathbf{v},e^{\prime}.\mathbf{v}\Big)

10: if s≥θEs\geq\theta_{E} then

11: M​(e)←e∗M(e)\leftarrow e^{*}

12: else

13: M​(e)←eM(e)\leftarrow e

14: end if

15: end if

16: end for

17: ℰmerged←ℰ2∪{e∣M​(e)∉ℰ2}\mathcal{E}_{\text{merged}}\leftarrow\mathcal{E}_{2}\cup\\{e\mid M(e)\notin\mathcal{E}_{2}\\}

18: // —— Relation’s Name Resolution —— //

19: ℛ1updated←∅\mathcal{R}_{1}^{\text{updated}}\leftarrow\emptyset

20: for all relation r∈ℛ1r\in\mathcal{R}_{1} do

21: Update endpoints: r.startEntity←M(r.startEntity)r.\text{startEntity}\leftarrow M(r.\text{startEntity}), r.endEntity←M(r.endEntity)r.\text{endEntity}\leftarrow M(r.\text{endEntity})

22: Compute sr←maxr′∈ℛ2cos(r.𝐯,r′.𝐯)s_{r}\leftarrow\max\limits_{r^{\prime}\in\mathcal{R}_{2}}\cos\Big(r.\mathbf{v},r^{\prime}.\mathbf{v}\Big) ⊳\triangleright Cosine similarity of r and r’ names embeddings 

23: Let r∗←argmaxr′∈ℛ2cos(r.𝐯,r′.𝐯)r^{*}\leftarrow\arg\max\limits_{r^{\prime}\in\mathcal{R}_{2}}\cos\Big(r.\mathbf{v},r^{\prime}.\mathbf{v}\Big)

24: if sr≥θRs_{r}\geq\theta_{R} then

25: Update names: Update r.name←r∗.namer.\text{name}\leftarrow r^{*}.\text{name}

26: end if// —— Temporal Resolution —— //

27: if there exists r′∈ℛ2r^{\prime}\in\mathcal{R}_{2} such that rr is similar to r′r^{\prime} then // For similar relations, their times are merged

28: Update start time: r′.ts​t​a​r​t←r′.ts​t​a​r​t∪r.ts​t​a​r​tr^{\prime}.t_{start}\leftarrow r^{\prime}.t_{start}\cup r.t_{start}

29: Update end time: r′.te​n​d←r′.te​n​d∪r.te​n​dr^{\prime}.t_{end}\leftarrow r^{\prime}.t_{end}\cup r.t_{end}

30: Update observation time: r′.to​b​s←r′.to​b​s∪r.to​b​sr^{\prime}.t_{obs}\leftarrow r^{\prime}.t_{obs}\cup r.t_{obs}

31: end if

32: ℛ1updated←ℛ1updated∪{r}\mathcal{R}_{1}^{\text{updated}}\leftarrow\mathcal{R}_{1}^{\text{updated}}\cup\\{r\\}

33: end for

34: ℛmerged←ℛ2∪ℛ1updated\mathcal{R}_{\text{merged}}\leftarrow\mathcal{R}_{2}\cup\mathcal{R}_{1}^{\text{updated}}

35: return (ℰmerged,ℛmerged)(\mathcal{E}_{\text{merged}},\mathcal{R}_{\text{merged}})

36:end function

Algorithm A.2 Parallel Merge of TKGs

1:function ParallelMerge(T​K​G​s,θE,θRTKGs,\,\theta_{E},\,\theta_{R}) 

2: Input: A list of temporal knowledge graphs  | T​K​G​s={T​K​G1,T​K​G2,…,T​K​Gn}TKGs=\\{TKG_{1},TKG_{2},\ldots,TKG_{n}\\} |   
---|---|---  
  
3: c​u​r​r​e​n​t←T​K​G​scurrent\leftarrow TKGs

4: while |c​u​r​r​e​n​t|>1|current|>1 do

5: m​e​r​g​e​d​R​e​s​u​l​t​s←∅mergedResults\leftarrow\emptyset

6: Let n←|c​u​r​r​e​n​t|n\leftarrow|current|

7: Form pairs: | p​a​i​r​s←{(c​u​r​r​e​n​t​[2​i],c​u​r​r​e​n​t​[2​i+1])∣0≤i<⌊n/2⌋}pairs\leftarrow\\{\,(current[2i],\,current[2i+1])\mid 0\leq i<\lfloor n/2\rfloor\,\\} |   
---|---|---  
  
8: if nn is odd then

9: l​e​f​t​o​v​e​r←c​u​r​r​e​n​t​[n−1]leftover\leftarrow current[n-1]

10: else

11: l​e​f​t​o​v​e​r←nullleftover\leftarrow\text{null}

12: end if

13: for all each pair (T​K​Ga,T​K​Gb)(TKG_{a},\,TKG_{b}) in p​a​i​r​spairs in parallel do

14: m​e​r​g​e​d←BinaryMerge​(T​K​Ga,T​K​Gb,θE,θR)merged\leftarrow\textbf{{BinaryMerge}}(TKG_{a},\,TKG_{b},\,\theta_{E},\,\theta_{R})

15: Add m​e​r​g​e​dmerged to m​e​r​g​e​d​R​e​s​u​l​t​smergedResults

16: end for

17: if l​e​f​t​o​v​e​r≠nullleftover\neq\text{null} then

18: Add l​e​f​t​o​v​e​rleftover to m​e​r​g​e​d​R​e​s​u​l​t​smergedResults

19: end if

20: c​u​r​r​e​n​t←m​e​r​g​e​d​R​e​s​u​l​t​scurrent\leftarrow mergedResults

21: end while

22: return c​u​r​r​e​n​t​[0]current[0]

23:end function

##  Appendix B Example of the atomic fact decomposition

Example: _(Observed in 01-01-2025) On June 18, 2024, Real Madrid won the Champions League final with a 2-1 victory. Following the triumph, fans of Real Madrid celebrated the Champions League victory across the city._

  * •

_Real Madrid won the Champions League final match on June 18, 2024._ (Observation to​b​s=[01−01−2025]t_{obs}=[01-01-2025], ts​t​a​r​t=[18−06−2024]t_{start}=[18-06-2024], te​n​d=[.]t_{end}=[.])

  * •

_The Champions League final match ended with a 2-1 victory for Real Madrid on June 18, 2024._ (Observation to​b​s=[01−01−2025]t_{obs}=[01-01-2025], ts​t​a​r​t=[18−06−2024]t_{start}=[18-06-2024], te​n​d=[.]t_{end}=[.])

  * •

_Fans of Real Madrid celebrated the Champions League final match victory across the city on June 18, 2024._ (Observation to​b​s=[01−01−2025]t_{obs}=[01-01-2025], ts​t​a​r​t=[18−06−2024]t_{start}=[18-06-2024], te​n​d=[.]t_{end}=[.])




##  Appendix C Estimating the merging thresholds

The merging thresholds θR\theta_{R} and θE\theta_{E} were estimated by (Lairgi et al., 2024), based on the mean cosine similarity of 1,500 pairs of similar entities and relation names generated by gpt-4-0613101010<https://platform.openai.com/docs/models/gpt-4>; however, because entity typology is not considered, a hybrid similarity measure is proposed, combining the entity name embedding and the entity label embedding as: λ⋅embeddingsname+β⋅embeddingslabel\lambda\,\cdot\,\text{embeddings}_{\text{name}}+\beta\,\cdot\,\text{embeddings}_{\text{label}}. Using this measure, 1,200 pairs of similar entities incorporating typology were generated, and λ\lambda is optimized to maximize the resulting cosine similarity, after which it is determined that λ=0.8\lambda=0.8, β=0.2\beta=0.2, and θE=0.8\theta_{E}=0.8, while θR=0.7\theta_{R}=0.7 is retained as previously estimated in (Lairgi et al., 2024).

Figure F.1:  Example overview of ATOM’s pipeline. It begins with atomic fact decomposition, followed by the extraction of atomic TKGs from these facts, which are then merged in parallel. When an incoming update arrives, ATOM handles the temporal resolution by transforming the end action into the affirmative part while modifying only the te​n​dt_{end}, then merges the resulting atomic TKG with the existing DTKG.  Table T.1: 2020-COVID-NYT Statistics Analysis. We use lead paragraphs as they encapsulate the article’s key facts, while the full article text is often unavailable in research datasets due to licensing restrictions and would introduce unnecessary verbosity without proportional information gain. Metric | Value  
---|---  
Basic Dataset Information  
Total Articles | 1,076  
Grouped Articles (by pub. date) | 274  
Average Tokens per Group | 206 ± 156  
Date Range | 2020-01-09 to 2020-12-30  
Atomic Facts Analysis  
Total atomic facts | 4,223  
Atomic facts with validity time | 2,037  
Atomic facts without validity time | 2,186  
Knowledge Graph Structure  
Total 5-tuples | 7,210  
Number of atomic TKGs | 4,223  
Avg number of 5-tuples per atomic TKG | ∼2\sim 2  
  
##  Appendix D Observation time modeling

Modeling the observation time is essential both for capturing the dynamism of the TKG and for inferring relative times. For historical or retrospective unstructured text streams (e.g., past news articles or archive documents), the observation time should correspond to the original publication time rather than the time at which the document was processed and ingested into the DTKG. This distinction is essential for preserving the correct temporal ordering of events and enabling reliable inference of relative times.

Conversely, for prospective or continuously monitored sources, where data is ingested automatically as it becomes available, the observation time can be treated as the ingestion time. In such settings, ingestion reflects the earliest feasible moment at which the information could be known by the system.

The granularity of observation time is application-dependent and may be defined according to user requirements. For instance, COVID-19 news was simulated using daily observation snapshots, whereas social media streams may require a per-post snapshot.

Figure F.2:  Two DTKGs constructed using ATOM and Graphiti from 09-01-2020 (in UNIX, 1578524400) to 23-01-2020 (in UNIX, 1579734000) from 2020-COVID-NYT dataset. Left (ATOM): Preserves observation times (to​b​st_{obs}) separately from validity periods, with timestamps encoded in UNIX format to eliminate overhead associated with string parsing operations and timezone conversion calculations. Right (Graphiti): Treats observation time as validity start time. v​a​l​i​d​_​a​tvalid\\_at corresponds to ts​t​a​r​tt_{start} in Graphiti’s time modeling. The highlighted fact “The mysterious respiratory virus spread to at least 10 other countries” is observed on 23-01-2020, but this does not guarantee the spread occurred at that time. ATOM’s dual-time modeling prevents such temporal misattribution.  Figure F.3: Temporal resolution comparison between ATOM and Graphiti. Two atomic facts observed on January 28, 2020, report death counts from January 24 (26 deaths) and January 27 (at least 80 deaths). Left (ATOM): performs temporal resolution by detecting similar relations and extending their validity period history (te​n​dt_{end} in the figure). Right (Graphiti): creates separate relations for each atomic fact, resulting in duplication. Moreover, Graphiti misinterprets “By January 24, 2020” and “By January 27, 2020” as validity start times rather than validity end times, leading to temporal misattribution.  Figure F.4: Temporal resolution comparison between ATOM and Graphiti. Two atomic facts observed on different dates (April 16 and April 19, 2020) describe protest activities during two time periods (“the week of April 13” and “the week of April 19”). Left (ATOM): merges similar relations and extend their validity periods (ts​t​a​r​tt_{start} and te​n​dt_{end} in the figure). Right (Graphiti): maintains separate relations for each atomic fact. Moreover, Graphiti fails to translate “In the week of April 13, 2020” and “In the week of April 19, 2020” into proper validity periods as ATOM does.

[◄](/html/2510.22589) [](/) [Feeling  
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion  
report](/log/2510.22590) [Report  
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2510.22590) [View original  
on arXiv](https://arxiv.org/abs/2510.22590)[►](/html/2510.22591)

[](javascript:toggleColorScheme\(\) "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Wed Nov 5 17:29:33 2025 by [LaTeXML](http://dlmf.nist.gov/LaTeXML/)
