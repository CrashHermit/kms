in the Appendix provides details.

HippoRAG 1&2 Settings In our implementation of HippoRAG (Gutiérrez et al., 2024), we extend the original framework to operate on a customized graph. Initially presented in the foundational paper, we employ Named Entity Recognition (NER) to build a personalized dictionary for PageRank execution. Regarding HippoRAG2 (Gutiérrez et al., 2025), we select the top 30 edges (musique dataset 50 edges) for LLM filtering, incorporating a weight adjustment factor of 0.9. Considering the capability of our graph to effectively locating subgraphs, combined with various graph configurations (entity, event, concept) resulting in graphs of differing densities, we set the damping factor to 0.9 to concentrate on propagation within the local subgraph. For further implementation details, please refer to Algorithm 2.

Implementation Details The knowledge graph is constructed from the corresponding context corpora for each dataset following (Gutiérrez et al., 2024) using the framework of AutoSchemaKG with \( L_{max} = 1024 \) and \( B = 16 \), and the schema induction pipeline (Section B.2) with \( B_s = 5 \). We employ Meta's LLaMA-3.1-8B-Instruct to construct the graphs, optimized with bfloat16 precision and Flash Attention 2. The graph is stored in NetworkX for retrieval, with subgraphs fed into LLaMA-3.3-70B-Instruct for answer generation.

Evaluation Results The experimental results in Figure 5 and Figure 8 demonstrate AutoSchemaKG's effectiveness in multi-hop question answering across three benchmark datasets. With HippoRAG2 integration, the Full-KG configuration (entities, events, and concepts) outperforms traditional retrieval approaches like BM25 and Contriever by 12-18%, highlighting its strength in complex reasoning scenarios. Notably, AutoSchemaKG achieves comparable or better results using LLaMA-3.1-8B-Instruct as graph constructor compared to the original HippoRAG2 implementation that requires LLaMA-3.3-70B-Instruct for both construction and QA reading.

Advantages of Events and Concepts Our case studies revealed two key benefits of event and concept nodes: 1) Event nodes provide enriched context. As shown in Figure 9, they serve as valuable retrieval targets when critical information in triples is ambiguous or missed, helping identify relevant

|  Corpus | Method | Acc | F1  |
| --- | --- | --- | --- |
|  - | - | 54.08 | 26.79  |

|  Text Corpora  |   |   |   |
| --- | --- | --- | --- |
|  Wikipedia | Random | 52.77 | 25.56  |
|   |  BM25 | 56.15 | 30.43  |
|   |  Dense Retrieval | 56.04 | 30.33  |
|  Pes2o-Abstract | Random | 53.34 | 26.00  |
|   |  BM25 | 54.60 | 27.95  |
|   |  Dense Retrieval | 55.43 | 29.19  |
|  Common Crawl | Random | 53.31 | 26.45  |
|   |  BM25 | 54.56 | 28.32  |
|   |  Dense Retrieval | 54.42 | 28.49  |

|  Knowledge Graph  |   |   |   |
| --- | --- | --- | --- |
|  Freebase | Think on Graph | 53.75 | 24.81  |
|  ATLAS-Wiki |  | 56.43 | 30.48  |
|  ATLAS-Pes2o | HippoRAG2 | 55.30 | 28.12  |
|  ATLAS-CC |  | 55.56 | 29.57  |

Table 6: Balanced accuracy (%) and F1 score (%) on FELM benchmark of Llama-3.1-8b-Instruct with retrieval methods. The best results are in bold, and the second best results are underlined.

subgraphs containing passage nodes; 2) Concept nodes create alternative pathways. These nodes establish connections beyond direct entities and events, addressing complex multi-hop question answering limitations. Figure 10 shows how concept nodes link knowledge across disparate subgraphs, enabling systems like HippoRAG to bridge separate subgraph influences via PageRank algorithms.

### 5.3 Enhancing LLM Factuality with KGs

We evaluated our KG's effectiveness in enhancing factuality using the FELM benchmark (Chen et al., 2023), which contains 847 samples across five domains with 4,425 fine-grained text segments. Following FELM's protocol, we applied RAG to three domains (world knowledge, science/technology, and writing/recommendation) while maintaining vanilla settings for math and reasoning domains. For a comprehensive comparison, we evaluated against multiple retrieval methods: HippoRAG v2, BM25, and dense retrieval using MiniLM (Wang et al., 2021). The retrieval process on text corpora is implemented in ElasticSearch database system (Elasticsearch, 2018). Our decision to use MiniLM rather than larger language model-based embedding approaches was driven by computational constraints. Implementing dense retrieval with higher-dimensional embeddings (e.g., 4096 dimensions) across our one billion nodes would require approximately 16 terabytes of storage using standard 32-bit floating-point representation. These