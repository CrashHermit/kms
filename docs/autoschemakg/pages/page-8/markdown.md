|  Knowledge Source | History | Law | Religion | Phil/Eth | Med/Hlth | GlbFct | SocSci | Logic | Average  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  None | 76.59 | 66.86 | 83.04 | 63.55 | 70.38 | 66.72 | 79.74 | 64.35 | 72.10  |
|  Freebase-ToG | 78.42 | 69.00 | 75.44 | 65.67 | 72.65 | 67.27 | 76.00 | 66.03 | 72.34  |
|  Random Baseline  |   |   |   |   |   |   |   |   |   |
|  Wikipedia | 76.64 | 66.82 | 79.53 | 59.26 | 70.34 | 66.46 | 77.78 | 59.21 | 70.62  |
|  Common Crawl | 74.89 | 66.52 | 79.53 | 61.74 | 69.82 | 68.11 | 77.52 | 59.30 | 70.60  |
|  Pes2o-Abstract | 76.24 | 64.16 | 80.70 | 62.01 | 70.62 | 66.59 | 77.16 | 62.39 | 70.82  |
|  Text Corpora + DBM25  |   |   |   |   |   |   |   |   |   |
|  Wikipedia | 76.67 | 67.35 | 78.36 | 63.34 | 69.35 | 61.98 | 76.99 | 62.30 | 70.61  |
|  Common Crawl | 76.15 | 66.36 | 80.12 | 60.43 | 69.58 | 64.67 | 76.71 | 63.18 | 70.36  |
|  Pes2o-Abstract | 78.01 | 65.89 | 78.95 | 63.83 | 71.01 | 65.78 | 77.07 | 59.34 | 71.22  |
|  Text Corpora + Dense Retrieval  |   |   |   |   |   |   |   |   |   |
|  Wikipedia | 73.59 | 69.60 | 79.53 | 63.58 | 70.82 | 62.41 | 76.83 | 62.21 | 70.86  |
|  Common Crawl | 74.47 | 68.98 | 79.53 | 60.46 | 69.29 | 64.09 | 75.21 | 61.86 | 69.56  |
|  Pes2o-Abstract | 75.79 | 61.82 | 78.36 | 65.15 | 69.72 | 66.77 | 76.47 | 63.05 | 70.52  |
|  ATLAS + HippoRAG2  |   |   |   |   |   |   |   |   |   |
|  ATLAS-Wiki | 76.73 | 67.38 | 84.21 | 66.01 | 70.82 | 68.36 | 79.16 | 63.65 | 72.53  |
|  ATLAS-CC | 78.16 | 70.85 | 83.04 | 65.60 | 71.28 | 63.95 | 78.16 | 65.42 | 72.66  |
|  ATLAS-Pes2o | 77.13 | 68.41 | 81.29 | 65.05 | 72.75 | 65.67 | 81.19 | 62.98 | 73.25  |
|  ATLAS + ToG  |   |   |   |   |   |   |   |   |   |
|  ATLAS-Wiki | 77.91 | 66.60 | 84.21 | 65.10 | 70.69 | 63.85 | 78.31 | 67.08 | 72.18  |
|  ATLAS-CC | 77.07 | 68.18 | 83.63 | 65.24 | 72.03 | 66.87 | 79.72 | 66.59 | 73.07  |
|  ATLAS-Pes2o | 77.52 | 66.95 | 84.80 | 63.44 | 71.15 | 68.92 | 81.59 | 67.87 | 73.28  |

Table 7: Performance comparison of Llama-3.1-8b-Instruct with our KG-integrated HippoRAG2 and ToG versus baseline methods across Wikipedia, Common Crawl, and Pes2o-Abstract corpora on MMLU benchmarks. Tasks are grouped by subject, with bold and underlined values indicating first and second-highest scores. Phil/Eth, Med/Hlth, GlbFct, and SocSci denote Philosophy/Ethics, Medicine/Health, Global Facts, and Social Sciences.

baselines represent state-of-the-art approaches in graph-based RAG and standalone retrieval systems. All experiments were implemented using the same LLaMA-3.1-8B-Instruct model with Neo4j integration and zero-shot CoT settings, ensuring fair comparison across methods. Performance was measured using balanced accuracy (giving equal weight to true and false segments) and F1 score for detecting factual errors (Table 6). Our results demonstrate that HippoRAG2 with our KG consistently outperforms baselines on Wikipedia (56.43% accuracy, 30.48% F1) and Common Crawl corpora, while achieving competitive results on Pes2o-Abstract. The superior performance on Wikipedia likely stems from FELM samples being partially sourced from Wikipedia content. Detailed implementation specifics and extended results are available in Appendix G.1.

### 5.4 General Domain Knowledge Capabilities

To assess AutoSchemaKG's ability to construct knowledge graphs across various domains, we evaluated it on MMLU (Hendrycks et al., 2021), a comprehensive benchmark for LLM reasoning. Previ-

ous research on KNN-LMs (Khandelwal et al.) suggests that retrieval-augmented generation can sometimes hinder LLMs' reasoning capabilities (Wang et al., 2023a; Geng et al., 2025). While we do not expect RAG to universally improve LLM performance, our findings demonstrate significant improvements in knowledge-intensive domains, even those covered in LLM training data. Using the same retrieval and generation settings as our FELM experiments, we classified MMLU tasks into subject categories (detailed mapping in Appendix G.2) and focused on knowledge-intensive domains including History, Law, Religion, Philosophy/Ethics, Medicine/Health, Global Facts, Social Sciences, and Logic.

As shown in Table 7, our ATLAS knowledge graphs enhanced performance across these domains on all tested corpora. Notably, each ATLAS variant demonstrated distinct strengths: ATLAS-Pes2o excelled in Religion, Medicine/Health, Global Facts, and Social Sciences, reflecting its academic paper-sourced knowledge; ATLAS-Wiki showed advantages in general knowledge areas like Religion, Philosophy/Ethics, and Global Facts; while ATLAS-