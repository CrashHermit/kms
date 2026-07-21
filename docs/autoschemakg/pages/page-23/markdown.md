|  Model/Dataset | MuSiQue |   | 2Wiki |   | HotpotQA  |   |
| --- | --- | --- | --- | --- | --- | --- |
|  Metric | Recall@2 | Recall@5 | Recall@2 | Recall@5 | Recall@2 | Recall@5  |
|  Baseline Retrievers  |   |   |   |   |   |   |
|  Contriever | 34.8 | 46.6 | 46.6 | 57.5 | 58.4 | 75.3  |
|  BM25 | 32.4 | 43.5 | 55.3 | 65.3 | 57.3 | 74.8  |
|  LLM Embeddings  |   |   |   |   |   |   |
|  GTE-Qwen2-7B-Instruct | 48.1 | 63.6 | 66.7 | 74.8 | 75.8 | 89.1  |
|  GritLM-7B | 49.7 | 65.9 | 67.3 | 76.0 | 79.2 | 92.4  |
|  NV-Embed-v2 (7B) | 52.7 | 69.7 | 67.1 | 76.5 | 84.1 | 94.5  |
|  Existing Graph-based RAG Methods  |   |   |   |   |   |   |
|  RAPTOR (Llama-3.3-70B-Instruct) | 47.0 | 57.8 | 58.3 | 66.2 | 76.8 | 86.9  |
|  HippoRAG (Llama-3.3-70B-Instruct) | 41.2 | 53.2 | 71.9 | 90.4 | 60.4 | 77.3  |
|  HippoRAG2 (Llama-3.3-70B-Instruct) | 56.1 | 74.7 | 76.2 | 90.4 | 83.5 | 96.3  |
|  AutoSchemaKG (LLama-3.1-8B-Instruct) + HippoRAG1  |   |   |   |   |   |   |
|  Entity-KG (Llama-3-8B-Instruct) | 41.37 | 51.08 | 61.72 | 75.45 | 51.89 | 65.95  |
|  Entity-Event-KG (Llama-3-8B-Instruct) | 41.28 | 51.12 | 61.37 | 74.56 | 51.31 | 65.93  |
|  Full-KG (Llama-3-8B-Instruct) | 40.78 | 50.36 | 61.08 | 71.9 | 52.8 | 65.4  |
|  AutoSchemaKG (LLama-3.1-8B-Instruct) + HippoRAG2  |   |   |   |   |   |   |
|  Entity-KG (Llama-3-8B-Instruct) | 48.33 | 72.58 | 67.34 | 84.25 | 77.59 | 92.16  |
|  Entity-Event-KG (Llama-3-8B-Instruct) | 48.83 | 72.7 | 68.59 | 85.85 | 81.26 | 92.66  |
|  Full-KG (Llama-3-8B-Instruct) | 49.12 | 72.48 | 68.46 | 84.6 | 84.17 | 93.04  |

Table 8: Recall @ 2 and Recall @ 5.

Table 9: Recall performance in the knowledge graph created by Llama-3-8B-Instruct shows strong performance that is comparable with the knowledge graph created with 70B model.

|  Corpus | Method | World Knowledge |   |   |   | Science and Technology |   |   |   | Writing/Recommendation  |   |   |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |   |  P | R | F1 | Acc | P | R | F1 | Acc | P | R | F1 | Acc  |
|  - | - | 36.67 | 29.93 | 32.96 | 55.10 | 15.43 | 24.51 | 18.94 | 50.49 | 25.95 | 12.73 | 17.09 | 52.69  |
|  Text Corpora  |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  Wikipedia | Random | 31.78 | 27.89 | 29.71 | 52.52 | 7.95 | 11.76 | 9.49 | 43.94 | 21.95 | 26.97 | 24.20 | 53.78  |
|   |  BM25 | 26.82 | 32.65 | 29.45 | 49.31 | 15.23 | 38.24 | 21.79 | 50.48 | 29.21 | 48.69 | 36.52 | 62.40  |
|   |  Dense Retrieval | 33.93 | 38.78 | 36.19 | 54.97 | 16.92 | 43.14 | 24.31 | 53.01 | 25.22 | 43.45 | 31.91 | 58.68  |
|  Pes2o-Abstract | Random | 27.36 | 19.73 | 22.92 | 49.86 | 10.13 | 15.69 | 12.31 | 45.64 | 26.92 | 28.84 | 27.85 | 56.50  |
|   |  BM25 | 32.43 | 32.65 | 32.54 | 53.34 | 11.18 | 17.65 | 13.69 | 46.54 | 26.75 | 32.96 | 29.53 | 57.34  |
|   |  Dense Retrieval | 31.43 | 29.93 | 30.66 | 52.50 | 20.51 | 47.06 | 28.57 | 57.55 | 25.37 | 32.21 | 28.38 | 56.51  |
|  Common Crawl | Random | 30.14 | 29.93 | 30.03 | 51.72 | 11.17 | 21.57 | 14.72 | 45.75 | 23.73 | 28.09 | 25.73 | 54.91  |
|   |  BM25 | 27.21 | 27.21 | 27.21 | 49.71 | 12.21 | 25.49 | 16.51 | 46.68 | 27.32 | 40.82 | 32.73 | 59.42  |
|   |  Dense Retrieval | 25.50 | 34.69 | 29.39 | 48.00 | 15.48 | 36.27 | 21.70 | 50.78 | 24.41 | 38.95 | 30.01 | 57.27  |
|  Knowledge Graph  |   |   |   |   |   |   |   |   |   |   |   |   |   |
|  Freebase | Think on Graph | 26.00 | 8.84 | 13.20 | 49.62 | 23.76 | 23.53 | 23.65 | 55.15 | 42.86 | 12.36 | 19.19 | 54.51  |
|  ATLAS-Wiki | HippoRAG2 | 33.33 | 42.18 | 37.24 | 54.98 | 16.45 | 50.00 | 24.76 | 52.75 | 32.82 | 32.21 | 32.51 | 59.43  |
|  ATLAS-Pes2o |   | 39.17 | 31.97 | 35.21 | 56.51 | 21.35 | 40.20 | 27.89 | 57.13 | 30.37 | 15.36 | 20.40 | 54.11  |
|  ATLAT-CC |   | 33.80 | 48.98 | 40.00 | 56.18 | 18.06 | 52.94 | 26.93 | 55.42 | 24.20 | 25.47 | 24.82 | 54.66  |

Table 10: Factuality results (%) on different domains of FELM benchmark with different Text Corporas and retrieval methods. P, R, F1, and Acc denote Precision, Recall, F1 score, and Balanced Accuracy, respectively.