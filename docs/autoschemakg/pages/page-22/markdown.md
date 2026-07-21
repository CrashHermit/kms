## F The Recall Metrics in Opendomain QA Tasks

We also use Retrieval Quality metrics at  \( k \in \{2, 5\} \) : PR@k =  \( |D_k \cap S|/|S| \)  where PR@k (Partial Recall) measures the fraction supporting document is in top-k,  \( D_k \)  is the set of top-k retrieved documents, and S is the set of supporting documents. For multi-hop QA datasets (HotpotQA, 2WikiMultihopQA, MuSiQue), these retrieval metrics are crucial as they measure how effectively our system retrieves the evidence needed for multi-step reasoning.

Questions in datasets like 2WikiMultihopQA (Ho et al., 2020) and HotpotQA (Yang et al., 2018) tend to be more entity-centric, with relationships and entities more explicitly represented, which aids retrievers in easily locating relevant subgraphs. In contrast, MuSiQue (Trivedi et al., 2022), due to its questions' increased complexity in both description and multi-hop nature, poses greater challenges for retrieval. Additionally, differences in graph construction cause the retrievers to perform differently across datasets.

## G Details and Full Results on General Benchmarks

### G.1 Implementation and Evaluation Details on FELM

For the evaluation metrics, we follow the original paper (Chen et al., 2023) and use balanced accuracy and F1 score to evaluate the factuality checking capability. For the classification of segments in an instance, we ask the model to generate the ID of false segments, and then get the true positive (TP), false positive (FP), true negative (TN) and false negative (FN) results. The balanced accuracy is calculated as:

\[
\text { Balanced   Accuracy } = \frac {T P}{T P + F N} + \frac {T N}{T N + F P} \tag {6}
\]

Since we use F1 score to evaluate the factual error detection capability, we calculate the F1 score as:

\[
\mathrm{F} 1 = \frac {2 \cdot \text { Precision } \cdot \text { Recall }}{\text { Precision } + \text { Recall }} \tag {7}
\]

where Precision =  \( \frac{TN}{TN+FN} \)  and Recall =  \( \frac{TN}{TN+FP} \) .

We use the Retrieval-Augmented Generation method with different knowledge bases on 3 domains (world knowledge, science and technology, and writing/recommendation) of FELM benchmark. For the math domain and reasoning domain, we use the vanilla setting and their results are the

same across different knowledge bases. The detailed results of the 3 domains are shown in Table 10.

### G.2 Implementation and Evaluation Details on MMLU

Table 11 presents our classification for organizing MMLU tasks into distinct subject categories, providing a structured framework for domain-specific performance analysis. Table 12 displays comprehensive results across all MMLU subject areas, revealing an important insight: while retrieval-augmented generation enhances performance in knowledge-intensive domains, it can negatively impact performance on reasoning-focused tasks such as mathematics and logical reasoning. This finding aligns with previous research suggesting that RAG may sometimes interfere with LLMs' inherent reasoning capabilities.