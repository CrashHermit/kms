|   | Question Answering Corpora |   |   | Pre-training Corpora  |   |   |
| --- | --- | --- | --- | --- | --- | --- |
|   | MuSiQue | 2WikiQA | HotpotQA | ATLAS-Wiki | ATLAS-Pes2o | ATLAS-CC  |
|  # Text Chunks | 11,656 | 6,119 | 9,221 | 9.599M | 7.918M | 35.040M  |
|  # Entities | 108,582 | 48,782 | 95,686 | 70.104M | 75.857M | 241.061M  |
|  # Events | 99,747 | 50,910 | 82,833 | 165.717M | 92.636M | 696.195M  |
|  # Concepts | 37,414 | 19,830 | 32,410 | 8.091M | 5.895M | 31.070M  |
|  # Nodes | 245,743 | 119,522 | 210,929 | 243.912M | 174.387M | 937.256M  |
|  # Entity-Entity Edges | 91,186 | 40,748 | 78,467 | 0.114B | 0.076B | 0.414B  |
|  # Event-Entity Edges | 143,254 | 63,680 | 123,527 | 0.265B | 0.208B | 1.063B  |
|  # Event-Event Edges | 45,157 | 21,062 | 36,602 | 0.071B | 0.044B | 0.295B  |
|  # Conceptulization Edges | 933,330 | 432,869 | 789,608 | 1.041B | 0.821B | 4.178B  |
|  # Edges | 1,212,927 | 558,359 | 1,028,204 | 1.492B | 1.150B | 5.958B  |

Table 1: Statistics of knowledge graph construction across QA datasets (MuSiQue, 2WikiQA, HotpotQA) and LLM pre-training corpora (En-Wiki, Pes2o-Abstract, Common Crawl) for ATLAS knowledge graphs, showing counts of text chunks, nodes (entities/events), concepts, edges (Entity-Entity, Event-Entity, Event-Event), and conceptualizations. M = million, B = billion.

optimized precision settings and GPU acceleration. Extracted triples with their corresponding texts and metadata are serialized into JSON files.

### 3.2 Schema Induction

Following triple extraction, we perform schema induction to abstract specific entities, events, and relations into generalized types. This process uses LLMs to generate conceptual phrases representing types of each graph element, aligning with our formal definition  \( G = (V, E, C, \phi, \psi) \) . For each category (events, entities, and relations), we process elements in batches. The LLM generates at least three phrases per element that encapsulate its type or related concepts at varying abstraction levels. For entities  \( (e \in V_{N}) \) , we enhance abstraction by incorporating contextual information from neighboring nodes. We sample up to  \( N_{ctx} \)  neighbors to construct a context string that provides additional semantic cues. The schema induction pipeline processes the graph serialized from the triple extraction phase. Elements are partitioned into batches, with options for slicing for distributed computation. The generated phrases are recorded in a CSV file, mapping each node  \( v \in V \)  and relation  \( r \in R \)  to a subset of concepts in C via  \( \phi \)  and  \( \psi \) . This automated schema enhances the knowledge graph's adaptability across varied domains without requiring manual curation.

## 4 Construction of ATLAS Families

Corpora As shown in Table 1, the ATLAS-Wiki, ATLAS-Pes2o, and ATLAS-CC are constructed from the subsets from Dolma's subset of Wikipedia & Wikibooks, Semantic Scholar, and Dolma's CC

respectively. \( ^{2} \)  We use the full Wikipedia & Wiki-books to construct the ATLAS-Wiki, and we use the abstract part of Semantic Scholar to construct ATLAS-Pes2o, and we use the each of 3% from cc-head, cc-middle, and cc-tail to construct ATLAS-CC. According to (Soldaini et al., 2024) the head, middle, tail of CC are used to measure the distribution similarity to Wikipedia text.

Computational Cost We constructed our knowledge graphs using 80GB GPUs with 1,513 TFLOPS of FP16 compute, running Llama-3-8B-instruct with Flash Attention. The computational demands were substantial: 14,300 GPU hours for En-Wiki (243.9M nodes, 1.49B edges), 11,800 GPU hours for Pes2o-Abstract (174.4M nodes, 1.15B edges), and 52,300 GPU hours for Common Crawl (937.3M nodes, 5.96B edges). Processing 1024-token chunks in batches, we invested approximately 78,400 GPU hours total to extract billions of semantic relationships.

## 5 Experiment

In the this section, we show that the AutoSchemaKG has accurate triplex extraction, can coherently induce schemas, and has very high information preservations in section 5.1.

### 5.1 Evaluating AutoSchemaKG

Evaluating Triple Extraction Accuracy We use a rigorous counting-based evaluation method. Rather than relying on subjective scoring, we employ DeepSeek-V3 (Liu et al., 2024) as a judge in a structured verification process. For each document: (1) We present DeepSeek-V3 with both the