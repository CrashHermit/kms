## C Experiment Settings of Schema Accuracy

### C.1 Datasets

Entity Typing. We conduct experiments on the typed entities of two real-world knowledge graphs, FB15kET (Bordes et al., 2013) and YAGO43kET (Moon et al., 2017a), which are the subsets of Freebase (Bollacker et al., 2008) and YAGO (Suchanek et al., 2007), respectively. The types of entities are collected from (Moon et al., 2017b). There are 3,584 and 45,182 entity types in FB15kET and YAGO43kET, respectively. We utilize the entities in the testing sets of these two datasets with their types as ground truths to validate the entity induction performance of our schema induction method.

Event Typing. We conduct experiments on the typed events of wikiHow (Koupaee and Wang, 2018), which is an online community contains a collection of professionally edited how-to guideline articles. The types of events are collected by P2GT (Chen et al., 2020). There are 625 event types among 12,795 events. We utilize the events in the testing set of wikiHow with their types as ground truths to validate the event induction performance of our schema induction method.

Relation Typing. There are no datasets designed for the relation typing task, so here we make use of the domain segments separated by "/" in FB15kET (Bordes et al., 2013) to extract the types. These domain segments serve as ground truth types, with the last domain component functioning as the relation itself. There are 607 relation types among 1,345 relations in FB15kET. We utilize the relations in the testing set of FB15kET with their types as ground truths to validate the relation induction performance of our schema induction method.

### C.2 Metrics

We employ BertScore-Recall and BertScore-Coverage as the evaluation metrics, which are denoted as BS-R and BS-C respectively. They are used to calculate how many types in each instance or entire testing set are recalled by our schema induction method. The BertScore (Zhang et al., 2019), which is denoted as BS, between each pair of type and induced schema are calculated as follows:

\[
\text { BertRecall } = \frac {1}{| t |} \sum_ {\hat {t} _ {i} \in \hat {t}} \max _ {\hat {t} _ {j} \in \hat {t}} \mathbf {x} _ {\hat {t} _ {i}} ^ {\top} \mathbf {x} _ {\hat {t} _ {j}}, \tag {1}
\]

\[
\text { BertPrec } = \frac {1}{| \hat {t} |} \sum_ {\hat {t} _ {i} \in \hat {t}} \max _ {\hat {t} _ {j} \in t} \mathbf {x} _ {\hat {t} _ {j}} ^ {\top} \mathbf {x} _ {\hat {t} _ {i}}, \tag {2}
\]

\[
\mathrm{BS} (t, \hat {t}) = 2 \frac {\text { BertRecall } \cdot \text { BertPrec }}{\text { BertRecall } + \text { BertPrec }}, \tag {3}
\]

where t and  \( \hat{t} \)  represents the tokens of a ground truth type and induced schema, respectively. The embedding vector of each token  \( t_{i} \)  or  \( \hat{t}_{j} \)  of a type t or induced schema  \( \hat{t} \)  is denoted as  \( x_{t_{i}} \)  and  \( x_{\hat{t}_{i}} \), which are obtained with RoBERTa (Liu et al., 2019). Then the BS-R and BS-C can be calculated as follows:

\[
\mathrm{BS-R} (\mathcal {T}, \hat {\mathcal {T}}) = \frac {1}{| \hat {\mathcal {T}} |} \sum_ {\hat {t} \in \hat {\mathcal {T}}} \max _ {t \in \mathcal {T}} \mathrm{BS} (t, \hat {t}), \tag {4}
\]

\[
\mathrm{BS-C} (\mathcal {S} _ {t}, \mathcal {S} _ {\hat {t}}) = \frac {1}{| \mathcal {S} _ {\hat {t}} |} \sum_ {\hat {t} \in \mathcal {S} _ {\hat {t}}} \max _ {t \in \mathcal {S} _ {t}} \mathrm{BS} (t, \hat {t}), \tag {5}
\]

where \(\mathcal{T}\) and \(\hat{\mathcal{T}}\) represent a set of ground truth types and induced schemas in each testing instance, respectively. Similarly, \(S_{t}\) and \(S_{\hat{t}}\) denote the set of ground truth types and induced schemas across the entire testing set, respectively.

## D Case Study Examples

Figures 9 and 10 demonstrate specific cases where events and concepts are crucial for effective knowledge graph utilization in retrieval-augmented generation. Figure 9 illustrates how event nodes provide essential contextual information that entity-only representations miss, while Figure 10 showcases how concept nodes establish semantic bridges across otherwise disconnected subgraphs, enabling more comprehensive reasoning for complex questions.

## E Algorithm for RAG

We include all the algorithms used in our RAG evaluation on various graphs constructed by AutoSchemaKG. Algorithm 1 presents the Think-on-Graph reasoning method that leverages our knowledge graphs for multi-hop question answering. For adapting our entity-event-concept graphs at different scales, we implemented two variants of HippoRAG2: Algorithm 2 for smaller, more focused graph traversal, and Algorithm 3 for large-scale graph exploration with optimized memory management. These adaptations enable efficient navigation of the rich semantic structures in our ATLAS knowledge graphs.