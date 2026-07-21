dramatically improving construction efficiency.

The power of AutoSchemaKG extends beyond its construction methodology to deliver substantial performance improvements in downstream applications. In rigorous evaluations, our approach outperforms state-of-the-art baselines by 12-18% on multi-hop question answering tasks (Trivedi et al., 2022; Yang et al., 2018; Ho et al., 2020) and enhances large language model factuality by up to 9% (Chen et al., 2023). Moreover, we found that our constructed knowledge graph is helpful for Llama3.1 7B models on general reasoning task on various domains that intensively requeries background knowledge, including Global Facts, History, Law, Religion, Philosophy and Ethics, Medicine and Health, and Social Sciences. These gains stem from our system's ability to create richer semantic representations through the integration of entities, events, and their conceptual abstractions—enabling better reasoning over complex information across different domains and data sources. Our key contributions include:

- We develop an entity-event-concept extraction framework that captures not only traditional entity relationships but also complex event structures and their conceptual categorizations, creating a multi-dimensional knowledge representation.
- We apply our efficient knowledge extraction and integration approach to web-scale data, processing billions of triples while maintaining semantic consistency. The resulting ATLAS-family knowledge graphs are, to the best of our knowledge, both the largest automatically constructed knowledge graphs and the largest graph-based Retrieval Augmented Generation (Graph RAG) datasets available.
- We build a retrieval augmented generation pipeline on the billion-scale ATLAS KGs, demonstrating AutoSchemaKG's effectiveness across diverse domains without domain-specific customization, establishing a truly general-purpose knowledge acquisition framework.

AutoSchemaKG represents a fundamental rethinking of knowledge graph construction, transforming what was once a heavily supervised process requiring significant domain expertise into a fully automated pipeline. This advancement not

only accelerates KG development but also dramatically expands the potential application domains for knowledge-intensive AI systems.

## 2 Problem Definition

We formally outline the tasks involved in automatically constructing knowledge graphs. We begin by providing a precise definition of a knowledge graph equipped with a conceptual schema.

Definition 1 (Knowledge Graph with Conceptual Schema). Consider a knowledge graph denoted as $G = (V, E, C, \phi, \psi)$, where: $V = V_E \cup V_N$ represents the collection of nodes, with $V_E$ as the set of event nodes, $V_N$ as the set of entity nodes, and $V_E \cap V_N = \emptyset$. $E \subseteq V \times V \times R$ defines the set of edges, where $R$ denotes relation types. Edges may connect entity-entity, entity-event, or event-event nodes. $C$ is the set of conceptual categories. $\phi : V \to \mathcal{P}(C)$ assigns each node a subset of concepts, where $\phi(v) \subseteq C$ for every $v \in V$. $\psi : R \to \mathcal{P}(C)$ links each relation type to a subset of concepts, where $\psi(r) \subseteq C$ for every $r \in R$. $\mathcal{P}(C)$ denotes the power set of $C$, encompassing all possible subsets. Additional constraints: $\forall v \in V : \phi(v) \neq \emptyset$ and $\forall r \in R : \psi(r) \neq \emptyset$.

## 3 AutoSchemaKG Framework

In this section, we elaborate on the process of fully automating knowledge graph construction.

### 3.1 Triple Extraction

Our approach employs a multi-phase pipeline using Large Language Models to convert unstructured text into knowledge triples from the Dolma corpus (Soldaini et al., 2024). This pipeline extracts Entity-Entity, Entity-Event, and Event-Event relationships through three sequential stages. We preprocess texts by filtering for English language and segmenting documents that exceed token limits. The segmented texts are grouped into processing batches. Stage 1 extracts Entity-Entity relationships using a system prompt $P_{EE}$ that instructs the LLM to detect entities and their interrelations. The output is parsed into triples $(e_1, r, e_2)$, where $e_1, e_2 \in V_N$ are entity nodes and $r \in R$ is a relation type. Stage 2 identifies Entity-Event relationships with prompt $P_{EV}$, producing triples $(e, r, v)$ or $(v, r, e)$, where $e \in V_N$, $v \in V_E$, and $r \in R$. Stage 3 targets Event-Event relationships with prompt $P_{VV}$, generating triples $(v_1, r, v_2)$, where $v_1, v_2 \in V_E$ and $r \in R$. The pipeline supports various LLMs with