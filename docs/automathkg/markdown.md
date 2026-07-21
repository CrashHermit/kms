arXiv:2505.13406v1 [cs.AI] 19 May 2025

# AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database

Rong Bian \( ^{1,2} \) , Yu Geng \( ^{1,2} \) , Zijian Yang \( ^{1,2} \) , Bing Cheng \( ^{1,3,4*} \)

\( ^{1*} \) Academy of Mathematics and Systems Science, Chinese Academy of Sciences, Beijing, 100190, China.

\( ^{2} \) School of Mathematical Sciences, University of Chinese Academy of Sciences, Beijing, 100049, China.

\( ^{3*} \) AMSS Center for Forecasting Science, Chinese Academy of Sciences, Beijing, 100190, China.

\( ^{4*} \) State Key Laboratory of Mathematical Science, Academy of Mathematics and Systems Science, Chinese Academy of Sciences, Beijing, 100190, China.

*Corresponding author(s). E-mail(s): bc2@amss.ac.cn;

Contributing authors: bianrong2021@amss.ac.cn;

gengyu2020@amss.ac.cn; yangzijian20@mails.ucas.ac.cn;

## Abstract

A mathematical knowledge graph (KG) presents knowledge within the field of mathematics in a structured manner. Constructing a math KG using natural language is an essential but challenging task. There are two major limitations of existing works: first, they are constrained by corpus completeness, often discarding or manually supplementing incomplete knowledge; second, they typically fail to fully automate the integration of diverse knowledge sources. This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG capable of automatic updates. AutoMathKG regards mathematics as a vast directed graph composed of Definition, Theorem, and Problem entities, with their reference relationships as edges. It integrates knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing entities and relationships with large language models (LLMs) via in-context learning for data augmentation. To search for similar entities, MathVD, a vector database, is built through two designed embedding strategies using SBERT. To automatically update, two

mechanisms are proposed. For knowledge completion mechanism, Math LLM is developed to interact with AutoMathKG, providing missing proofs or solutions. For knowledge fusion mechanism, MathVD is used to retrieve similar entities, and LLM is used to determine whether to merge with a candidate or add as a new entity. A wide range of experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM.

Keywords: knowledge graph, vector database, data augmentation, LLMs, in-context learning, mathematical reasoning capability

# 1 Introduction

A knowledge graph (KG) aims to construct a knowledge base by extracting a series of triples from unstructured or semi-structured texts [1]. With the advancement of artificial intelligence (AI) and big data technologies, large-scale KGs have found widespread application in natural language processing (NLP) [2]. The mathematical KG, as an important branch, organizes and structures knowledge within the field of mathematics, making it more accessible and usable [3]. Math KGs can provide a foundational data resource for training more advanced models in mathematical reasoning, promoting the development of AI for mathematics [4]. Since a substantial portion of mathematical knowledge exists in natural language text [5], it is of great significance to construct a high-quality and diverse math KG from extensive texts.

Mathematical knowledge can be expressed in two linguistic forms: formal and informal [6]. Formal mathematics is written in verifiable formal languages similar to source code, such as Metamath [7] and Lean 4 [8]. Informal language, on the other hand, is a blend of symbolic and natural language used by humans. Despite the significant advancements in formal mathematics driven by NLP technologies [9], it does not directly address the informal aspects of mathematics [10], failing to promote human understanding of mathematics as effectively as natural mathematical language. Therefore, this paper aims to investigate the construction of a mathematical KG written in natural language.

Existing mathematical KGs in natural language can be mainly categorized as problem-solving [11] and concept-integrating [12–14]. While popular, there are two major limitations. First, they are constrained by the completeness of the current corpus, often discarding or manually supplementing missing knowledge, without fully exploring the inherent logic of mathematical knowledge. Second, they are typically static in structure and fail to automatically integrate knowledge from different sources, requiring substantial time and human resources for updates. Addressing these drawbacks, we extend the research on math KGs written in natural language and propose a novel method to construct automatically updatable KG based on large language models (LLMs).

This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional KG for the field of mathematics, written in natural language and capable of automatic update. AutoMathKG covers multiple mathematical domains and levels through integrating diverse math sources, including web data from ProofWiki, book data from math textbooks, paper data from arXiv, and problem data from TheoremQA annotated by GPT-4 in MathInstruct [15]. Innovatively, AutoMathKG regards mathematics as a vast multi-dimensional network composed of definitions, theorems, and problems. LLM is used to enhance math knowledge and extract potential reference relationships through in-context learning (ICL) [16], organizing and storing all information in a structured manner. Furthermore, AutoMathKG updates automatically through two designed mechanisms based on vector database (VD) and LLM, facilitating the expansion for math knowledge.

AutoMathKG is represented as a directed graph, where vertices denote math entities, categorized into Definition (Def), Theorem (Thm), and Problem (Prob), and directed edges denote the reference relationships between entities. Regarding entity vertices, three levels of information are stored in JSON format  \( [17] \)  for different knowledge requirements. First, basic information is extracted in preprocessing, identifying entities and their relationships via rule-based matching. Second, advanced information with step-by-step proofs and derivations is augmented by the Llama-2  \( [18] \)  model, inspired by the formal language Lean 4 while avoiding overly complex strategies and operations. Third, query information is collected for all entities connected to the current entity. Regarding directed edges, Llama-2 is used to discover associations among all entities and identify tactic labels when entities are referenced through ICL, similar to the use of tactics in Lean 4.

Considering that similar entity searches through graph traversal are expensive, we design a VD for AutoMathKG, named MathVD. By vectorizing, entity similarity is transformed into vector similarity, making fuzzy search available for mathematical knowledge. We propose two strategies for embedding entity vectors with SBERT  \( [19] \) , which differ in how descriptive statements about entities are constructed, resulting in MathVD1 and MathVD2. AutoMathKG updates automatically through two proposed mechanisms. The first mechanism achieves automatic knowledge completion by building Math LLM, which is capable of solving various math problems to supplement incomplete proofs and solutions for new Theorem and Problem entities. The second mechanism enables automatic knowledge fusion from different sources, utilizing MathVD to search for similar candidate entities and Llama-2 to determine whether to merge with the candidate or add as a new entity. A wide range of experiments and case studies demonstrate superior reachability query results in MathVD compared to five baselines, as well as strong mathematical reasoning capabilities in Math LLM, highlighting the advanced performance of the AutoMathKG system. To sum up, our main contributions are as follows:

- We construct AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG, which can be automatically updated based on VD and LLMs.
- We propose two novel math entity embedding strategies for MathVD construction, enabling fuzzy search for math entities.

- We propose an automatic KG update method involving two mechanisms. One is for knowledge completion by interacting with a designed Math LLM, and another is for knowledge fusion by MathVD and Llama-2.
- Experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM.

The remainder of this paper is organized as follows. Section 2 introduces related work. Section 3 provides an overview of the AutoMathKG system. Section 4 describes the KG construction. Section 5 details the VD construction. Section 6 explains the automatic updates method. Section 7 presents the experiments and evaluations. Section 8 conducts a further discussion. Section 9 concludes our work.

## 2 Related work

### 2.1 Mathematical data resource

Regarding math KGs, there are two categories: one for solving problems and the other for integrating concepts. MathGraph [11] aims at solving high school math problems automatically. Math-KG [13] integrates math resources including Baidu Baike and Wikipedia through a pipeline approach. NaturalProofs [12] comprises math statements and their proofs to construct a reference graph with their reference links based on rule matching. MathGloss [14] is a linked database of undergraduate-level math concepts leveraging online resources. Regarding math corpora, there are various sources. AMPS [20] is a synthetic exercise set while lacks challenging math topics. ProofPile [21] focuses on math theorem proving but limited to formal proofs. OpenWebMath [22] is a large-scale corpus composed of web pages. MathPile [23] integrates resources from textbooks, web pages, and others. Regarding math benchmarks, there are various datasets for different levels of knowledge. Some datasets focus on elementary math problems, such as NumGLUE [24] and GSM8K-RFT [25], while some are more challenging at the middle and high school level, such as Camel-Math [26] and MATH [20]. Recently, TheoremQA [27] addresses university-level problems. MathInstruct [15] encompasses various problems with reasoning annotations based on the Chain of Thought (CoT) [28] and Procedure of Thought (PoT) [29] approaches. GHOSTS [30] aims to assess advanced mathematical abilities of LLMs.

### 2.2 Knowledge graph and vector database

Modern knowledge graphs (KGs) assert facts in the form of (subject, predicate, object) triplets, typically stored in a graph database and visualized in graph structures, where subjects and objects are modeled as nodes, and the relationships between them, namely the predicates, are modeled as edges [31]. Some popular KGs include DBpedia [32], YAGO [33], YAGO2 [34], and Google Knowledge Graph [35]. KGs enable graph retrieval augmented generation (RAG) and conversational document retrieval [36]. An important task is to search for similar entities given an input entity. Current methods

for semantic similarity retrieval in KG, such as ontology matching and pattern matching, face challenges with the expansion of KG scale [37]. On the other hand, as word embedding techniques emerge, there are many methods to map words into continuous vector space, such as Word2vec [38], GloVe [39], BERT [40], and SBERT [19]. Since closely connected embeddings are semantically related in vector space, the similarity between entity vectors in the vector database can be utilized to retrieve similar entities, and various popular KG embedding models have been proposed, such as TransE [41], KG2E [42], HoLE [43], R-GCN [44], and BoxE [45].

## 2.3 In-context learning with LLMs

Large Language Models (LLMs) typically refer to transformer language models containing billions (or more) of parameters, trained on massive text data, such as GPT-4 [46], PaLM [47], Galactica [48], Llama-2 [18], and Gemma [49]. With the increase in model scale and corpus size, LLMs demonstrate powerful capabilities in understanding natural language and exhibit the ability of in-context learning (ICL), where LLMs learn to predict based solely on the in context added a few examples [16]. ICL has emerged as a new paradigm in NLP. First, some examples are required to form demonstration contexts, often written in natural language templates. Then, a query question is concatenated with a demonstration context to form a prompt. Finally, LLMs are prompted for prediction. LLMs can perform a variety of complex tasks through ICL, exhibiting strong few-shot performance across many NLP tasks, such as question answering, information extraction, and mathematical reasoning. Olsson et al. [50] analyzed the mechanisms underlying this behavior. Saunshi et al. [51] argued that, by conditioning on prompts, the task of predicting the next word was close to linearly separable. Liu et al. [52] investigated the impact of the number of provided examples. Min et al. [53] demonstrated the effectiveness of structuring prompts in an input-label pairing format. Lampinen et al. [54] suggested that incorporating explanatory task instructions within the context can further enhance performance.

## 3 AutoMathKG system overview

In order to fully explore mathematical knowledge from text and store it in a structured manner, we proposed AutoMathKG, an automatically updatable KG in the field of mathematics written in natural math language. We collect corpora from four sources: ProofWiki, textbooks, arXiv papers, and the TheoremQA dataset, covering a wide range of math domains and levels, and use LLMs to enhance knowledge. AutoMathKG is designed as a directed graph, where vertices represent math entities and directed edges represent reference relationships between entities. Math entities are classified into three types: Definition (Def), Theorem (Thm), and Problem (Prob). Figure 1 illustrates examples of some entities and their relationships in AutoMathKG. To achieve our design, there are three main problems: problem 1 is how to extract math entities and relationships from natural language texts, problem 2 is how to effectively augment and store key information about mathematical entities, and problem 3 is how to automatically update entities and their relationships.

![img-0.jpeg](img-0.jpeg)

Fig. 1 Examples of entities and reference edges with tactics in AutoMathKG.

As for problem 1, regarding Definition and Theorem entities, we preprocess each collected corpus using the same rule-based matching approach as NaturalProofs dataset, extracting basic information about mathematical definitions, theorems, and theorem proofs, along with extracting reference relationships between them. Regarding Problem entities, since NaturalProofs does not involve math problems, we utilize the problems and their solutions from TheoremQA annotated by GPT-4 in MathInstruct. To associate Problem entities with Definition and Theorem entities, LLM is used to discover potential references to definitions and theorems in Problem entities through ICL. Thus, reference relationships among all kinds of entities are comprehensively extracted. Furthermore, when entities are referenced, we innovatively employ LLM to identify the tactic labels and store them in the directed edges, such as “premise”, “assumption”, “lemma” and so on. These tactic labels for math entities in natural language form clarify how references drive theorem proofs or problem solutions, similar to the use of tactics in the formal language Lean 4 to assist in constructing theorem proofs.

As for problem 2, we store three levels of information for each entity in order to meet the knowledge requirements in different scenarios. Firstly, basic information is obtained directly from the corpus during entity extraction preprocessing, including entity type, label, title, contents, and source. Secondly, advanced information provides step-by-step proofs and derivations, using LLM to divide theorem contents and proofs into logical segments, inspired by Lean 4 while eliminating overly complex strategies and operations. Lastly, query information contains all incoming and outgoing entities related to the current entity. The three levels of information capture the intrinsic logic of mathematical entities in both content statements and reference citations. All entity information is stored in JSON format, where each key-value pair represents an information attribute of the entity. Figure 2 illustrates the flowchart for extracting and storing information from the input corpus to construct the Input KG.

As for problem 3, we design two mechanisms to achieve automatic updates of our math KG. The first mechanism is to supplement the incomplete knowledge of new entities. We build a specialized mathematical LLM called Math LLM, capable of solving various mathematical problems, to complete missing proofs or solutions for

![img-1.jpeg](img-1.jpeg)

Fig. 2 Model architecture of AutoMathKG construction for building Input KG from input text.

![img-2.jpeg](img-2.jpeg)

Fig. 3 Model architecture of AutoMathKG automatic updates for knowledge completion by Math LLM and knowledge fusion by VD and LLM between Input KG and Existing KG.

new entities. The second mechanism is to add entities and relationships from new texts in different corpus sources to the Existing KG without duplication or omission. We utilize VD and LLM to update the new entities. First, an Input KG is constructed by extracting entities and relationships from the new input text. Then, Input VD and Existing VD are built from their respective KGs, allowing similar entity candidates for each input entity vector to be retrieved. Finally, through ICL, LLM is employed to determine whether to merge the input entity with a similar candidate or add it as a new entity. Figure 3 depicts the flowchart for the automatic updates of our math KG.

## 4 Method of AutoMathKG construction

### 4.1 Corpus collection

To construct AutoMathKG, we collected diverse data from four sources, including web data from ProofWiki, problem data from TheoremQA, book data from high-quality mathematics textbooks, and paper data from arXiv, covering a wide range

of math subjects. Table 1 presents the corpus statistics for AutoMathKG entities, including Definition (Def), Theorem (Thm), and Problem (Prob) entities, as well as other entities with uncertain types.

Table 1 Statistics of AutoMathKG corpus sources.

|  Courpus | Sample | Entity  |   |   |   |
| --- | --- | --- | --- | --- | --- |
|   |   |  Def | Thm | Prob | Other  |
|  ProofWiki | 9496 | 4743 | 1811 | 0 | 2942  |
|  Textbook | 1605 | 361 | 1209 | 0 | 35  |
|  ArXiv | 538 | 134 | 399 | 0 | 5  |
|  TheoremQA | 1749 | 0 | 0 | 1084 | 665  |
|  Total | 13388 | 5238 | 3419 | 1084 | 3647  |

### 4.1.1 ProofWiki

ProofWiki$^{1}$ is an online compendium of mathematical proofs, containing mathematical definitions, theorems, and their proofs across various domains, contributing significantly to the advancement of formal proof fields. By manually designing rules to filter pages, information such as type, title, contents, and references for each page can be obtained, similar to the approach used in NaturalProofs [12] and ProofPile [21]. This paper utilized a portion of ProofWiki data processed by NaturalProofs as the initial source of Definition and Theorem entity data for AutoMathKG.

### 4.1.2 TheoremQA

TheoremQA [27] is a theorem-driven QA dataset built upon university-level theorems sourced from various subfields such as algebra, number theory, graph theory, and information theory. Chen et al. [27] searched for problems related to these theorems from different sources and standardized the problems to ensure uniform solution formats. Yue et al. [15] supplemented TheoremQA with reasoning annotations using GPT-4 in MathInstruct, providing solutions through both CoT and PoT methods. This paper utilized TheoremQA data from the MathInstruct dataset with annotated rationales as the initial source of Problem entity data for AutoMathKG.

### 4.1.3 Textbook

Mathematical textbooks contain rich and rigorous math definitions, theorems, proofs, and explicit reference relationships. We conducted extensive manual searches on the internet, specifically targeting open-source mathematical textbooks available on freely accessible websites. For each textbook, we downloaded the $\LaTeX$ source code and extracted statements and proofs based on environment names. In total, we processed eight textbooks covering a variety of subjects, including real analysis, algebra, differential equations, elementary number theory, calculus, geometry, and others (see Table A1 in Appendix A for more details). Among these, two textbooks were selected from

NaturalProofs, while the remaining six were processed through our work. The collected textbooks were used as a data source for updating AutoMathKG.

### 4.1.4 ArXiv

ArXiv$^2$ is a free distribution service and an open-access archive for millions of scientific papers, providing valuable training data for many powerful language models. We collected 20 mathematical papers from arXiv, downloading the LaTeX source code for each paper and extracting statements and proofs based on environment names. These papers cover various subfields such as algebraic geometry, algebraic topology, differential geometry, statistics, and probability theory (see Table A2 in Appendix A for more details). The collected mathematical papers from arXiv were used as a data source for updating AutoMathKG.

### 4.2 AutoMathKG structure

AutoMathKG is represented as a directed graph $G = \{V, E\}$, where $V$ is the set of vertices corresponding to mathematical entities, and $E$ is the set of edges representing the relationships between mathematical entities. Below, we elaborate on the design and storage for vertices and edges.

#### 4.2.1 Entity vertices

There are three types of entities in AutoMathKG: Definition entities, Theorem entities, and Problem entities, corresponding to three types of vertices in the directed graph $G$.

Definition entities represent concise and comprehensive statements of mathematical concepts. Mathematical definitions serve as the foundation for mathematical theorems and their proofs, as well as the prerequisites for the formulation and solution of mathematical problems. Some complex mathematical definitions may refer to other fundamental definitions. For instance, defining a right triangle involves referencing the definition of a triangle.

Theorem entities represent mathematical statements that have been logically proven to be true, such as the *Pythagoras's Theorem*. Since there is a variety of terms indicating the roles of mathematical statements in specific topics, we consider Theorem entities to include theorems, propositions, corollaries, and lemmas. Proving theorems is a central activity in mathematics, and a theorem may have multiple proofs from different perspectives. Hence, in AutoMathKG, Theorem entities include not only the statement of the theorem but also the corresponding proof process.

Problem entities represent mathematical problems that can be formulated, analyzed, and potentially solved using mathematical methods. These problems may range from real world problems, such as calculating the orbits of planets in the solar system, to abstract problems, such as determining the order of a prime group. In the statement and solution of problems, references to other mathematical definitions or theorems are often made, which directly demonstrates the application of mathematical definitions and theorems. Hence, in AutoMathKG, Problem entities include not only the statement of the problem but also the corresponding solution process.

#### 4.2.2 Entity storage

To effectively store key information about mathematical entities, every entity is stored in JSON format, where each key-value pair represents an attribute of the entity. The following entity attributes are selected to store: "id", "type", "label", "title", "field", "contents", "bodylist", "refs", "references_tactics", "source", "proofs", "solutions", "in_refs", "in_ref_ids", "out_refs" and "out_ref_ids". The entity storage structure is depicted in Fig. 4, and detailed information on all attributes can be found in Table B3 of Appendix B.

Specifically, “contents” represents the entity content, presented as a list of sequence strings, with each sequence being a mixture of text and LATEX language. “Bodylist” denotes content segmentation, structured as a nested list of dictionaries, where each dictionary provides a description of the segmented content and its corresponding action label, stored in the keys of “description” and “action”, respectively. “References_tactics” signifies referenced entities and tactic labels, presented as a dictionary, with each key representing a referenced entity and its value indicating the associated tactic label. We assign the same labels to the action and tactic attributes, which are described in Section 4.2.3. Table 2 illustrates examples of content segmentation and reference tactics for Definition, Theorem, and Problem entities. “In_refs” and “out_refs” respectively store information about all incoming and outgoing edge entities of the current entity. Additionally, Theorem entities include “proofs”, and Problem entities include “solutions”. Each theorem proof and problem solution contains “contents”, “bodylist” (only in theorem proofs), “refs” and “references_tactics”, following the same format as above. An instance of a Definition entity is shown in Fig. E1 of Appendix E.

![img-3.jpeg](img-3.jpeg)

Fig. 4 The schema of entity storage in JSON format.

#### 4.2.3 Entity relation edges

The edges in AutoMathKG are directed edges, representing reference links. An edge from entity vertex  \( v_{i} \in V \)  to entity vertex  \( v_{j} \in V \) , denoted as  \( e_{i \to j} \in E \) , implies that entity  \( v_{j} \)  references entity  \( v_{i} \)  in its content statement, proof process, or solution

Table 2 Samples of different types of entities storage.

|  Definition: Symmetric Group  |   |
| --- | --- |
|  Contents | Let S be a set. Let Γ(S) denote the set of permutations on S. Let (Γ, o) be the algebraic structure such that o denotes the composition of mappings. Then (Γ, o) is called the symmetric group on S. If S has n elements, then (Γ, o) is often denoted Sn.  |
|  Bodylist | [{"description": "Let S be a set.", "action": "premise"}, {"description": "Let Γ(S) denote the set of permutations on S. Let (Γ, o) be the algebraic structure such that o denotes the composition of mappings. Then (Γ, o) is called the symmetric group on S. If S has n elements, then (Γ, o) is often denoted Sn.", "action": "definition"}]  |
|  References_tactics | {"Definition:Composition of Mappings": "definition", "Definition:Algebraic Structure": "definition", "Definition:Element": "definition", "Definition:Permutation": "definition", "Definition:Set": "premise"}  |
|  Theorem: Center of Symmetric Group is Trivial  |   |
|  Contents | Let n ∈ N be a natural number. Let Sn denote the symmetric group of order n. Let n ≥ 3. Then the center Z(Sn) of Sn is trivial.  |
|  Bodylist | [{"description": "Let n ∈ N be a natural number.", "action": "premise"}, {"description": "Let Sn denote the symmetric group of order n. Let n ≥ 3.", "action": "assumption"}, {"description": "Let n ≥ 3. Then the center Z(Sn) of Sn is trivial.", "action": "conclusion"}]  |
|  References_tactics | {"Definition:Order of Structure": "premise", "Definition:Center (Abstract Algebra)/Group": "premise", "Definition:Natural Numbers": "premise", "Definition:Trivial Group": "premise", "Definition:Symmetric Group": "premise"}  |
|  Problem: Positive Definiteness of a Matrix  |   |
|  Contents | Consider the matrix of A = [[1,4], [4,1]], is this a positive definite matrix?  |
|  References_tactics | {"definition:positive definite matrix": "deduction"}  |

process. Edges can point to any type of entity, but they always originate from a Definition or Theorem entity. These directed edges form logical or inferential chains among mathematical knowledge, demonstrating the dependency relationships and deduction processes between different mathematical entities.

All references between entities form a directed graph, which may contain cycles. For example, Pythagoras's Theorem and Sum of Squares of Sine and Cosine are mutually referenced in their proofs. Figure 5 illustrates all possible relationships between different types of entities. Additionally, to clarify the tactic role played by entities when referenced, we design nine tactic labels: premise, assumption, lemma, proposition, corollary, calculation, enumeration, definition, and conclusion. Each edge includes tactic label information, stored in the "in_refs" and "out_refs" fields of the entity JSON structure.

### 4.3 Information extraction and augmentation

This subsection will detail the instruction on how to extract and augment information to construct an Input KG according to the input text from a technical perspective.

![img-4.jpeg](img-4.jpeg)

Fig. 5 The diagram of all possible relationships between different types of entities.

### Step 1: Rule-based extraction

For each $\LaTeX$ source code of the input text, rule-based matching method [12], depending on $\LaTeX$ environment names, is conducted to extract basic information of different types of entities. Specifically, it searches for predefined environment names and parses the content contained in these environments into basic information corresponding to the types of entities. For instance, the content surrounded by \begin{theorem} and \end{theorem} in the input text will be parsed as the content of Theorem entities in AutoMathKG. After rule-based matching, most of the following attribute information of entities can be obtained for the current input text: “id”, “type”, “label”, “title”, “contents”, “refs”, “source”, “proofs” and “solutions”.

### Step 2: LLM-based augmentation

After rule-based matching, each entity undergoes content and relationship augmentation by LLM using ICL to complete the remaining attribute information. Specifically, for attributes such as “title”, “field”, “bodylist”, “references_tactics” and “refs”, prompt templates are designed to ask LLM for the information corresponding to each attribute. Considering the different requirements of attribute prompt templates for different entity types, we design a total of 12 prompt templates (see Table C4 in Appendix C for more details). Finally, the fully augmented information of each entity is stored in JSON format, resulting in an Input KG generated from the current input text.

## 5 Method of MathVD construction

Given that similarity searches through graph traversal become increasingly expensive as the scale of the KG expands, we build a VD for AutoMathKG, denoted as MathVD. Through vectorizing the query content and the KG, the similarity between entities is measured through the cosine similarity of their respective vectors, enabling fuzzy search for math knowledge. SBERT [19] is employed to embed the descriptions of math entities, generating 384-dimensional entity vectors. Two strategies for embedding

entity vectors are designed, differing in how descriptive statements about entities are constructed, resulting in two VDs: MathVD1 and MathVD2.

### 5.1 MathVD1

For each entity, MathVD1 uses SBERT to embed a long text containing all key information about the entity to generate its vector. First, it is necessary to determine what information is vital to represent math entities. For any entity node v, five important fields containing entity features from its JSON format storage are selected, namely “title”, “field”, “contents”, “in_refs”, and “out_refs”. The values stored in each field are obtained and presented as descriptive sentences  \( S_{1}(v), \cdots, S_{5}(v) \) , given by

\[
S _ {1} (v) = \text {"title:} v [ t i t l e ] ^ {\prime \prime},
\]

\[
S _ {2} (v) = \text {"field:} v [ f i e l d ] ^ {\prime \prime},
\]

\[
S _ {3} (v) = \text {"content:} v [ c o n t e n t s ] ^ {\prime \prime},
\]

\[
S _ {4} (v) = \text {"in references:} v [ i n \_ r e f s ] ^ {\prime \prime},
\]

\[
S _ {5} (v) = \text {"out references:} v [ o u t \_ r e f s ] ^ {\prime \prime}, \tag {1}
\]

where v[key] represents the value of the key field for the entity v stored in KG. Subsequently, MathVD1 concatenates each descriptive sentence sequentially to form a long text  \( S(v) \)  containing all key information about the entity, given by

\[
S (v) = S _ {1} (v) + S _ {2} (v) + S _ {3} (v) + S _ {4} (v) + S _ {5} (v). \tag {2}
\]

Finally, the long sentence \( S(v) \) is embedded by SBERT, obtaining the vector representation \( e(v) \in \mathbb{R}^{384} \).

MathVD1 considers  \( e(v) \)  as the vector embedding of the entity v. Table 3 illustrates an instance of similarity retrieval for the target entity in MathVD1, where cosine similarity is utilized as the similarity metric, indicating significant relevance of the retrieved content to the target entity.

Table 3 The result of a similar entity retrieval example in MathVD1.

|  Target entity: Set union is associative: \( A \cup (B \cup C) = (A \cup B) \cup C \)  |   |   |
| --- | --- | --- |
|  Retrieval results in MathVD1  |   |   |
|  Rank | Similar entity | Score  |
|  1 | Theorem: Set union is commutative: \( S \cup T = T \cup S \) | 0.8977  |
|  2 | Theorem: Symmetric difference is associative: \( R\triangle(S\triangle T) = (R\triangle S)\triangle T \) | 0.8300  |
|  3 | Theorem: Set intersection is associative: \( A \cap (B \cap C) = (A \cap B) \cap C \) | 0.8217  |

### 5.2 MathVD2

For each entity, MathVD2 uses SBERT to separately embed each key information description of the entity, then weights and sums these vectors based on their respective importance to generate the entity vector. First, we need to determine the descriptive

sentence of key information for each entity. For any entity node v, the same method as described in Section 5.1 is used to obtain key information, resulting in five sentences  \( S_{1}(v), \cdots, S_{5}(v) \) , containing “title”, “field”, “contents”, “in_refs” and “out_refs” information, according to Equation 1. Subsequently, each sentence is embedded by SBERT, obtaining corresponding vectors  \( e_{i}(v) \in \mathbb{R}^{384} \)  for each sentence  \( S_{i}(v), i = 1, \cdots, 5 \) . Finally, these vectors are weighted and summed based on the importance of the information. Specifically, let the weight vector W be defined as follows:

\[
W = (w _ {1}, w _ {2}, w _ {3}, w _ {4}, w _ {5}),
\]

where \( w_{j} \) represents the weight assigned to the \( j \)-th information, satisfying \( \sum_{j=1}^{5} w_{j} = 1 \). Then the weighted sum of all vectors is computed by

\[
\tilde {e} (v) = w _ {1} \cdot e _ {1} (v) + w _ {2} \cdot e _ {2} (v) + w _ {3} \cdot e _ {3} (v) + w _ {4} \cdot e _ {4} (v) + w _ {5} \cdot e _ {5} (v). \tag {3}
\]

MathVD2 considers  \( \tilde{e}(v) \)  as the vector embedding of the entity v. Since “contents” can best differentiate between entities, followed by “title”, we set the weight vector as  \( W = (0.5, 0.3, 0.1, 0.05, 0.05) \)  in our work. Similarly, Table 4 illustrates a similarity retrieval instance for the same target entity in MathVD2, also indicating significant relevance to the target entity. Comparing Tables 3 and 4, we can see that both MathVD1 and MathVD2 are effective in retrieving similar entities, although there are slight differences in the retrieval results. Besides, some similar math Problem entities with the same mathematical essence exhibit high cosine similarity between each other in both MathVDs, shown as Table 5.

Table 4 The result of a similar entity retrieval example in MathVD2.

|  Target entity: Set union is associative: \( A \cup (B \cup C) = (A \cup B) \cup C \)  |   |   |
| --- | --- | --- |
|  Retrieval results in MathVD2  |   |   |
|  Rank | Similar entity | Score  |
|  1 | Theorem: Set union is commutative: \( S \cup T = T \cup S \) | 0.8034  |
|  2 | Theorem: Set union is idempotent: \( S \cup S = S \) | 0.7662  |
|  3 | Theorem: Set intersection is associative: \( A \cap (B \cap C) = (A \cap B) \cap C \) | 0.7600  |

Table 5 The similarity results of similar Problem entity pairs in MathVD1 and MathVD2.

|  Similar Problem entity pair | Similarity score  |   |   |
| --- | --- | --- | --- |
|   |   |  MathVD1 | MathVD2  |
|  Pair 1 | Entity 1: Consider the matrix of \( A = [[1,4], [4,1]] \), is this a positive definite matrix? | 0.9346 | 0.9518  |
|   |  Entity 2: Consider the matrix of \( A = [[1,-1], [-1,4]] \), is this a positive definite matrix?  |   |   |
|  Pair 2 | Entity 3: Are the vectors [1,2], [2,3], and [3,4] linearly independent? | 0.8291 | 0.8846  |
|   |  Entity 4: Are the vectors \( v_1 = [1,2,3], v_2 = [4,5,6], v_3 = [7,8,9] \) linearly independent?  |   |   |

## 6 Method of automatic updates

In this section, we propose two mechanisms for automatically updating our math KG, in order to fully leverage existing mathematical knowledge and maintain synchronization with ongoing advancements in the field. In the first mechanism, a specialized mathematical LLM for solving various math problems is designed, named Math LLM, to achieve automatic knowledge completion by supplementing incomplete proofs or solutions. In the second mechanism, VDs and LLM are utilized to realize automatic knowledge fusion by merging and updating new entities and their relationships from different sources.

### 6.1 Automatic knowledge completion

Given that small-scale LLMs with fewer parameters tend to perform less effectively in mathematical reasoning, this paper proposes Math LLM, specifically designed to address various types of mathematical problems through the integration of task adapters, retrieval augmentation, and self-calibration. Figure 6 illustrates the structure of Math LLM. First, for each mathematical problem input, preliminary information on problem types and knowledge fields is generated. Next, a CoT adapter agent analyzes the information to determine the most appropriate adapter for problem-solving. Meanwhile, relevant knowledge is retrieved from AutoMathKG and MathVD through a two-stage retrieval augmentation. Subsequently, answers are generated using the relevant knowledge as prompts, with the adapter activated. Finally, the self-calibration module reviews and outputs calibrated answers.

![img-5.jpeg](img-5.jpeg)

Fig. 6 Model architecture of Math LLM.

#### 6.1.1 Task adapters

An adapter is a parameter-efficient fine-tuning strategy that provides plug-and-play functionality, stability, and generalization. It introduces lightweight layers into the model, which are activated only when necessary, allowing the majority of the model

parameters to remain unchanged. Given that, Math LLM employs three types of mathematical expert adapters to address three categories of problems: application, calculation, and proof, utilizing the LORA [55], PoT [29], and CoT [28] approaches, respectively. The application adapter, trained on an application problem dataset, enables the model to abstract real-world problems into mathematical formulations. The calculation adapter, trained on a programming dataset, equips the model with the capability to write Python programs to solve calculation problems, significantly reducing hallucination issues commonly encountered by LLMs in calculations. The proof adapter, trained on a mathematical theorem proof dataset, assists the model in developing a rigorous mathematical reasoning process, thereby strengthening its theoretical framework and logical structure. Finally, a CoT agent is integrated into Math LLM to coordinate these adapters, selecting the most appropriate one for problem-solving (see Appendix D for more training details).

### 6.1.2 Retrieval augmentation

Retrieving knowledge from external databases provides LLMs with verified information, enabling more grounded reasoning and mitigating hallucinations. Thus, Math LLM is designed to interact with AutoMathKG and MathVD for specialized knowledge retrieval. First, Math LLM analyzes the input problem and generates a set of relevant unverified knowledge points, including theorems and definitions. Next, Math LLM extracts relevant knowledge from both databases based on these knowledge points. Given that exact searches are resource-intensive and fuzzy searches may introduce less precise results, we designed a balanced retrieval augmentation strategy comprising two stages: exact search and fuzzy search. In the first stage, an exact search is performed in AutoMathKG to match Definition and Theorem entities directly to the knowledge points. In the second stage, a fuzzy search is conducted in MathVD to identify entities similar to the knowledge points and those found in the first stage within the vector space. After this, verified theorem and Definition entities are extracted. Similarly, the two-stage retrieval process is used to search for relevant Problem entities across both databases. Finally, all verified auxiliary knowledge entities are packaged and provided to Math LLM, which selects those contributing to the problem-solving process as the final specialized knowledge.

### 6.1.3 Self-calibration

A self-calibration module can effectively reduce hallucinations [56], by providing a feedback mechanism for LLMs. Given that, this paper introduces a self-calibration module within Math LLM. Since engaging in trial and error is a common approach for solving unfamiliar and complex problems, our self-calibration module refines the answer incrementally based on CoT. We implement specific principles for calibration, with different types of problems adhering to distinct rules. The model verifies these rules step by step. If the model passes the self-calibration, it outputs the final answer. Otherwise, it identifies the violated principles and attempts to correct the errors. The corrected answer is then fed back into Math LLM for another round of self-calibration. If it still fails, Math LLM generates an error summary, which is combined with the

original problem and used as a prompt to initiate another cycle of answer revision until it passes the self-calibration module.

## 6.2 Automatic knowledge fusion

Considering that mathematical corpora from different sources may have different statements of the same knowledge, we design a mechanism to automatically achieve knowledge fusion. For each new math entity from other sources, we utilize VD to search for similar candidate entities of Existing KG and employ LLM to determine whether to merge with a candidate or add the new entity. Given new input text, the following steps are taken for automatic knowledge fusion.

### *Step 1: Construction of Input KG from new input*

Information extraction and augmentation are conducted on the new input text as described in Section 4.3, for entity recognition preprocessing and knowledge enhancement, generating Input KG from the current input.

### *Step 2: Construction of VDs from Input KG and Existing KG*

Entities of Input KG and Existing KG are embedded using SBERT as described in Section 5, constructing two VDs, named Input VD and Existing VD, respectively.

### *Step 3: Fuzzy search for similar candidate entities in VD*

For each input entity $v_{input}$, a fuzzy search for its similar entities is conducted by comparing the cosine similarity between their entity vectors. Specifically, let $e_{input}$ be the vector embedding of $v_{input}$ in Input VD. Then, for each entity vector in Existing VD, calculate the cosine similarity with $e_{input}$. Finally, sort them in descending order and select the top $n$ entity vectors as the similar candidate entities for the input entity $v_{input}$.

### *Step 4: Determination of entity consistency by LLM*

After obtaining $n$ similar candidate entities, LLM is used to determine whether $v_{input}$ is consistent with any of the candidates. Specifically, we employ LLM through ICL to determine the consistency (see Table C5 in Appendix C for more details). For each similar candidate entity $v_i, i = 1, \dots, n$, LLM is asked to output “yes” if it is consistent to $v_{input}$; otherwise, LLM outputs “no”.

### *Step 5: Selection of update approach by LLM*

Finally, we analyze the consistency results and select the appropriate update approach for each input entity. If LLM outputs “no” for all similar candidate entities, $v_{input}$ will be considered as a new entity and added to the Existing KG. If LLM outputs “yes” for only one candidate, $v_{input}$ will be merged with it, with reference relationships between $v_{input}$ and other entities in Input KG being added. If LLM outputs “yes” for multiple candidates, LLM will perform a secondary judgment through ICL on all the candidates with “yes” output, selecting the most similar candidate entity (see Table C5 in Appendix C for more details). Then, $v_{input}$ will be merged with that candidate, with relationships being added.

## 7 Experiments

### 7.1 Settings

As for the construction of AutoMathKG, the Llama-2-7b model \( ^{3} \) was employed for entity augmentation and all ICL tasks. As for the construction of MathVD, the all-MiniLM-L6-v2 model \( ^{4} \) from SBERT [19] was utilized. When it comes to the automatic update with MathVD, the number of retrieved similar candidate entities was set to 5, and cosine similarity was employed as the search metric. As for the construction of Math LLM, the gemma-7b-it model \( ^{5} \) was selected as the base model, a lightweight model that demonstrates excellent performance across various tasks [49]. Initially, the model was fine-tuned on a comprehensive mathematical dataset to develop its basic mathematical problem-solving capabilities. Subsequently, task adapters were trained on different types of datasets. Detailed information regarding the training datasets and hyperparameters is provided in Appendix D. All experiments were conducted on a computer equipped with four RTX 3090 GPUs and 96GB of VRAM, utilizing Python 3.10.

### 7.2 Results

Following the experimental procedure described above, we constructed and updated AutoMathKG. Figure 7 illustrates the visualization results generated using the PiVis library, where yellow, blue, and red nodes represent Definition, Theorem, and Problem entities, respectively. An interactive graph for AutoMathKG was also created, with title information displayed on the nodes, allowing users to zoom in and out, as well as move nodes and edges to explore the interconnections between entities \( ^{6} \) . AutoMathKG encompasses a broad range of mathematical fields, including Analysis, Algebra, Geometry, Logic, Probability, Statistics, and others, as shown in Fig. 8. Moreover, detailed information regarding the graph structure of AutoMathKG is presented in Table 6. The graph is directed and comprises 13,388 entity vertices and 29,459 directed edges. The distribution of head and leaf nodes indicates that many knowledge paths within AutoMathKG start from basic concepts, gradually deepen, and eventually reach specific conclusions. Additionally, the presence of numerous cycles highlights complex relationships and mutual dependence among mathematical knowledge, where simple cycles represent the most fundamental and direct interrelationships.

Table 6 The statistics of graph structure for AutoMathKG.

|  Node | All nodes | Head node | Leaf node  |
| --- | --- | --- | --- |
|  Number | 13388 | 4216 | 3789  |
|  Edge | All edges | Cycle | Simple cycle  |
|  Number | 29459 | 82153 | 145  |

![img-6.jpeg](img-6.jpeg)

Fig. 7 The visualization of AutoMathKG, with yellow, blue, and red nodes representing Definition, Theorem, and Problem entities.

### 7.3 Evaluation

#### 7.3.1 Reachability query

To assess the capability of the constructed VDs in capturing structural information from the KG, we conducted reachability query experiments to evaluate whether structurally related entities in the KG could be effectively retrieved based on the similarity between the entity embeddings. Specifically, we considered the k-hop reachability query between entities. Two entity nodes A and B are considered k-hop reachable if they can be connected by no more than k directed edges in the graph. For any entity vector e, we retrieved the top q most similar entities using cosine similarity and counted the number of k-hop reachable entities, denoted as r. The hit rate for q queries (Hits@q) is defined as follows:

\[
H i t s @ q = \frac {r}{q}. \tag {4}
\]

We compared MathVD against five baselines, TransE [41], KG2E [42], HoLE [43], R-GCN [44], and BoxE [45], covering different kinds of KG embedding models. We used PyKEEN \( ^{7} \) to implement them, where the vector dimension was set to 384, the same as MathVD. In the experiments, we randomly sampled 100 entities based on the category proportions, including 50 Definition entities, 30 Theorem entities, and 20 Problem entities. Main results of Hits@q for 5-hop reachability query using baselines and our models are shown in Table 7. The best results are highlighted in bold.

The results indicate that for query numbers q = 5, 10, and 15, both MathVD models outperform all baseline models on Hits@q. This suggests that in our constructed VD, most retrieval queries based on cosine similarity are structurally related to the target

![img-7.jpeg](img-7.jpeg)

Fig. 8 Mathematical field distribution in AutoMathKG.

Table 7 The results of Hits@q for 5-hop reachability query with q = 1, 5, 10, and 15.

|  Model | Hits@1 | Hits@5 | Hits@10 | Hits@15  |
| --- | --- | --- | --- | --- |
|  TransE | 0.9610 | 0.7766 | 0.7182 | 0.6797  |
|  KG2E | 0.7403 | 0.7325 | 0.7156 | 0.7091  |
|  HoLE | 0.8442 | 0.7688 | 0.7312 | 0.7022  |
|  R-GCN | 0.7273 | 0.7065 | 0.6948 | 0.6840  |
|  BoxE | 0.9351 | 0.8338 | 0.7610 | 0.7247  |
|  MathVD1 | 0.8831 | 0.8364 | 0.8182 | 0.7861  |
|  MathVD2 | 0.8974 | 0.8385 | 0.8013 | 0.7786  |

![img-8.jpeg](img-8.jpeg)

(a) MathVD1

![img-9.jpeg](img-9.jpeg)

(b) MathVD2

Fig. 9 The influence of hyperparameter k on Hits@q for MathVD1 and MathVD2.

entities in the KG, showing the effectiveness of our proposed embedding strategies for mathematical entities. Although our VDs are slightly surpassed by the classic TransE model when q = 1, their performance is notably superior at higher q values. This is primarily because TransE is trained by predicting tail entities, which specifically optimizes for 1-hop reachability queries. Consequently, for a single query, TransE achieves a higher hit rate through supervised learning, but exhibits weaker generalization as the query number increases. In contrast, our model generalizes well across varying query numbers. Additionally, comparing our two VDs, it is found that MathVD2 performs slightly better for smaller q, while MathVD1 shows a little higher Hits@q for larger q.

There is one hyperparameter, namely the hop of reachability query denoted as $k$. $k$-hop reachability queries with different values of $k$ reflect varying degrees of association between entities. For $k = 1$, the 1-hop reachability query is equivalent to the common link prediction task. For $k > 1$, $k$-hop reachability queries could provide additional information about the relationship between two entities, such as multi-hop inference. Generally, entities that are reachable within smaller $k$ hops tend to exhibit stronger associations. Figure 9 illustrates the influence of hyperparameter $k$ on Hits@$q$ for MathVD1 and MathVD2, with $q = 1$, 5, 10 and 15. Initially, Hits@$q$ shows a sharp increase as $k$ increases from very small values; however, after $k$ increases to a certain threshold, the improvement becomes less significant. This indicates that most retrieved $k$-hop reachable entities can actually be accessed via smaller hops, resulting in increasing $k$ beyond this threshold having little effect on the hit rate. This not only highlights the insensitivity of the hit rate to larger values of the hyperparameter $k$, but also underscores the effectiveness of using cosine similarity to retrieve structurally related entities from our VDs.

### 7.3.2 Math LLM reasoning capability

To assess the reasoning behavior of our designed Math LLM, we developed a framework for evaluating the mathematical capabilities of LLMs, based on an extensive literature review. Specifically, we rated the responses generated by the LLM on a scale from 1 to 5, with 5 representing the highest rating. The scoring criteria are as follows: a rating of 5 is assigned when the answer is nearly perfect; 4 is assigned when the model provides a mostly correct answer but with some issues, such as the use of unnecessarily complex methods; 3 is assigned when the model demonstrates correct reasoning but makes significant errors in execution; 2 is assigned when only a small part of the reasoning process is correct; and 1 is assigned when the output is largely irrelevant to the correct answer.

In the experiments, 234 test questions were selected and refined from GHOSTS [30], categorized into six areas: Complement, Prealgebra, Algebra, Theorem, Probability, and Topology. Notably, the Complement category involves questions where parts of the proofs are intentionally omitted, requiring the LLM to fill in the gaps, which effectively aligns with the task of automatic knowledge completion by Math LLM. The evaluation was conducted using an online question-and-answer format, where the dataset question served as the prompt and the feedback was recorded in the output. Subsequently, we manually rated the answers generated by Math LLM according to the established framework. The sample size and average rating for each category of test questions are shown in Table 8, and the rating distribution is illustrated in Fig. 10.

Table 8 The average evaluation rating of Math LLM.

|  Question category | Complement | Prealgebra | Algebra | Theorem | Probability | Topology  |
| --- | --- | --- | --- | --- | --- | --- |
|  Sample size | 35 | 43 | 46 | 27 | 44 | 39  |
|  Average rating | 3.3714 | 3.2558 | 3.7826 | 3.3704 | 3.0455 | 3.0513  |

![img-10.jpeg](img-10.jpeg)

Fig. 10 Rating distribution in test questions.

The evaluation results indicate that Math LLM is capable of demonstrating correct reasoning and providing mostly correct answers in most cases, particularly for challenging problems involving theorem proving. This performance is commendable for a model with 7 billion parameters. Leveraging the various modules we have designed, Math LLM exhibits strong understanding, proof, and computational capabilities, thereby demonstrating its reliability for automatic knowledge completion.

#### 7.3.3 Internal retrieval relevance

To assess the internal quality of our VD, we conducted human evaluations to quantify the precision of relevance in retrieving similar entities across both VDs. For any entity vector e, the top t entities ranked by cosine similarity were retrieved as predictions of t similar entities. If the retrieved entity is related to the target entity in terms of content, domain, or application, it is considered a correct similar prediction. Let s denote the number of correct predictions for entity vector e. The precision p of similar entity retrieval for entity vector e is defined as follows:

\[
p = \frac {s}{t}. \tag {5}
\]

In the experiments, 50 random test samples were drawn from each VD based on the category proportions, including 25 Definition entities, 15 Theorem entities, and 10 Problem entities. These samples are denoted as  \( \{e_{1}^{(1)},\cdots,e_{50}^{(1)}\} \)  for MathVD1, and  \( \{e_{1}^{(2)},\cdots,e_{50}^{(2)}\} \)  for MathVD2 respectively. Similar entity retrieval was performed for each sample, with t=10. After manually evaluating the prediction results, the precision of similar entity retrieval for each sample was calculated, resulting in two series of precision data,

\[
P ^ {(1)} = \left\{p _ {1} ^ {(1)}, \dots , p _ {5 0} ^ {(1)} \right\}, P ^ {(2)} = \left\{p _ {1} ^ {(2)}, \dots , p _ {5 0} ^ {(2)} \right\}. \tag {6}
\]

Figure 11 presents box plots of  \( P^{(1)} \)  and  \( P^{(2)} \) , visually illustrating the detailed distribution of two sets of precision data, both of which are concentrated around 1. A Kolmogorov-Smirnov (K-S) test was conducted on  \( P^{(1)} \)  and  \( P^{(2)} \) , indicating no significant difference in the distribution between the two sets of data. Additionally, Table 9 compares the average precision with the standard deviation (STDEV) of similar entity retrieval within MathVD1 and MathVD2. The precision of similar retrieval for all three types of entities exceeds 90%, with MathVD1 exhibiting better performance in the retrieval of Definition and Problem entities, while performing equally well as MathVD2 in Theorem entity retrieval. The average precision of MathVD1 and MathVD2 are both around 95%, with relatively small fluctuations. Therefore, MathVD1 and MathVD2 are both effective in capturing the relationships among internal entities, enabling the retrieval of relevant entities.

![img-11.jpeg](img-11.jpeg)

Fig. 11 The box plots of MathVD1 and MathVD2.

Table 9 The comparison results of average precision for similar entity retrieval in MathVD1 and MathVD2.

|  Sample type | Def | Thm | Prob | All (STDEV)  |
| --- | --- | --- | --- | --- |
|  MathVD1 | 96.0% | 94.0% | 97.0% | 95.2% (0.0953)  |
|  MathVD2 | 95.3% | 94.0% | 96.0% | 94.8% (0.1035)  |

#### 7.3.4 External retrieval relevance

To assess the external applicability of our VD, we conducted experiments on similar entity retrieval for external queries to evaluate their relevance. First, the queries are embedded into vectors using SBERT, after which similarity retrieval in MathVD is performed, similar to the internal entity retrieval process. Tables 10 and 11 present the retrieved entity contents for the same external query using MathVD1 and MathVD2, respectively. For the query of “Expectation in Probability and Statistics”, both MathVD1 and MathVD2 retrieve definitions and theorems related to “Expectation”, while MathVD1 also retrieves relevant problems, assisting the query requester in understanding this mathematical concept from different perspectives. Therefore, for

queries from external sources, both of our VDs are capable of retrieving various types of highly relevant entities, indicating their effectiveness in handling external queries.

Table 10 The results of external query example in MathVD1.

|  External query: Expectation in Probability and Statistics  |   |
| --- | --- |
|  Retrieval results in MathVD1  |   |
|  Rank | Entity  |
|  1 | Theorem: expectation of geometric distribution Let $X$ be a discrete random variable with the geometric distribution with parameter $p$. Then the expectation of $X$ is given by: $E(X) = \frac{1-p}{p}$.  |
|  2 | Problem: expectation of waiting time Let $X_0, X_1, X_2, \cdots$ be drawn i.i.d. from $p(x)$, and $x \in \{1, 2, 3, \cdots, 100\}$. Let $N$ be the waiting time to the next occurrence of $X_0$. Compute $E(N)$.  |
|  3 | Theorem: expectation of discrete uniform distribution Let $X$ be a discrete random variable with the discrete uniform distribution with parameter $n$. Then the expectation of $X$ is given by: $E(X) = \frac{n+1}{2}$.  |
|  4 | Definition: expectation of random vector Let $X_1, X_2, \cdots, X_n$ be random variables on a probability space $(\Omega, \Sigma, Pr)$. Let $X = (X_1, X_2, \cdots, X_n)$ be a random vector. Then the expected value of $X$, $E(X)$, is defined by: $E(X) = (E(X_1), E(X_2), \cdots, E(X_n))$.  |
|  5 | Theorem: sum of expectations of independent trials: Let $E_1, E_2, \cdots, E_n$ be a sequence of experiments whose outcomes are independent of each other. Let $X_1, X_2, \cdots, X_n$ be discrete random variables on $E_1, E_2, \cdots, E_n$ respectively. Let $E(X_j)$ denote the expectation of $X_j$ for $j \in \{1, 2, \cdots, n\}$. Then we have, whenever both sides are defined: $E(\sum_{j=1}^n X_j) = \sum_{j=1}^n E(X_j)$. That is, the sum of the expectations equals the expectation of the sum.  |

Table 11 The results of external query example in MathVD2.

|  External query: Expectation in Probability and Statistics  |   |
| --- | --- |
|  Retrieval results in MathVD2  |   |
|  Rank | Entity  |
|  1 | Definition: expectation The expectation of a random variable is the arithmetic mean of its value.  |
|  2 | Theorem: expectation of binomial distribution Let $X$ be a discrete random variable with the binomial distribution with parameters $n$ and $p$ for some $n \in \mathbb{N}$ and $0 \le p \le 1$. Then the expectation of $X$ is given by: $E(X) = np$.  |
|  3 | Theorem: expectation of geometric distribution Let $X$ be a discrete random variable with the geometric distribution with parameter $p$. Then the expectation of $X$ is given by: $E(X) = \frac{1-p}{p}$.  |
|  4 | Theorem: sum of expectations of independent trials: Let $E_1, E_2, \cdots, E_n$ be a sequence of experiments whose outcomes are independent of each other. Let $X_1, X_2, \cdots, X_n$ be discrete random variables on $E_1, E_2, \cdots, E_n$ respectively. Let $E(X_j)$ denote the expectation of $X_j$ for $j \in \{1, 2, \cdots, n\}$. Then we have, whenever both sides are defined: $E(\sum_{j=1}^n X_j) = \sum_{j=1}^n E(X_j)$. That is, the sum of the expectations equals the expectation of the sum.  |
|  5 | Theorem: expectation of bernoulli distribution Let $X$ be a discrete random variable with a Bernoulli distribution with parameter $p$. Then the expectation of $X$ is given by: $E(X) = p$.  |

## 8 Discussion

### 8.1 Comparison with existing mathematical KGs

There are significant advantages in AutoMathKG in terms of corpus sources, entity coverage and update methods, compared to existing mathematical KGs written in natural language, including MathGraph [11], MathGloss [14], Math-KG [13], and NaturalProofs [12], as shown in Table 12.

For corpus sources, AutoMathKG not only includes commonly covered web data and multiple high-quality textbooks, but also incorporates synthetic data and mathematical paper data that other mathematical graphs lack. The synthetic data augmented by LLM provides more detailed mathematical knowledge, while mathematical papers offer deeper domain-specific knowledge, giving AutoMathKG an advantage in richness and depth of math knowledge. For entity coverage, AutoMathKG breaks the limitation of existing mathematical graphs focusing on only one aspect of mathematical concepts or problems. It views mathematics as a vast multidimensional network composed of definitions, theorems, and problems, fully exploring the relationships between them, and generating a directed graph that captures the interdependencies of mathematical knowledge. For update methods, similar to existing mathematical graphs, AutoMathKG can integrate knowledge from different sources. The difference lies in the ability of AutoMathKG to automatically supplement incomplete knowledge and add it to the existing graph. Considering the abundance of incomplete knowledge in existing mathematical text corpora, AutoMathKG effectively enhances the utilization of existing corpus data and reduces human resources through LLM automatic updates.

Table 12 The comparison results between AutoMathKG and existing math KGs.

|  KG | Corpus source |   |   |   |   | Entity coverage |   |   | Update method  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|   |  Online | Synthetic | Exercise | #Textbook | #Paper | Thm | Def | Prob | Automated  |
|  MathGraph | - | - | √ | - | - | √ | √ | √ | -  |
|  MathGloss | √ | - | - | - | - | - | √ | - | -  |
|  Math-KG | √ | - | - | - | - | √ | √ | - | -  |
|  NaturalProofs | √ | - | - | 2 | - | √ | √ | - | -  |
|  AutoMathKG | √ | √ | √ | 8 | 20 | √ | √ | √ | √  |

### 8.2 Ablation study for knowledge representation

This paper proposes an innovative approach to knowledge representation for MathVD by leveraging the dependency relationships between entities within the AutoMathKG graph structure. To evaluate the contribution of reference information from incoming and outgoing entities in entity embeddings, we conducted an ablation study by removing this reference information to observe the resulting changes in performance for similar vector retrieval. Specifically, for each entity node, the descriptive sentences corresponding to the “in_refs” and “out_refs” attributes in Section 5 were removed, and only those corresponding to the “title”, “field” and “contents” attributes were used for vector representation, resulting in a VD variant, denoted as VD-no refs.

Table 13 shows a comparison of the performance in similar entity retrieval between the variant and the original MathVD using the first vector representation method proposed in Section 5.1, denoted as VD1-no refs and VD1-all, respectively. For the target Theorem entity “probability measure is subadditive”, the top 5 similar entities retrieved in VD1-no refs are all intuitively related to the target entity in the field, with most of their entity titles containing “probability measure” or “probability”. In contrast, the first entity retrieved in VD1-all seems not similar to the target entity in terms of title, making it difficult for people from non-mathematical fields to identify the connection between them based on the title. However, upon examining the content of this entity, it is found to be a direct consequence of the target entity. This indicates that VD1-all can retrieve entities that are similar in essence to the target entity, not solely relying on linguistic similarity. Furthermore, another example in Table E6 of Appendix E shows that the cosine similarity between similar entities in VD1-all is higher overall, indicating denser similar entities in the vector space of VD1-all. The reason for this advantage is that VD1-all contains the interdependence between entities and stores it in the vector representation. These results demonstrate that the reference information provided by incoming and outgoing entities plays a crucial role in the vector representation of mathematical knowledge, enabling to retrieve essentially relevant entities.

Table 13 The results of ablation study.

|  Theorem: probability measure is subadditive Let (Ω, Σ, Pr) be a probability space. Then Pr is a subadditive function.  |   |   |
| --- | --- | --- |
|  Retrieval results in VD1-no refs  |   |   |
|  Rank | Score | Entity  |
|  1 | 0.7632 | Theorem:elementary properties of probability measure  |
|  2 | 0.7362 | Definition:probability measure/definition 1  |
|  3 | 0.7324 | Definition:probability function  |
|  4 | 0.6869 | Definition:probability measure/definition 2  |
|  5 | 0.6343 | Probability measure is monotone  |
|  Retrieval results in VD1-all  |   |   |
|  Rank | Score | Entity  |
|  1 | 0.7632 | Theorem: boole's inequality  |
|  2 | 0.7237 | Measure is subadditive  |
|  3 | 0.6980 | Definition:probability function  |
|  4 | 0.6960 | Definition:probability measure/definition 1  |
|  5 | 0.6827 | Theorem:elementary properties of probability measure  |

### 8.3 Case study for automatic updates

#### 8.3.1 Knowledge completion

In order to achieve automatic knowledge completion, this paper utilizes the designed Math LLM to supplement incomplete proofs and solutions for new theorems and problems. Table 14 shows the supplementation effects of theorem proofs and problem solutions by Math LLM, where the incomplete entities are packaged as Question sent to Math LLM, and the supplementation knowledge is output as Answer.

**Table 14** The results of the case study on automatic knowledge completion.

|  Theorem proof  |   |
| --- | --- |
|  Question | Let $f$ have a continuous derivative on $[a, b]$. We want to show that $f'$ satisfies a Lipschitz condition on $[a, b]$. By [MISSING], $f(x) - f(y) = f'(x_i)(x - y)$ for some $x_i \in (y, x)$. But since $f'$ is continuous and $[a, b]$ is compact, then $f'$ is bounded in that interval, say by $C$. Thus taking absolute values yields $|f(x) - f(y)| \leq C|x - y|$. Can you name the result used in [MISSING]?  |
|  Answer | The answer is the Mean Value Theorem. The missing part in the text is the Mean Value Theorem. The Mean Value Theorem states that for a continuous function $f$ on the interval $[a, b]$, there exists a point $c$ in $(a, b)$ such that $f(b) - f(a) = f'(c)$ $(b - a)$. This theorem is used to show that $f$ satisfies a Lipschitz condition on $[a, b]$.  |
|  Problem solution  |   |
|  Question | What is the smallest positive multiple of 6 that is also a multiple of 8? Solution: To find a positive number that is both a multiple of 6 and a multiple of 8, we'll list out the multiples of 6, starting with 6, and check to see if any of them are also multiples of 8. 6 does not divide 8, so it isn't a multiple of 8. 12 and 18 are also not evenly divisible by 8. [MISSING]. Fill in the missing steps.  |
|  Answer | Sure, here is the answer to the question: To find the smallest positive multiple of 6 that is also a multiple of 8, we'll list out the multiples of 6, starting with 6, and check to see if any of them are also multiples of 8. 6 does not divide 8, so it isn't a multiple of 8. 12 and 18 are also not evenly divisible by 8. The next multiple of 6 is 24, and it is also a multiple of 8. Therefore, the smallest positive multiple of 6 that is also a multiple of 8 is 24.  |

### 8.3.2 Knowledge fusion

In order to achieve automatic knowledge fusion, this paper utilizes VDs and LLM to update entities, ensuring that entities and relationships from different sources are added to the Existing KG without redundancy or omission. Table 15 presents an example of Theorem entity fusion from different sources. The Theorem entity “*Mean Value Theorem*” from a textbook is a necessary condition for the Theorem entity “*Carathéodory’s Theorem (Analysis)*” from ProofWiki. During the update of AutoMathKG, the two entities were successfully merged, and the reference relationships associated with the “*Mean Value Theorem*” entity were added to the “*Carathéodory’s Theorem (Analysis)*” entity.

**Table 15** The results of the case study on automatic knowledge fusion.

|  Source | ProofWiki  |
| --- | --- |
|  Theorem | Carathéodory’s Theorem (Analysis)  |
|  Contents | Let $I \subseteq \mathbb{R}$. Let $c \in I$ be an interior point of $I$. Let $f : I \rightarrow \mathbb{R}$ be a real function. Then $f$ is differentiable at $c$ if and only if: There exists a real function $\varphi : I \rightarrow \mathbb{R}$ that is continuous at $c$ and satisfies: (1): $\forall x \in I, f(x) - f(c) = \varphi(x)(x - c)$ (2): $\varphi(c) = f'(c)$.  |
|  Source | Textbook  |
|  Theorem | Mean Value Theorem  |
|  Contents | If $f$ is continuous on the closed interval $[a, b]$ and differentiable on the open interval $(a, b)$, then, $f'(c) = \frac{f(b) - f(a)}{b - a}$, for some $c$ in $(a, b)$.  |

## 8.4 Limitation

One limitation of our work is the choice of LLM during ICL. To save computational resources, we only used a small-scale model, Llama-2-7b, while larger language models may perform better. Another limitation is the trade-off between corpus richness and mathematical knowledge specialization. To obtain accurate mathematical knowledge, the corpus collected in this paper focuses on formal texts and does not include informal corpora, such as the discussion dialogues in mathematical forums, which may provide different types of mathematical knowledge.

## 9 Conclusion

This paper proposes AutoMathKG, a novel high-quality, wide-coverage, and multi-dimensional knowledge graph in the field of mathematics that can be automatically updated. AutoMathKG integrates math knowledge from various domains and levels in natural language, establishing connections among the three major dimensions of mathematics: definitions, theorems, and problems, all augmented by LLMs. In order to retrieve similar entities, MathVD is created as a VD of AutoMathKG, with two strategies for entity embeddings using SBERT. In order to automatically update, two mechanisms are proposed. One is for knowledge completion through interacting with Math LLM, a specialized LLM designed to supplement missing proofs and solutions. Another is for knowledge fusion across different sources, through fuzzy search in MathVD and the determination of update approaches by LLM. Comprehensive experimental results demonstrate the advanced performance of AutoMathKG, MathVD, and Math LLM, as well as the effectiveness and applicability of the automatic update method. AutoMathKG system opens up new possibilities for applying AI techniques in the acquisition of mathematical knowledge and the enhancement of mathematical reasoning, making a significant contribution to the field of AI for mathematics.

In future work, we plan to use more advanced models beyond Llama-2-7b for data augmentation. Moreover, we intend to explore a broader range of mathematical corpora, such as math forums, to increase the diversity of mathematical knowledge.

Acknowledgements. This research was supported by Key Lab of Random Complex Structures and Data Science, Chinese Academy of Sciences (2008DP173182); National Key R&D Program of China (2021ZD0111204).

## Declarations

- **Author contribution** All authors contributed to this research. Data collection, model implementation, and experimental evaluation were conducted by R.B., Y.G., and Z.Y. B.C. formulated the research idea, coordinated the efforts, and supervised the overall process. R.B. drafted the manuscript, and all authors reviewed and commented on its previous versions.

- **Funding** This research was supported by Key Lab of Random Complex Structures and Data Science, Chinese Academy of Sciences (2008DP173182); National Key R&D Program of China (2021ZD0111204).

- **Conflict of interest** The authors declare that they have no conflict of interest.

- Ethics approval and consent to participate Not applicable.

- Consent for publication Not applicable.

- Data availability The data collected to construct the knowledge graph can be accessed through the provided links or citations in this article. The constructed knowledge graph will be made available on request.

## Appendix A More information for corpus

Table A1 presents the title information of the eight textbooks analyzed in this study. The first six textbooks were found through an online search for open-source textbooks, while the last two were provided by the NaturalProofs dataset. Table A2 shows the subject information of the arXiv papers collected in this study.

Table A1 The title information of selected textbooks.

|  Textbook titles  |
| --- |
|  Linear Transformations on Vector Spaces  |
|  Elementary Differential  |
|  Basic Analysis: Introduction to Real Analysis  |
|  Ossifrage and Algebra  |
|  Multivariable and Vector Calculus  |
|  An Introduction to Group Theory  |
|  Introduction to Real Analysis  |
|  Elementary Number Theory: Primes, Congruences, and Secrets  |

Table A2 The subject information of selected papers from arXiv.

|  Paper subjects  |
| --- |
|  math.AG, math.AT, math.AP, math.CT, math.CA, math.CO, math.AC, math.CV, math.DG, math.PR, math.ST  |

## Appendix B Instructions of entity storage format

Table B3 details the entity storage format, including the storage type and structure of all attributes as well as the applicable entity types.

## Appendix C In-context learning templates

Table C4 presents the prompt templates used by LLM during ICL for enhancing entity attributes, such as “title”, “field”, and “bodylist”, for different types of entities. Table C5 displays the prompt templates used in the automatic knowledge fusion process through ICL, including templates for Steps 4 and 5.

Table B3 The attributes stored in entities of AutoMathKG.

|  Attribute | Storage type | Storage structure | Entity  |
| --- | --- | --- | --- |
|  Id | Int | Number | Thm, Def, Prob  |
|  Type | String | Theorem/definition/problem | Thm, Def, Prob  |
|  Label | String | Entity label | Thm, Def, Prob  |
|  Title | String | Entity title | Thm, Def, Prob  |
|  Field | String | Algebra/geometry/analysis/probability and statistics/applied mathematics/... | Thm, Def, Prob  |
|  Contents | List | [content1, content2, ...] | Thm, Def, Prob  |
|  Bodylist | List nested dictionary | [{"description": content1,"action": action1},...] | Thm, Def  |
|  Refs | List | [reference1, reference2, ...] | Thm, Def, Prob  |
|  References_tactics | Dictionary | {"reference1": tactic1,"reference2": tactic2,...} | Thm, Def, Prob  |
|  Source | String | ProofWiki/textbook/arXiv/TheoremQA | Thm, Def, Prob  |
|  Proofs | List nested dictionary | [{"contents": [...],"refs": [...],"bodylist": [...],"references_tactics": {...}},...] | Thm  |
|  Solutions | List nested dictionary | [{"contents": [...],"refs": [...],"bodylist": [],"references_tactics": {...}},...] | Prob  |
|  In_refs | Dictionary | {"in_reference1": tactic1,"in_reference2": tactic2,...} | Thm, Def, Prob  |
|  In_ref_ids | List | [in_id1, in_id2,...] | Thm, Def, Prob  |
|  Out_refs | Dictionary | {"out_reference1": tactic1,"out_reference2": tactic2,...} | Thm, Def, Prob  |
|  Out_ref_ids | List | [out_id1, out_id2,...] | Thm, Def, Prob  |

## Appendix D Math LLM training details

### D.1 Training datasets

The gemma-7b-it model was chosen as the base model for the construction of Math-LLM. Firstly, the model was fine-tuned using LORA on the MathInstruct  \( [15] \)  dataset, which contains various types of mathematical problem sets, enabling the model to acquire foundational problem-solving capabilities. Furthermore, the application adapter was trained on the “Microsoft/orca-math-word-problems-200k”  \( [57] \)  dataset, which includes a diverse range of application problems. The sympy adapter was trained on the GSM8K-sympy-v2  \( [58] \)  dataset, enhancing its ability to solve problems using the SymPy package. The proof adapter was trained on the NaturalProofs  \( [12] \)  dataset, which provides a comprehensive collection of definitions, theories, and their corresponding proofs.

Table C4 The templates of entity augmentation.

|  Attribute | Entity | Template  |
| --- | --- | --- |
|  Title | Def | Your task is to generate a title that can summarize the content of the given math definition/theorem/problem.  |
|   |  Thm  |   |
|   |  Prob  |   |
|  Field | Def | Your task is to identify the most relevant mathematical field for the given math definition/theorem/problem from the following choices: “algebra”, “geometry”, “analysis”, “logic”, “probability and statistics”, “applied mathematics”, “foundations of mathematics”. Choose one from them.  |
|   |  Thm  |   |
|   |  Prob  |   |
|  Bodylist | Def | Your task is to label each element in the given content list of a math definition/theorem/theorem proof, in order to determine the role of each element. The labels of mathematical roles are: “premise”, “assumption”, “lemma”, “corollary”, “definition”, “conclusion”, “deduction”, “calculation”, “enumeration”. Choose only the most relative one from them when you label.  |
|   |  Thm (including proof)  |   |
|  Refs | Prob (including solution) | Your task is to identify the mathematical definitions or theorems referenced in the given math problem/problem solution. When there is at least one reference, output each reference in the following format: “definition:” or “theorem:” along with the original reference. When there is no reference, output an empty list.  |
|  References_tactics | Def | Your task is to label each reference shown in the given content list of a math definition/theorem/theorem proof/problem/problem solution, in order to determine the roles of each reference. The labels of mathematical roles are: “premise”, “assumption”, “lemma”, “corollary”, “definition”, “conclusion”, “deduction”, “calculation”, “enumeration”. Choose only the most relative one from them when you label.  |
|   |  Thm (including proof)  |   |
|   |  Prob (including solution)  |   |

## D.2 Training hyperparameters

For all training processes, including base model fine-tuning and task adapter training, we employed fully shared data parallelism (FSDP) with the configuration parameters: node=1 and proc_per_node=4 in the PyTorch framework. The optimizer used was paged_adaw_32bit. The gemma-7b-it model was loaded in torch.bf16 format. The maximum sequence length was set to 1024. The LoRA alpha was configured to 16, LoRA rank to 8, and LoRA dropout to 0.1. The learning rate was set to 2e-4, with a weight decay of 0.001. The training batch size was set to 4, with fine-tuning performed for 50 epochs and adapters trained for 25 epochs.

Table C5 The templates of automatic knowledge fusion in Steps 4 and 5.

|  Automatic knowledge fusion | Template  |
| --- | --- |
|  Step 4 | Your task is to decide if the first math theorem mean the same thing as the second theorem. If they have the same meaning, you should answer “yes”. Otherwise, you answer “no”.  |
|  Step 5 | Your task is to decide which of the candidate theorems mean the same thing as the new theorem. You should choose one candidate and output its id number as your answer.  |

## Appendix E More examples for study

Figure E1 presents an instance of a Definition entity stored in JSON format. Table E6 provides another example of similar entity retrieval for the ablation study.

![img-12.jpeg](img-12.jpeg)

Fig. E1 An instance of a Definition entity in AutoMathKG.

Table E6 The result of another example in ablation study.

|  Theorem: union is associative\( A \cup (B \cup C) = (A \cup B) \cup C \)  |   |   |
| --- | --- | --- |
|  Retrieval results in VD1-no refs  |   |   |
|  Rank | Score | Entity  |
|  1 | 0.7341 | Theorem: intersection is associative  |
|  2 | 0.7317 | Theorem: union is commutative  |
|  3 | 0.7298 | Theorem: union is idempotent  |
|  4 | 0.7090 | Theorem: union distributes over intersection  |
|  5 | 0.7028 | Theorem: cartesian product distributes over union  |
|  Retrieval results in VD1-all  |   |   |
|  Rank | Score | Entity  |
|  1 | 0.8977 | Theorem: union is commutative  |
|  2 | 0.8300 | Theorem: symmetric difference is associative  |
|  3 | 0.8217 | Theorem: set intersection is associative  |
|  4 | 0.7996 | Theorem: union as symmetric difference with intersection  |
|  5 | 0.7808 | Theorem: union with complement  |

## References

[1] Fensel, D., Simsek, U., Angele, K., Huaman, E., Kärle, E., Panasiuk, O., Toma, I., Umbrich, J., Wahler, A.: Knowledge graphs methodology, tools and selected use cases (2020)
[2] Zou, X.: A survey on application of knowledge graph. In: Journal of Physics: Conference Series, vol. 1487, p. 012016 (2020). IOP Publishing
[3] Zwaneveld, B.: Structuring mathematical knowledge and skills by means of knowledge graphs. International Journal of Mathematical Education in Science and Technology 31(3), 393–414 (2000)
[4] Zhang, C.E., Collins, K.M., Weller, A., Tenenbaum, J.B.: Ai for mathematics: A cognitive science perspective. arXiv preprint arXiv:2310.13021 (2023)
[5] Szegedy, C.: A promising path towards autoformalization and general artificial intelligence. In: Intelligent Computer Mathematics: 13th International Conference, CICM 2020, Bertinoro, Italy, July 26–31, 2020, Proceedings 13, pp. 3–20 (2020). Springer
[6] Schoenfeld, A.H.: On mathematics as sense-making: An informal attack on the unfortunate divorce of formal and informal mathematics. In: Informal Reasoning and Education (1991)
[7] Megill, N.D., Wheeler, D.A.: Metamath: A Computer Language for Pure Mathematics. Lulu Press, Morrisville, North Carolina (2019)
[8] Moura, L.d., Ullrich, S.: The lean 4 theorem prover and programming language. In: Automated Deduction–CADE 28: 28th International Conference on Automated Deduction, Virtual Event, July 12–15, 2021, Proceedings 28, pp. 625–635 (2021). Springer
[9] Wu, Y., Rabe, M.N., Li, W., Ba, J., Grosse, R.B., Szegedy, C.: Lime: Learning inductive bias for primitives of mathematical reasoning. In: International Conference on Machine Learning, pp. 11251–11262 (2021). PMLR
[10] Gowers, T., Barrow-Green, J., Leader, I.: The Princeton Companion to Mathematics. Princeton University Press, Princeton, New Jersey (2010)
[11] Zhao, T., Huang, Y., Yang, S., Luo, Y., Feng, J., Wang, Y., Yuan, H., Pan, K., Li, K., Li, H., et al.: Mathgraph: A knowledge graph for automatically solving mathematical exercises. In: Database Systems for Advanced Applications: 24th International Conference, DASFAA 2019, Chiang Mai, Thailand, April 22–25, 2019, Proceedings, Part I 24, pp. 760–776 (2019). Springer
[12] Welleck, S., Liu, J., Bras, R.L., Hajishirzi, H., Choi, Y., Cho, K.: Naturalproofs: Mathematical theorem proving in natural language. arXiv preprint

arXiv:2104.01112 (2021)

[13] Wang, J.: Math-kg: Construction and applications of mathematical knowledge graph. arXiv preprint arXiv:2205.03772 (2022)

[14] Horowitz, L., Paiva, V.: Mathgloss: Building mathematical glossaries from text. arXiv preprint arXiv:2311.12649 (2023)

[15] Yue, X., Qu, X., Zhang, G., Fu, Y., Huang, W., Sun, H., Su, Y., Chen, W.: Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653 (2023)

[16] Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in neural information processing systems 33, 1877–1901 (2020)

[17] Pezoa, F., Reutter, J.L., Suarez, F., Ugarte, M., Vrgoč, D.: Foundations of json schema. Proceedings of the 25th International Conference on World Wide Web (2016)

[18] Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023)

[19] Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084 (2019)

[20] Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., Steinhardt, J.: Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874 (2021)

[21] Azerbayev, Z., Piotrowski, B., Schoelkopf, H., Ayers, E.W., Radev, D., Avigad, J.: Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. arXiv preprint arXiv:2302.12433 (2023)

[22] Paster, K., Santos, M.D., Azerbayev, Z., Ba, J.: Openwebmath: An open dataset of high-quality mathematical web text. arXiv preprint arXiv:2310.06786 (2023)

[23] Wang, Z., Xia, R., Liu, P.: Generative ai for math: Part i–mathpile: A billion-token-scale pretraining corpus for math. arXiv preprint arXiv:2312.17120 (2023)

[24] Mishra, S., Mitra, A., Varshney, N., Sachdeva, B., Clark, P., Baral, C., Kalyan, A.: Numglue: A suite of fundamental yet challenging mathematical reasoning tasks. arXiv preprint arXiv:2204.05660 (2022)

[25] Yuan, Z., Yuan, H., Li, C., Dong, G., Tan, C., Zhou, C.: Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint

[26] Li, G., Hammoud, H.A.A.K., Itani, H., Khizbullin, D., Ghanem, B.: Camel: Communicative agents for "mind" exploration of large language model society. ArXiv abs/2303.17760 (2023)
[27] Chen, W., Yin, M., Ku, M., Lu, P., Wan, Y., Ma, X., Xu, J., Wang, X., Xia, T.: Theoremqa: A theorem-driven question answering dataset. (2023). The 2023 Conference on Empirical Methods in Natural Language Processing
[28] Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 24824–24837 (2022)
[29] Chen, W., Ma, X., Wang, X., Cohen, W.W.: Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588 (2022)
[30] Frieder, S., Pinchetti, L., Griffiths, R.-R., Salvatori, T., Lukasiewicz, T., Petersen, P., Berner, J.: Mathematical capabilities of chatgpt. Advances in Neural Information Processing Systems 36 (2024)
[31] Ji, S., Pan, S., Cambria, E., Marttinen, P., Philip, S.Y.: A survey on knowledge graphs: Representation, acquisition, and applications. IEEE transactions on neural networks and learning systems 33(2), 494–514 (2021)
[32] Auer, S., Bizer, C., Kobilarov, G., Lehmann, J., Cyganiak, R., Ives, Z.: Dbpedia: A nucleus for a web of open data. In: International Semantic Web Conference, pp. 722–735 (2007). Springer
[33] Suchanek, F.M., Kasneci, G., Weikum, G.: Yago: a core of semantic knowledge, pp. 697–706 (2007). Proceedings of the 16th international conference on World Wide Web
[34] Hoffart, J., Suchanek, F.M., Berberich, K., Weikum, G.: Yago2: A spatially and temporally enhanced knowledge base from wikipedia. Artificial intelligence 194, 28–61 (2013)
[35] Singhal, A.: Introducing the Knowledge Graph: things, not strings. Google Blog. 2020-11-13 (2012). https://www.blog.google/products/search/introducing-knowledge-graph-things-not/
[36] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems 33, 9459–9474 (2020)

[37] Zheng, W., Zou, L., Peng, W., Yan, X., Song, S., Zhao, D.: Semantic sparql similarity search over rdf knowledge graphs. *Proceedings of the VLDB Endowment* **9**(11), 840–851 (2016)[38] Mikolov, T., Chen, K., Corrado, G., Dean, J.: Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781* (2013)[39] Pennington, J., Socher, R., Manning, C.D.: Glove: Global vectors for word representation, pp. 1532–1543 (2014). *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*[40] Devlin, J., Chang, M.-W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805* (2018)[41] Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., Yakhnenko, O.: Translating embeddings for modeling multi-relational data. *Advances in neural information processing systems* **26** (2013)[42] He, S., Liu, K., Ji, G., Zhao, J.: Learning to represent knowledge graphs with gaussian embedding, pp. 623–632 (2015). *Proceedings of the 24th ACM international conference on information and knowledge management*[43] Nickel, M., Rosasco, L., Poggio, T.: Holographic embeddings of knowledge graphs. In: *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 30 (2016)[44] Schlichtkrull, M., Kipf, T.N., Bloem, P., Van Den Berg, R., Titov, I., Welling, M.: Modeling relational data with graph convolutional networks. In: *The Semantic Web: 15th International Conference, ESWC 2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings 15*, pp. 593–607 (2018). Springer[45] Abboud, R., Ceylan, I., Lukasiewicz, T., Salvatori, T.: Boxe: A box embedding model for knowledge base completion. *Advances in Neural Information Processing Systems* **33**, 9649–9661 (2020)[46] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. *arXiv preprint arXiv:2303.08774* (2023)[47] Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research* **24**(240), 1–113 (2023)[48] Taylor, R., Kardas, M., Cucurull, G., Scialom, T., Hartshorn, A., Saravia, E., Poulton, A., Kerkez, V., Stojnic, R.: Galactica: A large language model for science. *arXiv preprint arXiv:2211.09085* (2022)

[49] Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Rivière, M., Kale, M.S., Love, J., et al.: Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295 (2024)[50] Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al.: In-context learning and induction heads. arXiv preprint arXiv:2209.11895 (2022)[51] Saunshi, N., Malladi, S., Arora, S.: A mathematical exploration of why language models help solve downstream tasks. arXiv preprint arXiv:2010.03648 (2020)[52] Liu, J., Shen, D., Zhang, Y., Dolan, B., Carin, L., Chen, W.: What makes good in-context examples for gpt-3? arXiv preprint arXiv:2101.06804 (2021)[53] Min, S., Lyu, X., Holtzman, A., Artetxe, M., Lewis, M., Hajishirzi, H., Zettlemoyer, L.: Rethinking the role of demonstrations: What makes in-context learning work? arXiv preprint arXiv:2202.12837 (2022)[54] Lampinen, A.K., Dasgupta, I., Chan, S.C., Matthewson, K., Tessler, M.H., Creswell, A., McClelland, J.L., Wang, J.X., Hill, F.: Can language models learn from explanations in context? arXiv preprint arXiv:2204.02329 (2022)[55] Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021)[56] Zhao, Y., Yan, L., Sun, W., Xing, G., Meng, C., Wang, S., Cheng, Z., Ren, Z., Yin, D.: Knowing what llms do not know: A simple yet effective self-detection method. arXiv preprint arXiv:2310.17918 (2023)[57] Mitra, A., Khanpour, H., Rosset, C., Awadallah, A.: Orca-math: Unlocking the potential of slms in grade school math. arXiv preprint arXiv:2402.14830 (2024)[58] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al.: Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021)