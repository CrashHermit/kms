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