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