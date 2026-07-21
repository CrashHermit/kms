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