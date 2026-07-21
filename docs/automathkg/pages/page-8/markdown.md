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