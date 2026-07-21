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