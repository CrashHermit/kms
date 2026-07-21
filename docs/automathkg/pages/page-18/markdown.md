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