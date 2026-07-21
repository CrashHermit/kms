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