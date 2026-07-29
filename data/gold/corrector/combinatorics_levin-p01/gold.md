### Example 2.1.7

Consider the graphs:

![1]()

![2]()

![3]()

![4]()

Here both $G_2$ and $G_3$ are subgraphs of $G_1$. But only $G_2$ is an *induced* subgraph. Every edge in $G_1$ that connects vertices in $G_2$ is also an edge in $G_2$. In $G_3$, the edge $\{a, b\}$ is in $E_1$ but not $E_3$, even though vertices $a$ and $b$ are in $V_3$.

The graph $G_4$ is NOT a subgraph of $G_1$, even though it looks like all we did is remove vertex $e$. The reason is that in $E_4$ we have the edge $\{c, f\}$, but this is not an element of $E_1$, so we don't have the required $E_4 \subseteq E_1$.

Back to some basic graph theory definitions. Notice that all the graphs we have drawn above have the property that no pair of vertices is connected more than once, and no vertex is connected to itself. Graphs like these are sometimes called **simple**, although we will just call them *graphs*. This is because our definition of a graph says that the edges form a set of 2-element subsets of the vertices. Remember that it doesn't make sense to say a set contains an element more than once. So no pair of vertices can be connected by an edge more than once. Also, since each edge must be a set containing two vertices, we cannot have a single vertex connected to itself by an edge.

That said, there are times we want to consider double (or more) edges and single-edge loops. For example, the "graph" we drew for the Bridges of Königsberg problem had double edges because there really are two bridges connecting a particular island to the near shore. We will call these objects **multigraphs**. This is a good name: A *multiset* is a set in which we are allowed to include a single element multiple times.

The graphs above are also **connected**: you can get from any vertex to any other vertex by following some path of edges. A graph that is not connected can be thought of as two separate graphs drawn close together. For example, the following graph is NOT connected because there is no path from $a$ to $b$:

![5]()

Vertices in a graph do not always have edges between them. If we add all possible edges, then the resulting graph is called **complete**. That is, a graph is complete