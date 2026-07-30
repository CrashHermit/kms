$$\{c, d\}, \{c, b\}, \{c, a\}, \{b, a\},$$

which are precisely the edges in $G_2$. Thus $g$ is an isomorphism, so $G_1 \cong G_2$

Sometimes we will talk about a graph with a special name (like $K_n$ or the *Petersen graph*) or perhaps draw a graph without any labels. In this case, we are really referring to *all* graphs isomorphic to any copy of that particular graph. A collection of isomorphic graphs is often called an **isomorphism class**.$^1$

There are other relationships between graphs that we care about, other than equality and being isomorphic. For example, compare the following pair of graphs:

![1]()

![2]()

These are definitely not isomorphic, but notice that the graph on the right looks like it might be part of the graph on the left, especially if we draw it like this:

![3]()

We would like to say that the smaller graph is a *subgraph* of the larger.

We should give a careful definition of this. In fact, there are two reasonable notions for what a subgraph should mean.

#### Definition 2.1.6 Subgraphs.

We say that $G' = (V', E')$ is a **subgraph** of $G = (V, E)$, and write $G' \subseteq G$, provided $V' \subseteq V$ and $E' \subseteq E$.

We say that $G' = (V', E')$ is an **induced subgraph** of $G = (V, E)$ provided $V' \subseteq V$ and every edge in $E$ whose vertices are still in $V'$ is also an edge in $E'$.

Notice that every induced subgraph is also an ordinary subgraph, but not conversely. Think of a subgraph as the result of deleting some vertices and edges from the larger graph. For the subgraph to be an induced subgraph, we can still delete vertices, but now we only delete those edges that included the deleted vertices.