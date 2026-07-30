if every pair of vertices is connected by an edge. Since a graph is determined completely by which vertices are adjacent to which other vertices, there is only one complete graph with a given number of vertices. We give these a special name: $K_n$ is the complete graph on $n$ vertices.

Each vertex in $K_n$ is adjacent to $n - 1$ other vertices. We call the number of edges emanating from a given vertex the **degree** of that vertex. So every vertex in $K_n$ has degree $n - 1$. How many edges does $K_n$ have? One might think the answer should be $n(n - 1)$, since we count $n - 1$ edges $n$ times (once for each vertex). However, each edge is incident to 2 vertices, so we counted every edge exactly twice. Thus there are $n(n - 1)/2$ edges in $K_n$. Alternatively, we can say there are $\binom{n}{2}$ edges, since to draw an edge we must choose 2 of the $n$ vertices.

In general, if we know the degrees of all the vertices in a graph, we can find the number of edges. The sum of the degrees of all vertices will always be *twice* the number of edges, since each edge adds to the degree of two vertices. Notice this means that the sum of the degrees of all vertices in any graph must be even!

This is our first example of a general result about all graphs. It seems innocent enough, but we will use it to prove all sorts of other statements. So let's give it a name and state it formally.

#### Lemma 2.1.8 Handshake Lemma.

*In any graph, the sum of the degrees of vertices in the graph is always twice the number of edges.*

The handshake lemma$^2$ is sometimes called the *degree sum formula*, and can be written symbolically as

$$\sum_{v \in V} d(v) = 2e.$$

Here we are using the notation $d(v)$ for the degree of the vertex $v$.

One use for the lemma is to actually find the number of edges in a graph. To do this, you must be given the **degree sequence** for the graph (or be able to find it from other information). This is a list of every degree of every vertex in the graph, generally written in non-increasing order.

#### Example 2.1.9

How many vertices and edges must a graph have if its degree sequence is

$$(4, 4, 3, 3, 3, 2, 1)?$$

**Solution.** The number of vertices is easy to find. It is the number of degrees in the sequence: 7. To find the number of edges, we compute the degree sum

$$4 + 4 + 3 + 3 + 3 + 2 + 1 = 20,$$

$^2$A *lemma* is a mathematical statement that is primarily of importance in that it is used to establish other results.