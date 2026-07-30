so the number of edges is half this: 10.

The handshake lemma also tells us what is not possible.

### Example 2.1.10

At a recent math seminar, 9 mathematicians greeted each other by shaking hands. Is it possible that each mathematician shook hands with exactly 8 people at the seminar?

**Solution.** It seems like this should be possible. Each mathematician chooses one person to not shake hands with. But this cannot happen. We are asking whether a graph with 9 vertices can have each vertex have degree 7. If such a graph existed, the sum of the degrees of the vertices would be $9 \cdot 7 = 63$. This would be twice the number of edges (handshakes) resulting in a graph with 31.5 edges. That is impossible. Thus at least one (in fact an odd number) of the mathematicians must have shaken hands with an *even* number of people at the seminar.

We can generalize the previous example to get the following proposition.$^{3}$

### Proposition 2.1.11

*In any graph, the number of vertices with odd degree must be odd.*

**Proof.** Suppose there were a graph with an odd number of vertices with odd degree. Then the sum of the degrees in the graph would be odd, which is impossible, by the handshake lemma.

We will consider further applications of the handshake lemma in the exercises.

One final definition: We say a graph is **bipartite** if the vertices can be divided into two sets, $A$ and $B$, with no two vertices in $A$ adjacent and no two vertices in $B$ adjacent. The vertices in $A$ can be adjacent to some or all of the vertices in $B$. If each vertex in $A$ is adjacent to all the vertices in $B$, then the graph is a **complete bipartite graph**, and gets a special name: $K_{m,n}$, where $|A| = m$ and $|B| = n$.

**Named Graphs.** Some graphs are used more than others and get special names.

|  $K_n$ | The complete graph on $n$ vertices.  |
| --- | --- |
|  $K_{m,n}$ | The complete bipartite graph with sets of $m$ and $n$ vertices.  |
|  $C_n$ | The cycle on $n$ vertices, just one big loop.  |
|  $P_n$ | The path on $n + 1$ vertices (so $n$ edges), just one long path.  |

$^{3}$A **proposition** is a general statement in mathematics, similar to a theorem, although generally of lesser importance.