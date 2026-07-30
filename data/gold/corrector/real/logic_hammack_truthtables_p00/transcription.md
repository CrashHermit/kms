The truth table for $\Leftrightarrow$ is shown below. Notice that in the first and last rows, both $P \Rightarrow Q$ and $Q \Rightarrow P$ are true (according to the truth table for $\Leftrightarrow$), so $(P \Rightarrow Q) \wedge (Q \Rightarrow P)$ is true, and hence $P \Leftrightarrow Q$ is true. However, in the middle two rows one of $P \Rightarrow Q$ or $Q \Rightarrow P$ is false, so $(P \Rightarrow Q) \wedge (Q \Rightarrow P)$ is false, making $P \Leftrightarrow Q$ false.

|  $P$ | $Q$ | $P \Leftrightarrow Q$  |
| --- | --- | --- |
|  $T$ | $T$ | $T$  |
|  $T$ | $F$ | $F$  |
|  $F$ | $T$ | $F$  |
|  $F$ | $F$ | $T$  |

Compare the statement $R : (a \text{ is even}) \Leftrightarrow (a \text{ is divisible by 2})$ with this truth table. If $a$ is even then the two statements on either side of $\Leftrightarrow$ are true, so according to the table $R$ is true. If $a$ is odd then the two statements on either side of $\Leftrightarrow$ are false, and again according to the table $R$ is true. Thus $R$ is true no matter what value $a$ has. In general, $P \Leftrightarrow Q$ being true means $P$ and $Q$ are both true or both false.

Not surprisingly, there are many ways of saying $P \Leftrightarrow Q$ in English. The following constructions all mean $P \Leftrightarrow Q$:

$$\left. \begin{array}{l} P \text{ if and only if } Q. \\ P \text{ is a necessary and sufficient condition for } Q. \\ \text{For } P \text{ it is necessary and sufficient that } Q. \\ P \text{ is equivalent to } Q. \\ \text{If } P, \text{ then } Q, \text{ and conversely.} \end{array} \right\} P \Leftrightarrow Q$$

The first three of these just combine constructions from the previous section to express that $P \Rightarrow Q$ and $Q \Rightarrow P$. In the last one, the words “...and conversely” mean that in addition to “If $P$, then $Q$” being true, the converse statement “If $Q$, then $P$” is also true.

### Exercises for Section 2.4

Without changing their meanings, convert each of the following sentences into a sentence having the form “$P$ if and only if $Q$.”

1. For matrix $A$ to be invertible, it is necessary and sufficient that $\det(A) \neq 0$.
2. If a function has a constant derivative then it is linear, and conversely.
3. If $xy = 0$ then $x = 0$ or $y = 0$, and conversely.
4. If $a \in \mathbb{Q}$ then $5a \in \mathbb{Q}$, and if $5a \in \mathbb{Q}$ then $a \in \mathbb{Q}$.
5. For an occurrence to become an adventure, it is necessary and sufficient for one to recount it. (Jean-Paul Sartre)

Free PDF version
![Creative Commons License Icon]() CC BY-NC-SA