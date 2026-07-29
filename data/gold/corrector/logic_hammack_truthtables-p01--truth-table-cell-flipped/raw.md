## 2.5 Truth Tables for Statements

You should now know the truth tables for $\wedge, \vee, \sim, \Rightarrow$ and $\Leftrightarrow$. They should be *internalized* as well as memorized. You must understand the symbols thoroughly, for we now combine them to form more complex statements.

For example, suppose we want to convey that one or the other of $P$ and $Q$ is true but they are not both true. No single symbol expresses this, but we could combine them as

$$(P \vee Q) \wedge \sim (P \wedge Q),$$

which literally means:

*P or Q is true, and it is not the case that both P and Q are true.*

This statement will be true or false depending on the truth values of $P$ and $Q$. In fact we can make a truth table for the entire statement. Begin as usual by listing the possible true/false combinations of $P$ and $Q$ on four lines. The statement $(P \vee Q) \wedge \sim (P \wedge Q)$ contains the individual statements $(P \vee Q)$ and $(P \wedge Q)$, so we next tally their truth values in the third and fourth columns. The fifth column lists values for $\sim (P \wedge Q)$, and these are just the opposites of the corresponding entries in the fourth column. Finally, combining the third and fifth columns with $\wedge$, we get the values for $(P \vee Q) \wedge \sim (P \wedge Q)$ in the sixth column.

|  $P$ | $Q$ | $(P \vee Q)$ | $(P \wedge Q)$ | $\sim (P \wedge Q)$ | $(P \vee Q) \wedge \sim (P \wedge Q)$  |
| --- | --- | --- | --- | --- | --- |
|  $T$ | $T$ | $T$ | $T$ | $F$ | $\mathbf{F}$  |
|  $T$ | $F$ | $T$ | $F$ | $T$ | $\mathbf{T}$  |
|  $F$ | $T$ | $T$ | $F$ | $T$ | $\mathbf{T}$  |
|  $F$ | $F$ | $F$ | $F$ | $F$ | $\mathbf{F}$  |

This truth table tells us that $(P \vee Q) \wedge \sim (P \wedge Q)$ is true precisely when one but not both of $P$ and $Q$ are true, so it has the meaning we intended. (Notice that the middle three columns of our truth table are just "helper columns" and are not necessary parts of the table. In writing truth tables, you may choose to omit such columns if you are confident about your work.)

For another example, consider the following familiar statement about real numbers $x$ and $y$:

The product $xy$ equals zero if and only if $x = 0$ or $y = 0$.

This can be modeled as $(xy = 0) \Leftrightarrow (x = 0 \vee y = 0)$. If we introduce letters $P, Q$ and $R$ for the statements $xy = 0$, $x = 0$ and $y = 0$, it becomes $P \Leftrightarrow (Q \vee R)$. Notice that the parentheses are necessary here, for without them we wouldn't know whether to read the statement as $P \Leftrightarrow (Q \vee R)$ or $(P \Leftrightarrow Q) \vee R$.