The moral of this example is that people can lie, but true mathematical statements *never* lie.

We close this section with a word about the use of parentheses. The symbol $\sim$ is analogous to the minus sign in algebra. It negates the expression it precedes. Thus $\sim P \vee Q$ means $(\sim P) \vee Q$, not $\sim (P \vee Q)$. In $\sim (P \vee Q)$, the value of the entire expression $P \vee Q$ is negated.

### Exercises for Section 2.5

Write a truth table for the logical statements in problems 1–9:

1. $P \vee (Q \Rightarrow R)$
2. $(Q \vee R) \Leftrightarrow (R \wedge Q)$
3. $\sim (P \Rightarrow Q)$
4. $\sim (P \vee Q) \vee (\sim P)$
5. $(P \wedge \sim P) \vee Q$
6. $(P \wedge \sim P) \wedge Q$
7. $(P \wedge \sim P) \Rightarrow Q$
8. $P \vee (Q \wedge \sim R)$
9. $\sim (\sim P \vee \sim Q)$
10. Suppose the statement $((P \wedge Q) \vee R) \Rightarrow (R \vee S)$ is false. Find the truth values of $P, Q, R$ and $S$. (This can be done without a truth table.)
11. Suppose $P$ is false and that the statement $(R \Rightarrow S) \Leftrightarrow (P \wedge Q)$ is true. Find the truth values of $R$ and $S$. (This can be done without a truth table.)

### 2.6 Logical Equivalence

In contemplating the truth table for $P \Leftrightarrow Q$, you probably noticed that $P \Leftrightarrow Q$ is true exactly when $P$ and $Q$ are both true or both false. In other words, $P \Leftrightarrow Q$ is true precisely when at least one of the statements $P \wedge Q$ or $\sim P \wedge \sim Q$ is true. This may tempt us to say that $P \Leftrightarrow Q$ means the same thing as $(P \wedge Q) \vee (\sim P \wedge \sim Q)$.

To see if this is really so, we can write truth tables for $P \Leftrightarrow Q$ and $(P \wedge Q) \vee (\sim P \wedge \sim Q)$. In doing this, it is more efficient to put these two statements into the same table, as follows. (This table has helper columns for the intermediate expressions $\sim P, \sim Q, (P \wedge Q)$ and $(\sim P \wedge \sim Q)$.)

|  $P$ | $Q$ | $\sim P$ | $\sim Q$ | $(P \wedge Q)$ | $(\sim P \wedge \sim Q)$ | $(P \wedge Q) \vee (\sim P \wedge \sim Q)$ | $P \Leftrightarrow Q$  |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  $T$ | $T$ | $F$ | $F$ | $T$ | $F$ | $\mathbf{T}$ | $\mathbf{T}$  |
|  $T$ | $F$ | $F$ | $T$ | $F$ | $F$ | $\mathbf{F}$ | $\mathbf{F}$  |
|  $F$ | $T$ | $T$ | $F$ | $F$ | $F$ | $\mathbf{F}$ | $\mathbf{F}$  |
|  $F$ | $F$ | $T$ | $T$ | $F$ | $T$ | $\mathbf{T}$ | $\mathbf{T}$  |

The table shows that $P \Leftrightarrow Q$ and $(P \wedge Q) \vee (\sim P \wedge \sim Q)$ have the same truth value, no matter the values $P$ and $Q$. It is as if $P \Leftrightarrow Q$ and $(P \wedge Q) \vee (\sim P \wedge \sim Q)$ are algebraic expressions that are equal no matter what is “plugged into”

Richard Hammack *Book of Proof*