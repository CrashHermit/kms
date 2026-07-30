Making a truth table for $P \Leftrightarrow (Q \vee R)$ entails a line for each $T/F$ combination for the three statements $P$, $Q$ and $R$. The eight possible combinations are tallied in the first three columns of the following table.

|  $P$ | $Q$ | $R$ | $Q \vee R$ | $P \Leftrightarrow (Q \vee R)$  |
| --- | --- | --- | --- | --- |
|  $T$ | $T$ | $T$ | $T$ | $\mathbf{T}$  |
|  $T$ | $T$ | $F$ | $T$ | $\mathbf{T}$  |
|  $T$ | $F$ | $T$ | $T$ | $\mathbf{T}$  |
|  $T$ | $F$ | $F$ | $F$ | $\mathbf{F}$  |
|  $F$ | $T$ | $T$ | $T$ | $\mathbf{F}$  |
|  $F$ | $T$ | $F$ | $T$ | $\mathbf{F}$  |
|  $F$ | $F$ | $T$ | $T$ | $\mathbf{F}$  |
|  $F$ | $F$ | $F$ | $F$ | $\mathbf{T}$  |

We fill in the fourth column using our knowledge of the truth table for $\vee$. Finally the fifth column is filled in by combining the first and fourth columns with our understanding of the truth table for $\Leftrightarrow$. The resulting table gives the true/false values of $P \Leftrightarrow (Q \vee R)$ for all values of $P$, $Q$ and $R$.

Notice that when we plug in various values for $x$ and $y$, the statements $P: xy = 0$, $Q: x = 0$ and $R: y = 0$ have various truth values, but the statement $P \Leftrightarrow (Q \vee R)$ is always true. For example, if $x = 2$ and $y = 3$, then $P$, $Q$ and $R$ are all false. This scenario is described in the last row of the table, and there we see that $P \Leftrightarrow (Q \vee R)$ is true. Likewise if $x = 0$ and $y = 7$, then $P$ and $Q$ are true and $R$ is false, a scenario described in the second line of the table, where again $P \Leftrightarrow (Q \vee R)$ is true. There is a simple reason why $P \Leftrightarrow (Q \vee R)$ is true for any values of $x$ and $y$: It is that $P \Leftrightarrow (Q \vee R)$ represents $(xy = 0) \Leftrightarrow (x = 0 \vee y = 0)$, which is a *true mathematical statement*. It is absolutely impossible for it to be false.

This may make you wonder about the lines in the table where $P \Leftrightarrow (Q \vee R)$ is false. Why are they there? The reason is that $P \Leftrightarrow (Q \vee R)$ can also represent a false statement. To see how, imagine that at the end of the semester your professor makes the following promise.

You pass the class if and only if you get an “A” on the final or you get a “B” on the final.

This promise has the form $P \Leftrightarrow (Q \vee R)$, so its truth values are tabulated in the above table. Imagine it turned out that you got an “A” on the exam but failed the course. Then surely your professor lied to you. In fact, $P$ is false, $Q$ is true and $R$ is false. This scenario is reflected in the sixth line of the table, and indeed $P \Leftrightarrow (Q \vee R)$ is false (i.e., it is a lie).

Free PDF version
![Creative Commons License Logo]() BY-NC-ND