$$X \in \mathcal{T}_1 \quad \text{since} \quad X = f^{-1}(Y) \quad \text{and} \quad Y \in \mathcal{T}.$$

$$\emptyset \in \mathcal{T}_1 \quad \text{since} \quad \emptyset = f^{-1}(\emptyset) \quad \text{and} \quad \emptyset \in \mathcal{T}.$$

Therefore $\mathcal{T}_1$ has property (i) of Definitions 1.1.1.

To verify condition (ii) of Definitions 1.1.1, let $\{A_j : j \in J\}$ be a collection of members of $\mathcal{T}_1$, for some index set $J$. We have to show that $\bigcup_{j \in J} A_j \in \mathcal{T}_1$.

As $A_j \in \mathcal{T}_1$, the definition of $\mathcal{T}_1$ implies that $A_j = f^{-1}(B)_j$, where $B_j \in \mathcal{T}$. Also $\bigcup_{j \in J} A_j = \bigcup_{j \in J} f^{-1}(B_j) = f^{-1}\left(\bigcup_{j \in J} B_j\right)$. [See Exercises 1.3 # 1.]

Now $B_j \in \mathcal{T}$, for all $j \in J$, and so $\bigcup_{j \in J} B_j \in \mathcal{T}$, since $\mathcal{T}$ is a topology on $Y$. Therefore, by the definition of $\mathcal{T}_1$, $f^{-1}\left(\bigcup_{j \in J} B_j\right) \in \mathcal{T}_1$; that is, $\bigcup_{j \in J} A_j \in \mathcal{T}_1$.

So $\mathcal{T}_1$ has property (ii) of Definitions 1.1.1.

[Warning. You are reminded that not all sets are countable. (See the Appendix for comments on countable sets.) So it would not suffice, in the above argument, to assume that sets $A_1, A_2, \ldots, A_n, \ldots$ are in $\mathcal{T}_1$ and show that their union $A_1 \cup A_2 \cup \ldots \cup A_n \cup \ldots$ is in $\mathcal{T}_1$. This would prove only that the union of a countable number of sets in $\mathcal{T}_1$ lies in $\mathcal{T}_1$, but would not show that $\mathcal{T}_1$ has property (ii) of Definitions 1.1.1– this property requires all unions, whether countable or uncountable, of sets in $\mathcal{T}_1$ to be in $\mathcal{T}_1$.]

Finally, let $A_1$ and $A_2$ be in $\mathcal{T}_1$. We have to show that $A_1 \cap A_2 \in \mathcal{T}_1$.

As $A_1, A_2 \in \mathcal{T}_1$, $A_1 = f^{-1}(B_1)$ and $A_2 = f^{-1}(B_2)$, where $B_1, B_2 \in \mathcal{T}$.

$$A_1 \cap A_2 = f^{-1}(B_1) \cap f^{-1}(B_2) = f^{-1}(B_1 \cap B_2). \quad [\text{See Exercises 1.3 \#1.}]$$

As $B_1 \cap B_2 \in \mathcal{T}$, we have $f^{-1}(B_1 \cap B_2) \in \mathcal{T}_1$. Hence $A_1 \cap A_2 \in \mathcal{T}_1$, and we have shown that $\mathcal{T}_1$ also has property (iii) of Definitions 1.1.1.

So $\mathcal{T}_1$ is indeed a topology on $X$.

□