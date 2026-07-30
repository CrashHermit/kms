We now turn to a very important notion that you may not have met before.

1.3.7 Definition. Let $f$ be a function from a set $X$ into a set $Y$. If $S$ is any subset of $Y$, then the set $f^{-1}(S)$ is defined by

$$f^{-1}(S) = \{x : x \in X \text{ and } f(x) \in S\}.$$

The subset $f^{-1}(S)$ of $X$ is said to be the inverse image of $S$.

Note that an inverse function of $f: X \to Y$ exists if and only if $f$ is bijective. But the inverse image of any subset of $Y$ exists even if $f$ is neither one-to-one nor onto. The next example demonstrates this.

1.3.8 Example. Let $f$ be the function from the set of integers, $\mathbb{Z}$, into itself given by $f(z) = |z|$, for each $z \in \mathbb{Z}$.

The function $f$ is not one-to one, since $f(1) = f(-1)$.

It is also not onto, since there is no $z \in \mathbb{Z}$, such that $f(z) = -1$. So $f$ is certainly not bijective. Hence, by Proposition 1.3.6 (i), $f$ does not have an inverse function. However inverse images certainly exist. For example,

$$f^{-1}(\{1, 2, 3\}) = \{-1, -2, -3, 1, 2, 3\}$$

$$f^{-1}(\{-5, 3, 5, 7, 9\}) = \{-3, -5, -7, -9, 3, 5, 7, 9\}.$$

We conclude this section with an interesting example.

1.3.9 Example. Let $(Y, \mathcal{T})$ be a topological space and $X$ a non-empty set. Further, let $f$ be a function from $X$ into $Y$. Put $\mathcal{T}_1 = \{f^{-1}(S) : S \in \mathcal{T}\}$. Prove that $\mathcal{T}_1$ is a topology on $X$.

# Proof.

Our task is to show that the collection of sets, $\mathcal{T}_1$, is a topology on $X$; that is, we have to show that $\mathcal{T}_1$ satisfies conditions (i), (ii) and (iii) of Definitions 1.1.1