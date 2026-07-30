**Exercise 8.1.9:** Let us show that *Proposition 8.1.18* only works in finite dimensions. Take the space of polynomials $\mathbb{R}[t]$ and define the operator $A: \mathbb{R}[t] \to \mathbb{R}[t]$ by $A(P(t)) := tP(t)$. Show that $A$ is linear and one-to-one, but show that it is not onto.

**Exercise 8.1.10:** Finish the proof of *Proposition 8.1.17* in the finite-dimensional case. That is, suppose $\{x_1, x_2, \dots, x_n\}$ is a basis of $X$, $\{y_1, y_2, \dots, y_n\} \subset Y$, and define a function

$$A(x) := \sum_{k=1}^n b_k y_k, \quad \text{if} \quad x = \sum_{k=1}^n b_k x_k.$$

Prove that $A: X \to Y$ is linear.

**Exercise 8.1.11:** Prove *Proposition 8.1.19*. Hint: A linear transformation is determined by its action on a basis. So given two bases $\{x_1, \dots, x_n\}$ and $\{y_1, \dots, y_m\}$ for $X$ and $Y$ respectively, consider the linear operators $A_{jk}$ that send $A_{jk}x_j = y_k$, and $A_{jk}x_\ell = 0$ if $\ell \neq j$.

**Exercise 8.1.12 (Easy):** Suppose $X$ and $Y$ are vector spaces and $A \in L(X, Y)$ is a linear mapping.

a) Show that the nullspace $N := \{x \in X : Ax = 0\}$ is a vector space.
b) Show that the range $R := \{y \in Y : Ax = y \text{ for some } x \in X\}$ is a vector space.

**Exercise 8.1.13 (Easy):** Show by example that a union of convex sets need not be convex.

**Exercise 8.1.14:** Compute the convex hull of the set of 3 points $\{(0, 0), (0, 1), (1, 1)\}$ in $\mathbb{R}^2$.

**Exercise 8.1.15:** Show that the set $\{(x, y) \in \mathbb{R}^2 : y > x^2\}$ is a convex set.

**Exercise 8.1.16:** Show that the set $X \subset C([0, 1], \mathbb{R})$ of those functions such that $\int_0^1 f = 1$ is a convex set, but not a vector subspace. Compare *Exercise 8.1.4*.

**Exercise 8.1.17:** Show that every convex set in $\mathbb{R}^n$ is connected using the standard topology on $\mathbb{R}^n$.

**Exercise 8.1.18:** Suppose $K \subset \mathbb{R}^2$ is a convex set such that the only point of the form $(x, 0)$ in $K$ is the point $(0, 0)$. Further suppose that $(0, 1) \in K$ and $(1, 1) \in K$. Show that if $(x, y) \in K$ and $x \neq 0$, then $y > 0$.

**Exercise 8.1.19:** Prove that an arbitrary intersection of vector subspaces is a vector subspace. That is, if $X$ is a vector space and $\{V_\lambda\}_{\lambda \in I}$ is an arbitrary collection of vector subspaces of $X$, then $\bigcap_{\lambda \in I} V_\lambda$ is a vector subspace of $X$.

**Exercise 8.1.20 (Easy):** Finish the proof of *Proposition 8.1.16*, that is, prove the first four items of the proposition.