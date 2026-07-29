**Example 8.1.24:** The convex hull of $\{0, 1\}$ in $\mathbb{R}$ is $[0, 1]$. Proof: A convex set containing 0 and 1 must contain $[0, 1]$, so $[0, 1] \subset \text{co}(\{0, 1\})$. The set $[0, 1]$ is convex and contains $\{0, 1\}$, so $\text{co}(\{0, 1\}) \subset [0, 1]$.

Linear mappings preserve convex sets. So in some sense, convex sets are the right sort of sets when considering linear mappings or changes of coordinates.

**Proposition 8.1.25.** Let $X, Y$ be vector spaces, $A \in L(X, Y)$, and let $C \subset X$ be convex. Then $A(C)$ is convex.

Proof. Take two points $p, q \in A(C)$. Pick $u, v \in C$ such that $Au = p$ and $Av = q$. As $C$ is convex, $(1 - t)u + tv \in C$ for all $t \in [0, 1]$. So

$$(1 - t)p + tq = (1 - t)Au + tAv = A((1 - t)u + tv) \in A(C). \quad \square$$

## 8.1.5 Exercises

**Exercise 8.1.1:** Show that in $\mathbb{R}^n$ (with the standard euclidean metric), for every $x \in \mathbb{R}^n$ and every $r > 0$, the ball $B(x, r)$ is convex.

**Exercise 8.1.2:** Verify that $\mathbb{R}^n$ is a vector space.

**Exercise 8.1.3:** Let $X$ be a vector space. Prove that a finite set of vectors $\{x_1, x_2, \dots, x_n\} \subset X$ is linearly independent if and only if for every $k = 1, 2, \dots, n$

$$\text{span}(\{x_1, \dots, x_{k-1}, x_{k+1}, \dots, x_n\}) \subset \text{span}(\{x_1, x_2, \dots, x_n\}).$$

That is, the span of the set with one vector removed is strictly smaller.

**Exercise 8.1.4:** Show that the set $X \subset C([0, 1], \mathbb{R})$ of those functions such that $\int_0^1 f = 0$ is a vector subspace. Compare *Exercise 8.1.16*.

**Exercise 8.1.5 (Challenging):** Prove $C([0, 1], \mathbb{R})$ is an infinite-dimensional vector space where the operations are defined in the obvious way: $s = f + g$ and $m = af$ are defined as $s(x) := f(x) + g(x)$ and $m(x) := af(x)$. Hint: For the dimension, think of functions that are only nonzero on the interval $\left(\frac{1}{n+1}, \frac{1}{n}\right)$.

**Exercise 8.1.6:** Let $k: [0, 1]^2 \to \mathbb{R}$ be continuous. Show that $L: C([0, 1], \mathbb{R}) \to C([0, 1], \mathbb{R})$ defined by

$$Lf(y) := \int_0^1 k(x, y)f(x) \, dx$$

is a linear operator. That is, first show that $L$ is well-defined by showing that $Lf$ is continuous whenever $f$ is, and then showing that $L$ is linear.

**Exercise 8.1.7:** Let $\mathcal{P}_n$ be the vector space of polynomials in one variable of degree $n$ or less. Show that $\mathcal{P}_n$ is a vector space of dimension $n + 1$.

**Exercise 8.1.8:** Let $\mathbb{R}[t]$ be the vector space of polynomials in one variable $t$. Let $D: \mathbb{R}[t] \to \mathbb{R}[t]$ be the derivative operator (derivative in $t$). Show that $D$ is a linear operator.