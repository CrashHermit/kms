Oftentimes it is useful to consider a subset of a larger metric space as a metric space itself. We obtain the following proposition, which has a trivial proof.

**Proposition 7.1.10.** Let $(X, d)$ be a metric space and $Y \subset X$. Then the restriction $d|_{Y \times Y}$ is a metric on $Y$.

**Definition 7.1.11.** If $(X, d)$ is a metric space, $Y \subset X$, and $d' := d|_{Y \times Y}$, then $(Y, d')$ is said to be a subspace of $(X, d)$.

It is common to simply write $d$ for the metric on $Y$, as it is the restriction of the metric on $X$. Sometimes we say $d'$ is the subspace metric and $Y$ has the subspace topology.

A subset of the real numbers is bounded whenever all its elements are at most some fixed distance from 0. When dealing with an arbitrary metric space there may not be some natural fixed point 0, but for the purposes of boundedness it does not matter.

**Definition 7.1.12.** Let $(X, d)$ be a metric space. A subset $S \subset X$ is said to be bounded if there exists a $p \in X$ and a $B \in \mathbb{R}$ such that

$$d(p, x) \leq B \quad \text{for all } x \in S.$$

We say $(X, d)$ is bounded if $X$ itself is a bounded subset.

For example, the set of real numbers with the standard metric is not a bounded metric space. It is not hard to see that a subset of the real numbers is bounded in the sense of chapter 1 if and only if it is bounded as a subset of the metric space of real numbers with the standard metric.

On the other hand, if we take the real numbers with the discrete metric, then we obtain a bounded metric space. In fact, any set with the discrete metric is bounded.

There are other equivalent ways we could generalize boundedness, which are left as exercises. Suppose $X$ is nonempty to avoid a technicality. Then $S \subset X$ being bounded is equivalent to either

- (i) For every $p \in X$, there exists a $B > 0$ such that $d(p, x) \leq B$ for all $x \in S$.
- (ii) $\text{diam}(S) := \sup\{d(x, y) : x, y \in S\} < \infty$.

The quantity $\text{diam}(S)$ is called the diameter of a set and is usually only defined for a nonempty set.

### 7.1.1 Exercises

**Exercise 7.1.1:** Show that for every set $X$, the discrete metric ($d(x, y) = 1$ if $x \neq y$ and $d(x, x) = 0$) does give a metric space $(X, d)$.

**Exercise 7.1.2:** Let $X := \{0\}$ be a set. Can you make it into a metric space?

**Exercise 7.1.3:** Let $X := \{a, b\}$ be a set. Can you make it into two distinct metric spaces? (define two distinct metrics on it)