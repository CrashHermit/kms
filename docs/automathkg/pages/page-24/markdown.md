queries from external sources, both of our VDs are capable of retrieving various types of highly relevant entities, indicating their effectiveness in handling external queries.

Table 10 The results of external query example in MathVD1.

|  External query: Expectation in Probability and Statistics  |   |
| --- | --- |
|  Retrieval results in MathVD1  |   |
|  Rank | Entity  |
|  1 | Theorem: expectation of geometric distribution Let $X$ be a discrete random variable with the geometric distribution with parameter $p$. Then the expectation of $X$ is given by: $E(X) = \frac{1-p}{p}$.  |
|  2 | Problem: expectation of waiting time Let $X_0, X_1, X_2, \cdots$ be drawn i.i.d. from $p(x)$, and $x \in \{1, 2, 3, \cdots, 100\}$. Let $N$ be the waiting time to the next occurrence of $X_0$. Compute $E(N)$.  |
|  3 | Theorem: expectation of discrete uniform distribution Let $X$ be a discrete random variable with the discrete uniform distribution with parameter $n$. Then the expectation of $X$ is given by: $E(X) = \frac{n+1}{2}$.  |
|  4 | Definition: expectation of random vector Let $X_1, X_2, \cdots, X_n$ be random variables on a probability space $(\Omega, \Sigma, Pr)$. Let $X = (X_1, X_2, \cdots, X_n)$ be a random vector. Then the expected value of $X$, $E(X)$, is defined by: $E(X) = (E(X_1), E(X_2), \cdots, E(X_n))$.  |
|  5 | Theorem: sum of expectations of independent trials: Let $E_1, E_2, \cdots, E_n$ be a sequence of experiments whose outcomes are independent of each other. Let $X_1, X_2, \cdots, X_n$ be discrete random variables on $E_1, E_2, \cdots, E_n$ respectively. Let $E(X_j)$ denote the expectation of $X_j$ for $j \in \{1, 2, \cdots, n\}$. Then we have, whenever both sides are defined: $E(\sum_{j=1}^n X_j) = \sum_{j=1}^n E(X_j)$. That is, the sum of the expectations equals the expectation of the sum.  |

Table 11 The results of external query example in MathVD2.

|  External query: Expectation in Probability and Statistics  |   |
| --- | --- |
|  Retrieval results in MathVD2  |   |
|  Rank | Entity  |
|  1 | Definition: expectation The expectation of a random variable is the arithmetic mean of its value.  |
|  2 | Theorem: expectation of binomial distribution Let $X$ be a discrete random variable with the binomial distribution with parameters $n$ and $p$ for some $n \in \mathbb{N}$ and $0 \le p \le 1$. Then the expectation of $X$ is given by: $E(X) = np$.  |
|  3 | Theorem: expectation of geometric distribution Let $X$ be a discrete random variable with the geometric distribution with parameter $p$. Then the expectation of $X$ is given by: $E(X) = \frac{1-p}{p}$.  |
|  4 | Theorem: sum of expectations of independent trials: Let $E_1, E_2, \cdots, E_n$ be a sequence of experiments whose outcomes are independent of each other. Let $X_1, X_2, \cdots, X_n$ be discrete random variables on $E_1, E_2, \cdots, E_n$ respectively. Let $E(X_j)$ denote the expectation of $X_j$ for $j \in \{1, 2, \cdots, n\}$. Then we have, whenever both sides are defined: $E(\sum_{j=1}^n X_j) = \sum_{j=1}^n E(X_j)$. That is, the sum of the expectations equals the expectation of the sum.  |
|  5 | Theorem: expectation of bernoulli distribution Let $X$ be a discrete random variable with a Bernoulli distribution with parameter $p$. Then the expectation of $X$ is given by: $E(X) = p$.  |