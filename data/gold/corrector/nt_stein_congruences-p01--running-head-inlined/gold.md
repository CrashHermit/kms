We will prove that there is a primitive root modulo every prime $p$. Since the unit group $(\mathbf{Z}/p\mathbf{Z})^*$ has order $p-1$, this implies that $(\mathbf{Z}/p\mathbf{Z})^*$ is a cyclic group, a fact that will be extremely useful, since it completely determines the structure of $(\mathbf{Z}/p\mathbf{Z})^*$ as a group.

If $n$ is an odd prime power, then there is a primitive root modulo $n$ (see Exercise 2.28), but there is no primitive root modulo the prime power $2^3$, and hence none mod $2^n$ for $n \geq 3$ (see Exercise 2.27).

Section 2.5.1 is the key input to our proof that $(\mathbf{Z}/p\mathbf{Z})^*$ is cyclic; here we show that for every divisor $d$ of $p-1$ there are exactly $d$ elements of $(\mathbf{Z}/p\mathbf{Z})^*$ whose order divides $d$. We then use this result in Section 2.5.2 to produce an element of $(\mathbf{Z}/p\mathbf{Z})^*$ of order $q^r$ when $q^r$ is a prime power that exactly divides $p-1$ (i.e., $q^r$ divides $p-1$, but $q^{r+1}$ does not divide $p-1$), and multiply together these elements to obtain an element of $(\mathbf{Z}/p\mathbf{Z})^*$ of order $p-1$.

SAGE Example 2.5.2. Use the primitive_root command to compute the smallest positive integer that is a primitive root modulo $n$. For example, below we compute primitive roots modulo $p$ for each prime $p < 20$.

sage: for p in primes(20):
...     print p, primitive_root(p)
2 1
3 2
5 2
7 3
11 2
13 2
17 3
19 2

### 2.5.1 Polynomials over $\mathbf{Z}/p\mathbf{Z}$

The polynomials $x^2 - 1$ has four roots in $\mathbf{Z}/8\mathbf{Z}$, namely 1, 3, 5, and 7. In contrast, the following proposition shows that a polynomial of degree $d$ over a field, such as $\mathbf{Z}/p\mathbf{Z}$, can have at most $d$ roots.

**Proposition 2.5.3** (Root Bound). *Let $f \in k[x]$ be a nonzero polynomial over a field $k$. Then there are at most $\deg(f)$ elements $\alpha \in k$ such that $f(\alpha) = 0$.*
