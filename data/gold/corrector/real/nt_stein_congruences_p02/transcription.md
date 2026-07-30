*Proof.* We prove the proposition by induction on $\deg(f)$. The cases in which $\deg(f) \leq 1$ are clear. Write $f = a_n x^n + \cdots a_1 x + a_0$. If $f(\alpha) = 0$, then

$$\begin{aligned} f(x) &= f(x) - f(\alpha) \\ &= a_n(x^n - \alpha^n) + \cdots + a_1(x - \alpha) + a_0(1 - 1) \\ &= (x - \alpha)(a_n(x^{n-1} + \cdots + \alpha^{n-1}) + \cdots + a_2(x + \alpha) + a_1) \\ &= (x - \alpha)g(x), \end{aligned}$$

for some polynomial $g(x) \in k[x]$. Next, suppose that $f(\beta) = 0$ with $\beta \neq \alpha$. Then $(\beta - \alpha)g(\beta) = 0$, so, since $\beta - \alpha \neq 0$ and $k$ is a field, we have $g(\beta) = 0$. By our inductive hypothesis, $g$ has at most $n - 1$ roots, so there are at most $n - 1$ possibilities for $\beta$. It follows that $f$ has at most $n$ roots. $\square$

*SAGE Example 2.5.4.* We use Sage to find the roots of a polynomials over $\mathbf{Z}/13\mathbf{Z}$.

sage: R.<x> = PolynomialRing(Integers(13))
sage: f = x^15 + 1
sage: f.roots()
[(12, 1), (10, 1), (4, 1)]
sage: f(12)
0</x>

The output of the roots command above lists each root along with its multiplicity (which is 1 in each case above).

**Proposition 2.5.5.** *Let $p$ be a prime number and let $d$ be a divisor of $p - 1$. Then $f = x^d - 1 \in (\mathbf{Z}/p\mathbf{Z})[x]$ has exactly $d$ roots in $\mathbf{Z}/p\mathbf{Z}$.*

*Proof.* Let $e = (p - 1)/d$. We have

$$\begin{aligned} x^{p-1} - 1 &= (x^d)^e - 1 \\ &= (x^d - 1)((x^d)^{e-1} + (x^d)^{e-2} + \cdots + 1) \\ &= (x^d - 1)g(x), \end{aligned}$$

where $g \in (\mathbf{Z}/p\mathbf{Z})[x]$ and $\deg(g) = de - d = p - 1 - d$. Theorem 2.1.20 implies that $x^{p-1} - 1$ has exactly $p - 1$ roots in $\mathbf{Z}/p\mathbf{Z}$, since every nonzero element of $\mathbf{Z}/p\mathbf{Z}$ is a root! By Proposition 2.5.3, $g$ has *at most* $p - 1 - d$ roots and $x^d - 1$ has at most $d$ roots. Since a root of $(x^d - 1)g(x)$ is a root of either $x^d - 1$ or $g(x)$ and $x^{p-1} - 1$ has $p - 1$ roots, $g$ must have exactly $p - 1 - d$ roots and $x^d - 1$ must have exactly $d$ roots, as claimed. $\square$

*SAGE Example 2.5.6.* We use Sage to illustrate the proposition.

sage: R.<x> = PolynomialRing(Integers(13))
sage: f = x^6 + 1
sage: f.roots()
[(11, 1), (8, 1), (7, 1), (6, 1), (5, 1), (2, 1)]</x>