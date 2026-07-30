We will prove that there is a primitive root modulo every prime p. Since the unit group (Z/pZ)* has order p-1, this implies that (Z/pZ)* is a cyclic group, a fact that will be extremely useful, since it completely determines the structure of (Z/pZ)* as a group.

If n is an odd prime power, then there is a primitive root modulo n (see Exercise 2.28), but there is no primitive root modulo the prime power 2², and hence none mod 2ⁿ for n ≥ 3 (see Exercise 2.27).

Section 2.5.1 is the key input to our proof that (Z/pZ)* is cyclic; here we show that for every divisor d of p-1 there are exactly d elements of (Z/pZ)* whose order divides d. We then use this result in Section 2.5.2 to produce an element of (Z/pZ)* of order qʳ when qʳ is a prime power that exactly divides p-1 (i.e., qʳ divides p-1, but qʳ⁺¹ does not divide p-1), and multiply together these elements to obtain an element of (Z/pZ)* of order p-1.

SAGE Example 2.5.2. Use the primitive_root command to compute the smallest positive integer that is a primitive root modulo n. For example, below we compute primitive roots modulo p for each prime p ≤ 20.

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

### 2.5.1 Polynomials over Z/pZ

The polynomials x² - 1 has four roots in Z/8Z, namely 1, 3, 5, and 7. In contrast, the following proposition shows that a polynomial of degree d over a field, such as Z/pZ, can have at most d roots.

**Proposition 2.5.3** (Root Bound). *Let f ∈ k[x] be a nonzero polynomial over a field k. Then there are at most deg(f) elements α ∈ k such that f(α) = 0.*