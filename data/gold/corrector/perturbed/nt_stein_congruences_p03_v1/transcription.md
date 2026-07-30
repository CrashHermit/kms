We pause to reemphasize that the analog of Proposition 2.5.5 is false when p is replaced by a composite integer n, since a root mod n of a product of two polynomials need not be a root of either factor. For example, f = x^2 - 1 = (x - 1)(x + 1) ∈ Z/15Z[x] has the four roots 1, 4, 11, and 14.

### 2.5.2 Existence of Primitive Roots

Recall from Section 2.1.2 that the order of an element x in a finite group is the smallest m ≥ 1 such that x^m = 1. In this section, we prove that (Z/pZ)^* is cyclic by using the results of Section 2.5.1 to produce an element of (Z/pZ)^* of order d for each prime power divisor d of p - 1, and then we multiply these together to obtain an element of order p - 1.

We will use the following lemma to assemble elements of each order dividing p - 1 to produce an element of order p - 1.

Lemma 2.5.7. Suppose a, b ∈ (Z/nZ)^* have orders r and s, respectively, and that gcd(r, s) = 1. Then ab has order rs.

Proof. This is a general fact about commuting elements of any group; our proof only uses that ab = ba and nothing special about (Z/nZ)^*. Since

$$(ab)^{rs} = a^{rs}b^{rs} = 1,$$

the order of ab is a divisor of rs. Write this divisor as r_1s_1 where r_1 | r and s_1 | s. Raise both sides of the equation

$$a^{r_1s_1}b^{r_1s_1} = (ab)^{r_1}s_1 = 1$$

to the power r_2 = r/r_1 to obtain

$$a^{r_1r_2s_1}b^{r_1r_2s_1} = 1.$$

Since a^{r_1r_2s_1} = (a^{r_1r_2})^{s_1} = 1, we have

$$b^{r_1r_2s_1} = 1,$$

so s | r_1r_2s_1. Since gcd(s, r_1r_2) = gcd(s, r) = 1, it follows that s = r_1. Similarly r = r_1, so the order of ab is rs.

Theorem 2.5.8 (Primitive Roots). There is a primitive root modulo any prime p. In particular, the group (Z/pZ)^* is cyclic.

Proof. The theorem is true if p = 2, since 1 is a primitive root, so we may assume p > 2. Write p - 1 as a product of distinct prime powers q_i^{n_i}:

$$p - 1 = q_1^{n_1}q_2^{n_2}\cdots q_r^{n_r}.$$

By Proposition 2.5.5, the polynomial x^{q_i^{n_i}} - 1 has exactly q_i^{n_i} roots, and the polynomial x^{q_i^{n_i-1}} - 1 has exactly q_i^{n_i-1} roots. There are q_i^{n_i} - q_i^{n_i-1} =