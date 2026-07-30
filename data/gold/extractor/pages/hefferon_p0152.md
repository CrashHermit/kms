3.30 Give an example to show that, despite that they have the same dimension, the row space and column space of a matrix need not be equal. Are they ever equal?

3.31 Show that the set $\{(1, -1, 2, -3), (1, 1, 2, 0), (3, -1, 6, -6)\}$ does not have the same span as $\{(1, 0, 1, 0), (0, 2, 0, 3)\}$. What, by the way, is the vector space?

✓ 3.32 Show that this set of column vectors

$$\begin{cases} \begin{pmatrix} d_1 \\ d_2 \\ d_3 \end{pmatrix} \mid \text{there are } x, y, \text{ and } z \text{ such that:} \begin{array}{l} 3x + 2y + 4z = d_1 \\ x - z = d_2 \end{array} \\ 2x + 2y + 5z = d_3 \end{cases}$$

is a subspace of $\mathbb{R}^3$. Find a basis.

3.33 Show that the transpose operation is linear:

$$(rA + sB)^T = rA^T + sB^T$$

for $r, s \in \mathbb{R}$ and $A, B \in \mathcal{M}_{m \times n}$.

✓ 3.34 In this subsection we have shown that Gaussian reduction finds a basis for the row space.

- (a) Show that this basis is not unique—different reductions may yield different bases.
- (b) Produce matrices with equal row spaces but unequal numbers of rows.
- (c) Prove that two matrices have equal row spaces if and only if after Gauss-Jordan reduction they have the same nonzero rows.

3.35 Why is there not a problem with Remark 3.15 in the case that $r$ is bigger than $n$?

3.36 Show that the row rank of an $m \times n$ matrix is at most $m$. Is there a better bound?

3.37 Show that the rank of a matrix equals the rank of its transpose.

3.38 True or false: the column space of a matrix equals the row space of its transpose.

✓ 3.39 We have seen that a row operation may change the column space. Must it?

3.40 Prove that a linear system has a solution if and only if that system's matrix of coefficients has the same rank as its augmented matrix.

3.41 An $m \times n$ matrix has *full row rank* if its row rank is $m$, and it has *full column rank* if its column rank is $n$.

- (a) Show that a matrix can have both full row rank and full column rank only if it is square.
- (b) Prove that the linear system with matrix of coefficients $A$ has a solution for any $d_1, \dots, d_n$'s on the right side if and only if $A$ has full row rank.
- (c) Prove that a homogeneous system has a unique solution if and only if its matrix of coefficients $A$ has full column rank.
- (d) Prove that the statement "if a system with matrix of coefficients $A$ has any solution then it has a unique solution" holds if and only if $A$ has full column rank.

3.42 How would the conclusion of Lemma 3.3 change if Gauss's Method were changed to allow multiplying a row by zero?

3.43 What is the relationship between $\text{rank}(A)$ and $\text{rank}(-A)$? Between $\text{rank}(A)$ and $\text{rank}(kA)$? What, if any, is the relationship between $\text{rank}(A)$, $\text{rank}(B)$, and $\text{rank}(A + B)$?