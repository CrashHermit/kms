Let $E = \{\mathrm{HH}, \mathrm{HT}, \mathrm{TH}\}$ be the event that at least one head comes up. Then, the probability of $E$ can be calculated as follows:

$$
\begin{array}{l} P(E) = m(\mathrm{HH}) + m(\mathrm{HT}) \\ = \frac{1}{4} + \frac{1}{4} + \frac{1}{4} = \frac{3}{4}. \end{array}
$$

Similarly, if $F = \{\mathrm{HH}, \mathrm{HT}\}$ is the event that heads comes up on the first toss, then we have

$$
\begin{array}{l} P(F) = m(\mathrm{HH}) + m(\mathrm{HT}) \\ = \frac{1}{4} + \frac{1}{4} = \frac{1}{2}. \end{array}
$$

□

**Example 1.8** (Example 1.6 continued) The sample space for the experiment in which the die is rolled is the 6-element set $\Omega = \{1, 2, 3, 4, 5, 6\}$. We assumed that the die was fair, and we chose the distribution function defined by

$$
m(i) = \frac{1}{6}, \qquad \text{for } i = 1, \dots, 6.
$$

If $E$ is the event that the result of the roll is an even number, then $E = \{2, 4, 6\}$ and

$$
\begin{array}{l} P(E) = m(2) + m(4) + m(6) \\ = \frac{1}{6} + \frac{1}{6} + \frac{1}{6} = \frac{1}{2}. \end{array}
$$

□

Notice that it is an immediate consequence of the above definitions that, for every $\omega \in \Omega$,

$$
P(\{\omega\}) = m(\omega).
$$

That is, the probability of the elementary event $\{\omega\}$, consisting of a single outcome $\omega$, is equal to the value $m(\omega)$ assigned to the outcome $\omega$ by the distribution function.

**Example 1.9** Three people, A, B, and C, are running for the same office, and we assume that one and only one of them wins. The sample space may be taken as the 3-element set $\Omega = \{\mathrm{A}, \mathrm{B}, \mathrm{C}\}$ where each element corresponds to the outcome of that candidate's winning. Suppose that A and B have the same chance of winning, but that C has only $1/2$ the chance of A or B. Then we assign

$$
m(\mathrm{A}) = m(\mathrm{B}) = 3m(\mathrm{C}).
$$

Since

$$
m(\mathrm{A}) + m(\mathrm{B}) + m(\mathrm{C}) = 1,
$$