where each outcome $i$, for $i = 1, \dots, 6$, corresponds to the number of dots on the face which turns up. The event

$$E = \{2, 4, 6\}$$

corresponds to the statement that the result of the roll is an even number. The event $E$ can also be described by saying that $X$ is even. Unless there is reason to believe the die is loaded, the natural assumption is that every outcome is equally likely. Adopting this convention means that we assign a probability of $1/6$ to each of the six outcomes, i.e., $m(i) = 1/8$, for $1 \leq i \leq 6$. $\square$

## Distribution Functions

We next describe the assignment of probabilities. The definitions are motivated by the example above, in which we assigned to each outcome of the sample space a nonnegative number such that the sum of the numbers assigned is equal to 1.

**Definition 1.2** Let $X$ be a random variable which denotes the value of the outcome of a certain experiment, and assume that this experiment has only finitely many possible outcomes. Let $\Omega$ be the sample space of the experiment (i.e., the set of all possible values of $X$, or equivalently, the set of all possible outcomes of the experiment.) A *distribution function* for $X$ is a real-valued function $m$ whose domain is $\Omega$ and which satisfies:

1. $m(\omega) \geq 0$, for all $\omega \in \Omega$, and
2. $\sum_{\omega \in \Omega} m(\omega) = 1$.

For any subset $E$ of $\Omega$, we define the *probability* of $E$ to be the number $P(E)$ given by

$$P(E) = \sum_{\omega \in \Omega} m(\omega).$$

$\square$

**Example 1.7** Consider an experiment in which a coin is tossed twice. Let $X$ be the random variable which corresponds to this experiment. We note that there are several ways to record the outcomes of this experiment. We could, for example, record the two tosses, in the order in which they occurred. In this case, we have $\Omega = \{HH, HT, TH, TT\}$. We could also record the outcomes by simply noting the number of heads that appeared. In this case, we have $\Omega = \{0, 1, 2\}$. Finally, we could record the two outcomes, without regard to the order in which they occurred. In this case, we have $\Omega = \{HH, HT, TT\}$.

We will use, for the moment, the first of the sample spaces given above. We will assume that all four outcomes are equally likely, and define the distribution function $m(\omega)$ by

$$m(\mathrm{HH}) = m(\mathrm{HT}) = m(\mathrm{TH}) = m(\mathrm{TT}) = \frac{1}{4}.$$