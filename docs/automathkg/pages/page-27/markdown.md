**Table 14** The results of the case study on automatic knowledge completion.

|  Theorem proof  |   |
| --- | --- |
|  Question | Let $f$ have a continuous derivative on $[a, b]$. We want to show that $f'$ satisfies a Lipschitz condition on $[a, b]$. By [MISSING], $f(x) - f(y) = f'(x_i)(x - y)$ for some $x_i \in (y, x)$. But since $f'$ is continuous and $[a, b]$ is compact, then $f'$ is bounded in that interval, say by $C$. Thus taking absolute values yields $|f(x) - f(y)| \leq C|x - y|$. Can you name the result used in [MISSING]?  |
|  Answer | The answer is the Mean Value Theorem. The missing part in the text is the Mean Value Theorem. The Mean Value Theorem states that for a continuous function $f$ on the interval $[a, b]$, there exists a point $c$ in $(a, b)$ such that $f(b) - f(a) = f'(c)$ $(b - a)$. This theorem is used to show that $f$ satisfies a Lipschitz condition on $[a, b]$.  |
|  Problem solution  |   |
|  Question | What is the smallest positive multiple of 6 that is also a multiple of 8? Solution: To find a positive number that is both a multiple of 6 and a multiple of 8, we'll list out the multiples of 6, starting with 6, and check to see if any of them are also multiples of 8. 6 does not divide 8, so it isn't a multiple of 8. 12 and 18 are also not evenly divisible by 8. [MISSING]. Fill in the missing steps.  |
|  Answer | Sure, here is the answer to the question: To find the smallest positive multiple of 6 that is also a multiple of 8, we'll list out the multiples of 6, starting with 6, and check to see if any of them are also multiples of 8. 6 does not divide 8, so it isn't a multiple of 8. 12 and 18 are also not evenly divisible by 8. The next multiple of 6 is 24, and it is also a multiple of 8. Therefore, the smallest positive multiple of 6 that is also a multiple of 8 is 24.  |

### 8.3.2 Knowledge fusion

In order to achieve automatic knowledge fusion, this paper utilizes VDs and LLM to update entities, ensuring that entities and relationships from different sources are added to the Existing KG without redundancy or omission. Table 15 presents an example of Theorem entity fusion from different sources. The Theorem entity “*Mean Value Theorem*” from a textbook is a necessary condition for the Theorem entity “*Carathéodory’s Theorem (Analysis)*” from ProofWiki. During the update of AutoMathKG, the two entities were successfully merged, and the reference relationships associated with the “*Mean Value Theorem*” entity were added to the “*Carathéodory’s Theorem (Analysis)*” entity.

**Table 15** The results of the case study on automatic knowledge fusion.

|  Source | ProofWiki  |
| --- | --- |
|  Theorem | Carathéodory’s Theorem (Analysis)  |
|  Contents | Let $I \subseteq \mathbb{R}$. Let $c \in I$ be an interior point of $I$. Let $f : I \rightarrow \mathbb{R}$ be a real function. Then $f$ is differentiable at $c$ if and only if: There exists a real function $\varphi : I \rightarrow \mathbb{R}$ that is continuous at $c$ and satisfies: (1): $\forall x \in I, f(x) - f(c) = \varphi(x)(x - c)$ (2): $\varphi(c) = f'(c)$.  |
|  Source | Textbook  |
|  Theorem | Mean Value Theorem  |
|  Contents | If $f$ is continuous on the closed interval $[a, b]$ and differentiable on the open interval $(a, b)$, then, $f'(c) = \frac{f(b) - f(a)}{b - a}$, for some $c$ in $(a, b)$.  |