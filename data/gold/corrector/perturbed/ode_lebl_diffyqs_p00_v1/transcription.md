![1]()

Figure 1.5: Slope field of $y' = y$ with a graph of a few solutions.

What do you think is the answer? The answer to both questions seems to be yes, does it not? Well, it really is yes most of the time. But there are cases when the answer to either question can be no.

Since the equations we encounter in applications come from real life situations, it seems logical that a solution always exists. It also has to be unique if we believe our universe is deterministic. If the solution does not exist, or if it is not unique, we have probably not devised the correct model. Hence, it is good to know when things go wrong and why.

**Example 1.2.1:** Attempt to solve:

$$y' = \frac{1}{x}, \quad y(0) = 0.$$

Integrate to find the general solution $y = \ln |x| + C$. The solution does not exist at $x = 0$. See Figure 1.6 on the following page. You may say one can see the division by zero a mile away, but the equation may have been written as the seemingly harmless $xy' = 1$.

**Example 1.2.2:** Solve:

$$y' = 2\sqrt[3]{|y|}, \quad y(0) = 0.$$

See Figure 1.7 on the next page. Note that $y = 0$ is a solution. But another solution is the function

$$y(x) = \begin{cases} x^2 & \text{if } x \geq 0, \\ -x^2 & \text{if } x < 0. \end{cases}$$

It is hard to tell by staring at the slope field that the solution is not unique. Is there any hope? Of course there is. We have the following theorem, known as Picard's theorem*.

*Named after the French mathematician Charles Émile Picard (1856–1941).