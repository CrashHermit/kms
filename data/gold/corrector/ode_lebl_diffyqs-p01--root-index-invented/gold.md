![1]()

Figure 1.6: Slope field of $y' = 1/x$.

![2]()

Figure 1.7: Slope field of $y' = 2\sqrt{|y|}$ with two solutions satisfying $y(0) = 0$.

**Theorem 1.2.1** (Picard's theorem on existence and uniqueness). If $f(x, y)$ is continuous (as a function of two variables) and $\frac{\partial f}{\partial y}$ exists and is continuous near some $(x_0, y_0)$, then a solution to

$$y' = f(x, y), \quad y(x_0) = y_0,$$

exists (at least for $x$ in some small interval) and is unique.

Note that the problems $y' = 1/x$, $y(0) = 0$ and $y' = 2\sqrt{|y|}$, $y(0) = 0$ do not satisfy the hypothesis of the theorem. Even if we can use the theorem, we ought to be careful about this existence business. It is quite possible that the solution only exists for a short while.

**Example 1.2.3:** For some constant $A$, solve:

$$y' = y^2, \quad y(0) = A.$$

We know how to solve this equation. First assume that $A \neq 0$, so $y$ is not equal to zero at least for some $x$ near 0. So $x' = 1/y^2$, so $x = -1/y + C$, so $y = \frac{1}{C-x}$. If $y(0) = A$, then $C = 1/A$ so

$$y = \frac{1}{1/A - x}.$$

If $A = 0$, then $y = 0$ is a solution.

For instance, when $A = 1$ the solution "blows up" at $x = 1$. Hence, the solution does not exist for all $x$ even if the equation itself is nice everywhere—it only exists in the interval $(-\infty, 1)$. The equation $y' = y^2$ certainly looks nice.

For most of this course, we will be interested in equations where existence and uniqueness hold, and in fact hold "globally" unlike for the equation $y' = y^2$.