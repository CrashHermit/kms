![](images/seg0000_img001.png)

Figure 1.5: Slope field of $y' = -y$ with a graph of a few solutions.

What do you think is the answer? The answer to both questions seems to be yes, does it not? Well, it really is yes most of the time. But there are cases when the answer to either question can be no.

Since the equations we encounter in applications come from real life situations, it seems logical that a solution always exists. It also has to be unique if we believe our universe is deterministic. If the solution does not exist, or if it is not unique, we have probably not devised the correct model. Hence, it is good to know when things go wrong and why.

**Example 1.2.1:** Attempt to solve:

$$y' = \frac{1}{x}, \quad y(0) = 0.$$

Integrate to find the general solution $y = \ln |x| + C$. The solution does not exist at $x = 0$. See Figure 1.6 on the following page. You may say one can see the division by zero a mile away, but the equation may have been written as the seemingly harmless $xy' = 1$.

**Example 1.2.2:** Solve:

$$y' = 2\sqrt{|y|}, \quad y(0) = 0.$$

See Figure 1.7 on the next page. Note that $y = 0$ is a solution. But another solution is the function

$$y(x) = \begin{cases} x^2 & \text{if } x \geq 0, \\ -x^2 & \text{if } x < 0. \end{cases}$$

It is hard to tell by staring at the slope field that the solution is not unique. Is there any hope? Of course there is. We have the following theorem, known as Picard's theorem*.

*Named after the French mathematician Charles Émile Picard (1856–1941).

Figure 1.6: Slope field of $y' = 1/x$.

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

### 1.2.3 Exercises

1.2.1 Sketch the slope field for $y' = e^{x-y}$. How do the solutions behave as $x$ grows? Can you guess a particular solution by looking at the slope field?

1.2.2 Sketch the slope field for $y' = x^2$.

1.2.3 Sketch the slope field for $y' = y^2$.

1.2.4 Is it possible to solve the equation $y' = \frac{xy}{\cos x}$ for $y(0) = 1$? Justify.

1.2.5 Is it possible to solve the equation $y' = y\sqrt{|x|}$ for $y(0) = 0$? Is the solution unique? Justify.

1.2.6 Match equations $y' = 1 - x$, $y' = x - 2y$, $y' = x(1 - y)$ to slope fields. Justify.

a) ![a](image_a.png) b) ![b](image_b.png) c) ![c](image_c.png)

1.2.7 Take $y' = f(x, y)$, $y(0) = 0$, where $f(x, y) > 1$ for all $x$ and $y$. If the solution exists for all $x$, can you say what happens to $y(x)$ as $x$ goes to positive infinity? Explain.

1.2.8 Take $(y - x)y' = 0$, $y(0) = 0$.

- a) Find two distinct solutions.
- b) Explain why this does not violate Picard's theorem.

1.2.9 Suppose $y' = f(x, y)$. What will the slope field look like, explain and sketch an example, if you know the following about $f(x, y)$:

- a) $f$ does not depend on $y$.
- b) $f$ does not depend on $x$.
- c) $f(t, t) = 0$ for any number $t$.
- d) $f(x, 0) = 0$ and $f(x, 1) = 1$ for all $x$.

1.2.10 Find a solution to $y' = |y|$, $y(0) = 0$. Does Picard's theorem apply?

1.2.11 Take an equation $y' = (y - 2x)g(x, y) + 2$ for some function $g(x, y)$. Can you solve the problem for the initial condition $y(0) = 0$, and if so what is the solution?

**Exercise 1.2.12 (challenging):**

Suppose $y' = f(x, y)$ is such that $f(x, 1) = 0$ for every $x$, $f$ is continuous and $\frac{\partial f}{\partial y}$ exists and is continuous for every $x$ and $y$.

a) Guess a solution given the initial condition $y(0) = 1$.
b) Can graphs of two solutions of the equation for different initial conditions ever intersect?
c) Given $y(0) = 0$, what can you say about the solution. In particular, can $y(x) > 1$ for any $x$? Can $y(x) = 1$ for any $x$? Why or why not?

**Exercise 1.2.101:**

Sketch the slope field of $y' = y^3$. Can you visually find the solution that satisfies $y(0) = 0$?

**Exercise 1.2.102:**

Is it possible to solve $y' = xy$ for $y(0) = 0$? Is the solution unique?

**Exercise 1.2.103:**

Is it possible to solve $y' = \frac{x}{x^2 - 1}$ for $y(1) = 0$?

**Exercise 1.2.104:**

Match equations $y' = \sin x$, $y' = \cos y$, $y' = y \cos(x)$ to slope fields. Justify.

![](images/seg0003_img001.png)

![2]()

![3]()

**Exercise 1.2.105 (tricky):**

Suppose

$$f(y) = \begin{cases} 0 & \text{if } y > 0, \\ 1 & \text{if } y \leq 0. \end{cases}$$

Does $y' = f(y)$, $y(0) = 0$ have a continuously differentiable solution? Does Picard apply? Why, or why not?

**Exercise 1.2.106:**

Consider an equation of the form $y' = f(x)$ for some continuous function $f$, and an initial condition $y(x_0) = y_0$. Does a solution exist for all $x$? Why or why not?
