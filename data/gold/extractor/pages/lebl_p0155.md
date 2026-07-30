![1]()

![2]()

Figure 4.1: Graphical interpretation of the derivative.

Example 4.1.2: Let $f(x) := x^2$ be defined on the whole real line. Let $c \in \mathbb{R}$ be arbitrary. We find that if $x \neq c$,

$$\frac{x^2 - c^2}{x - c} = \frac{(x + c)(x - c)}{x - c} = (x + c).$$

Therefore,

$$f'(c) = \lim_{x \to c} \frac{x^2 - c^2}{x - c} = \lim_{x \to c} (x + c) = 2c.$$

Example 4.1.3: Let $f(x) := ax + b$ for numbers $a, b \in \mathbb{R}$. Let $c \in \mathbb{R}$ be arbitrary. For $x \neq c$,

$$\frac{f(x) - f(c)}{x - c} = \frac{a(x - c)}{x - c} = a.$$

Therefore,

$$f'(c) = \lim_{x \to c} \frac{f(x) - f(c)}{x - c} = \lim_{x \to c} a = a.$$

In fact, every differentiable function “infinitesimally” behaves like the affine function $ax + b$. You can guess many results and formulas for derivatives if you work them out for affine functions first.

Example 4.1.4: The function $f(x) := \sqrt{x}$ is differentiable for $x > 0$. To see this fact, fix $c > 0$, and suppose $x \neq c$ and $x > 0$. Compute

$$\frac{\sqrt{x} - \sqrt{c}}{x - c} = \frac{\sqrt{x} - \sqrt{c}}{(\sqrt{x} - \sqrt{c})(\sqrt{x} + \sqrt{c})} = \frac{1}{\sqrt{x} + \sqrt{c}}.$$

Therefore,

$$f'(c) = \lim_{x \to c} \frac{\sqrt{x} - \sqrt{c}}{x - c} = \lim_{x \to c} \frac{1}{\sqrt{x} + \sqrt{c}} = \frac{1}{2\sqrt{c}}.$$