# Randomized Kaczmarz with Tail Averaging

This repository presents code for the paper _Randomized Kaczmarz with tail averaging_ by [Ethan N. Epperly](https://www.ethanepperly.com), [Gil Goldshlager](https://ggoldshlager.com), and [Robert J. Webber](https://sites.google.com/ucsd.edu/rwebber/).
The [randomized Kaczmarz](https://www.math.ucdavis.edu/~strohmer/papers/2007/kaczmarz.pdf) (RK) method is a widely-studied method for solving a system of equations $\boldsymbol{Ax} = \boldsymbol{b}$, and it proceeds by repeatedly applying the following two steps:

1. _Sample_ an index $i$ with probabilities proportional to the row norms $\mathbb{P} \{ i = j \} = \\|\boldsymbol{A}(j,:)\\|^2 / \\|\boldsymbol{A}\\|_{\rm F}^2$.
2. _Update_ your solution $\boldsymbol{x} \gets \boldsymbol{x} + (b_i - \boldsymbol{A}(i,:) \boldsymbol{x}) \boldsymbol{A}(i,:)^\top / \\|\boldsymbol{A}(i,:)\\|^2$.

When applied to an [inconsistent system of linear equations](https://en.wikipedia.org/wiki/Consistent_and_inconsistent_equations#Linear_systems) (not possessing any solution to $Ax=b$), the iterates of RK do not converge. However, the paper demonstrate that the _tail average_

$$
\overline{\boldsymbol{x}}\_t = \frac{\boldsymbol{x}\_t + \boldsymbol{x}\_{t-1} + \cdots + \boldsymbol{x}\_{t_{\rm b}+1}}{t - t\_{\rm b}}
$$

of the RK iterates converge to the [least-squares solution](https://en.wikipedia.org/wiki/Linear_least_squares#Basic_formulation) $\boldsymbol{x}_\star = \text{argmin}\_{\boldsymbol{x}} \\|\boldsymbol{b} - \boldsymbol{Ax}\\|$.
Here, $t\_{\rm b}$ is a _burn-in_ time; $t\_{\rm b} = t/2$ is a good default value.
This observation results in the _tail-averaged randomized Kaczmarz_ (TARK) method, which outputs the tail average $\overline{\boldsymbol{x}}\_t$.

The TARK method converges at a Monte Carlo rate $\\|\overline{\boldsymbol{x}}\_t - \boldsymbol{x}_\star\\| = O(t^{-1/2})$.
As the paper shows, this rate of convergence is optimal for any row-access method that works by accessing row–entry pairs $(A(i,:),b_i)$.

## Code

This repository contains implementations for TARK, as well as other RK methods.
To call TARK, use the following code segment

```python
x = tark(A, b=b)
```

There are many optional parameters: the number of steps `num_steps`, the burn-in time `burn_in`, and an initialization `x0`.
Other RK methods include plain RK (`rk`), [RK with underrelaxation](https://link.springer.com/chapter/10.1007/978-3-642-28308-6_64) (`rku`), and [RK with averaging](https://arxiv.org/abs/2002.04126) (`rka`).
For these methods, the underrelaxation parameter schedule can be set using the optional argument `under_relax` (default value: `lambda T: 1/np.sqrt(T+1)`) and the number of threads can be set using `num_threads` (default value: 10).
These optional parameters can also be used with TARK, allowing for a mixing and matching of tail averaging, thread averaging, and underrelaxation.

### Ridge regularization

RK methods can also be useful for solving [ridge-regularized least-squares problems](https://en.wikipedia.org/wiki/Ridge_regression#Overview)

$$
\boldsymbol{x}\_{\rm reg} = \text{argmin}\_{\boldsymbol{x}} \\| \boldsymbol{b} - \boldsymbol{Ax} \\|^2 + \lambda \\|\boldsymbol{x}\\|^2.
$$

The paper develops variants RK-RR and TARK-RR of RK and TARK for ridge regression problems by combining stochastic updates to the least-squares loss $\\| \boldsymbol{b} - \boldsymbol{Ax} \\|^2$ with deterministic descent steps for the regularizer $\lambda \\|\boldsymbol{x}\\|^2$.
These methods can be called by setting the `lamb_reg` argument to the desired value of $\lambda$.
(We also allow for parameterizing the ridge parameter using a parameter $\mu$, leading to a ridge parameter $\lambda = (1-\mu) / \mu \cdot \\|A\\|_{\rm F}^2$.
Another set of methods for ridge regression are [dual methods](https://arxiv.org/abs/1507.05844), which can be called in our code by setting the `dual` argument to `True`.

### Experiments

The experiments from the paper can be reproduced using Python scripts `experiments/polynomial_example.py` and `experiments/fitting_rr_example.py`.
Jupyter notebooks for these experiments are in the `notebooks/` folder.
Additional experiments with random problem data are also provided in the `experiments/` folder.

### Active linear regression

When combined with preconditioning and good initialization, TARK leads to a simple algorithm for active linear regression problems with near-optimal rates of convergence. 
(Though this algorithm appears to be a constant factor worse than other methods, such as those based on [leverage score sampling](https://arxiv.org/abs/1104.5557v3).)
Functions are provided in `active_regression.py`, and an example use case is provided in `experiments/active_regression_example.py`.

