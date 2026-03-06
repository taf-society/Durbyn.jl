# Optimization Module

`Durbyn.Optimize` provides a unified optimization API plus direct solver wrappers.

The module preserves a stable public API (`optimize`, `nelder_mead`, `bfgs`, `lbfgsb`,
`brent`, option types, and result shapes), while each solver follows its own algorithmic
implementation details.

---

## Overview

| Algorithm | Function | Type | Typical Use |
|-----------|----------|------|-------------|
| Nelder-Mead | `nelder_mead` | Derivative-free | No gradient available |
| BFGS | `bfgs` | Quasi-Newton | Smooth unconstrained problems |
| L-BFGS-B | `lbfgsb` | Bounded quasi-Newton | Box-constrained problems |
| Brent | `brent` | 1D derivative-free | Scalar bounded minimization |
| ITP | `itp` | 1D bracketed root finding | Robust zero finding with sign-change bracket |

Primary exports:

```julia
using Durbyn.Optimize

optimize, nelder_mead, bfgs, lbfgsb, brent, itp
NelderMeadOptions, BFGSOptions, LBFGSBOptions, BrentOptions, ITPOptions
OptimizeResult
numerical_hessian
scaler, descaler
```

---

## Provenance and Implementation Basis

This module is documented and maintained with an equation-first workflow:

- Define each algorithm from published mathematical formulations in cited references.
- Implement those formulations directly in Julia.
- Validate behavior with module tests and explicit solver-level checks.

Solver provenance in this module:

- `nelder_mead`: implemented directly in this repository from the Nelder-Mead equations.
- `bfgs`: implemented directly in this repository from BFGS equations (inverse-Hessian rank-2 updates + strong Wolfe line search).
- `lbfgsb`: implemented directly in this repository from the L-BFGS-B equations (projected gradient, limited-memory updates, bound-feasible line search).
- `itp`: implemented directly in this repository from the ITP root-finding equations (interpolate-truncate-project).
- `brent`: implemented directly in this repository from Brent's bounded golden-section/parabolic interpolation equations.
- This page includes the governing equations and references so the implementation basis is transparent.

---

## Unified Interface: `optimize`

```julia
optimize(fn, x0, method=:nelder_mead; kwargs...)
```

### Supported methods

- `:nelder_mead`
- `:bfgs`
- `:lbfgsb`
- `:brent` (requires `length(x0) == 1`, finite `lower`, finite `upper`)
- `:itp` (requires `length(x0) == 1`, finite `lower`, finite `upper`, and a sign-change bracket)

### Key keyword arguments

- `gradient=nothing`: analytic gradient function `gradient(x) -> vector`
- `lower=-Inf`, `upper=Inf`: bounds (`:lbfgsb`, `:brent`, `:itp`)
- `max_iterations`: iteration/evaluation budget
- `param_scale`: parameter scaling vector
- `step_sizes`: finite-difference steps used for numerical Hessian estimation
- `fn_scale`: objective scaling (negative to maximize)
- `trace`: verbosity (`0` silent)
- `report_interval`: trace cadence
- `reltol`, `abstol`, `gtol`
- `factr`, `pgtol`, `memory_size` (`:lbfgsb`)
- `alpha`, `beta`, `gamma` (`:nelder_mead`)
- `hessian=false`: include numerical Hessian at solution
  - Not supported for `method = :itp`.

### Return value

`optimize` returns `OptimizeResult` with fields:

- `minimizer::Vector{Float64}`
- `minimum::Float64`
- `converged::Bool`
- `iterations::Int`
- `f_calls::Int`
- `g_calls::Int`
- `message::String`

If `hessian=true`, returns a named tuple with the same fields plus `hessian`.

### Example

```julia
using Durbyn.Optimize

rosenbrock(x) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2
rosenbrock_grad(x) = [
    -400.0 * x[1] * (x[2] - x[1]^2) - 2.0 * (1.0 - x[1]),
    200.0 * (x[2] - x[1]^2),
]

r1 = optimize(rosenbrock, [-1.2, 1.0], :nelder_mead)
r2 = optimize(rosenbrock, [-1.2, 1.0], :bfgs; gradient=rosenbrock_grad)
r3 = optimize(rosenbrock, [0.5, 0.5], :lbfgsb; lower=[0.0, 0.0], upper=[2.0, 2.0])
r4 = optimize(x -> (x[1] - 2.0)^2, [0.0], :brent; lower=-10.0, upper=10.0)
r5 = optimize(x -> x[1]^2 - 2.0, [1.0], :itp; lower=0.0, upper=2.0)
```

---

## Direct Solver Wrappers

These are useful when you want solver-specific controls or result tuples.

### `nelder_mead`

```julia
result = nelder_mead(f, x0, NelderMeadOptions(...))
```

Returns:

```julia
(x_opt, f_opt, fncount, fail)
```

- `fail == 0`: converged
- `fail == 1`: stopped before convergence
- `fail == 10`: shrink step failed to reduce simplex diameter (stagnation guard)

Key options:

- `abstol`, `reltol`, `alpha`, `beta`, `gamma`
- `maxit`, `trace`, `invalid_penalty`
- optional bound projection: `project_to_bounds=true`, `lower`, `upper`

Implemented equations (Nelder & Mead, 1965):

```math
\bar{P} = \frac{1}{n}\sum_{i \ne h} P_i,\qquad
P^* = (1 + \alpha)\bar{P} - \alpha P_h
```
```math
P^{**}_{\text{expand}} = \gamma P^* + (1-\gamma)\bar{P}
```
```math
P^{**}_{\text{contract}} = \beta P_h + (1-\beta)\bar{P}
```
```math
P_i \leftarrow \frac{P_i + P_l}{2}\quad\text{(shrink)}
```

Contraction acceptance follows:

```math
y_{\text{contract}} \le \min(y_h, y^*)
```

Stopping criterion uses simplex function-value spread:

```math
\sigma = \sqrt{\frac{1}{n}\sum_{i=0}^{n}(y_i - \bar{y})^2}
```

Default coefficients follow the original recommendations:
`alpha = 1`, `beta = 0.5`, `gamma = 2`.

Initial simplex is built in axial form from `x0`:
`P_i = P_0 + \delta_i e_i` for `i=1,\dots,n`.

---

### `bfgs`

```julia
result = bfgs(f, g!, x0; options=BFGSOptions(...), mask=trues(length(x0)))
```

`g!` mutates its first argument:

```julia
g!(grad, x) = (grad .= 2.0 .* x; nothing)
```

You can also pass `g! = nothing` to use numerical gradients.

Returns:

```julia
(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals)
```

Mathematical basis (BFGS update):

```math
p_k = -H_k \nabla f(x_k),\qquad x_{k+1} = x_k + \alpha_k p_k
```
```math
s_k = x_{k+1} - x_k,\quad y_k = \nabla f(x_{k+1}) - \nabla f(x_k),\quad
\rho_k = \frac{1}{y_k^\top s_k}
```
```math
H_{k+1} = (I - \rho_k s_k y_k^\top)H_k(I - \rho_k y_k s_k^\top) + \rho_k s_k s_k^\top
```

Strong Wolfe conditions used in line search:

```math
f(x_k+\alpha p_k) \le f(x_k) + c_1 \alpha \nabla f(x_k)^\top p_k,\qquad c_1=10^{-4}
```
```math
\left|\nabla f(x_k+\alpha p_k)^\top p_k\right| \le c_2 \left|\nabla f(x_k)^\top p_k\right|,\qquad c_2=0.9
```

Curvature acceptance for inverse-Hessian update:

```math
y_k^\top s_k > 0
```

Implementation note:

- This module implements BFGS directly (search direction `p=-Hg`, strong Wolfe bracketing/zoom, and in-place symmetric inverse-Hessian updates).
- If `g! = nothing`, gradients are computed by central finite differences.
- `mask` keeps non-active coordinates fixed at their initial values.

---

### `lbfgsb`

```julia
result = lbfgsb(f, g, x0; lower=..., upper=..., options=LBFGSBOptions(...))
```

- `g` can be `nothing` (internal bound-aware finite differences) or a gradient function `g(x) -> vector`
- Supports `mask` (masked variables are fixed at initial values)

Returns:

```julia
(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals, message)
```

`fail` conventions:

- `0`: converged
- `1`: iteration limit reached
- `52`: abnormal/invalid state (for example infeasible bounds or non-finite values)

Mathematical basis (box-constrained optimization):

```math
\min_{x \in \mathbb{R}^n} f(x)\quad\text{subject to}\quad l \le x \le u
```

Projected-gradient stationarity condition (used by L-BFGS-B style methods):

```math
\|P_{[l,u]}(x - \nabla f(x)) - x\|_\infty \le \varepsilon
```

Limited-memory correction pairs:

```math
s_k = x_{k+1} - x_k,\qquad y_k = \nabla f(x_{k+1}) - \nabla f(x_k)
```
```math
\text{accept update if}\quad s_k^\top y_k > \epsilon_{\text{mach}}\|y_k\|^2
```

Implementation note:

- This module implements the L-BFGS-B iteration directly (projected-gradient stop test, two-loop direction, bound-feasible line search, and limited-memory updates).
- For bounded `optimize(..., :lbfgsb)` calls, a deterministic interior restart may be attempted to avoid poor boundary stationary points; the better solution is kept.

---

### `brent`

```julia
result = brent(f, lower, upper; options=BrentOptions(...))
```

Returns:

```julia
(x_opt, f_opt, n_iter, fail, fn_evals)
```

`f` is scalar style: `f(x::Float64)`.

Mathematical basis (Brent):

```math
x_{k+1} =
\begin{cases}
x_k + \frac{p_k}{q_k}, & \text{if parabolic step is acceptable}\\
x_k + c\,(b_k-a_k), & \text{otherwise (golden-section fallback)}
\end{cases}
```
```math
c = \frac{3-\sqrt{5}}{2}
```

Implementation note:

- This module standardizes tolerance handling, traces, non-finite guards, and return fields.
- The iteration engine follows Brent's Chapter 5 structure directly with
  `(x_best, x_second, x_prev)` point tracking and guarded parabolic steps.

---

### `itp`

```julia
result = itp(f, a, b; options=ITPOptions(...))
```

Returns:

```julia
(x_root, f_root, n_iter, fail, fn_evals)
```

`fail` conventions:

- `0`: converged (`f(x_root) == 0` or bracket width `<= 2*tol`)
- `1`: max iterations reached before tolerance

ITP iteration equations:

```math
x_{1/2} = \frac{a+b}{2},\qquad
x_f = \frac{b f(a) - a f(b)}{f(a)-f(b)}
```

```math
\sigma = \operatorname{sign}(x_{1/2}-x_f),\qquad
\delta = \kappa_1 |b-a|^{\kappa_2}
```

```math
x_t =
\begin{cases}
x_f + \sigma\delta, & \delta \le |x_{1/2}-x_f|\\
x_{1/2}, & \text{otherwise}
\end{cases}
```

```math
r = \varepsilon\,2^{\,n_{1/2}+n_0-j} - \frac{b-a}{2},\qquad
x_{\text{ITP}} =
\begin{cases}
x_t, & |x_t-x_{1/2}| \le r\\
x_{1/2} - \sigma r, & \text{otherwise}
\end{cases}
```

Implementation note:

- This module implements ITP directly (interpolate-truncate-project) and preserves bracket guarantees.
- Default `k1` follows the paper scaling rule `k1 = 0.2/(b0-a0)` when not provided.

---

## Numerical Derivatives

### Hessian

```julia
H = numerical_hessian(f, x; gradient=nothing)
```

`numerical_hessian` uses Section 8.1 finite differences directly:

When `gradient` is provided (forward-difference Hessian columns, Eq. 8.7):

```math
\nabla^2 f(x)e_j \approx \frac{\nabla f(x + \varepsilon_j e_j) - \nabla f(x)}{\varepsilon_j}
```

When `gradient` is not provided (second-order finite differences of `f`, Eq. 8.9):

```math
\frac{\partial^2 f}{\partial x_i^2}(x) \approx \frac{f(x+\varepsilon_i e_i)-2f(x)+f(x-\varepsilon_i e_i)}{\varepsilon_i^2}
```

```math
\frac{\partial^2 f}{\partial x_i \partial x_j}(x) \approx
\frac{f(x+\varepsilon_i e_i+\varepsilon_j e_j)-f(x+\varepsilon_i e_i)-f(x+\varepsilon_j e_j)+f(x)}
{\varepsilon_i\varepsilon_j}
```

Symmetry is enforced numerically by using `0.5 * (H + H')` when the gradient-differencing path is used.

---

## Parameter Scaling Helpers

```julia
x_scaled = scaler(x, scale)      # x ./ scale
x_orig   = descaler(x_scaled, scale)  # x_scaled .* scale
```

Use scaling when parameters are on very different magnitudes.

---

## Practical Notes

- Exact function/gradient call counts may differ across methods because each algorithm
  uses method-specific line searches, simplex operations, or bound handling.
- The `optimize` API is the recommended entry point unless you explicitly need a
  direct solver tuple return.

---

## References

- Nelder, J. A. and Mead, R. (1965). *A simplex method for function minimization*.
- Lagarias, J. C. et al. (1998). *Convergence properties of the Nelder-Mead simplex method in low dimensions*.
- Gao, F. and Han, L. (2012). *Implementing the Nelder-Mead simplex algorithm with adaptive parameters*.
- Broyden, Fletcher, Goldfarb, and Shanno original quasi-Newton papers (1970).
- Byrd, Lu, Nocedal, and Zhu (1995). *A limited memory algorithm for bound constrained optimization*.
- Zhu, Byrd, Lu, and Nocedal (1997). *Algorithm 778: L-BFGS-B*.
- Brent, R. P. (1973/2002). *Algorithms for Minimization Without Derivatives*.
- Oliveira, I. F. D. & Takahashi, R. H. C. (2021). *An Enhancement of the Bisection Method Average Performance Preserving Minmax Optimality*. ACM TOMS, 47(1).
- Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*.
