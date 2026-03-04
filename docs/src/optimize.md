# Optimization Module

`Durbyn.Optimize` provides a unified optimization API plus direct solver wrappers.

Current solver implementations are wrapper layers around `Optim.jl`, while preserving
Durbyn's public API (`optimize`, `nelder_mead`, `bfgs`, `lbfgsb`, `brent`, option
types, and result shapes).

---

## Overview

| Algorithm | Function | Type | Typical Use |
|-----------|----------|------|-------------|
| Nelder-Mead | `nelder_mead` | Derivative-free | No gradient available |
| BFGS | `bfgs` | Quasi-Newton | Smooth unconstrained problems |
| L-BFGS-B | `lbfgsb` | Bounded quasi-Newton | Box-constrained problems |
| Brent | `brent` | 1D derivative-free | Scalar bounded minimization |

Primary exports:

```julia
using Durbyn.Optimize

optimize, nelder_mead, bfgs, lbfgsb, brent
NelderMeadOptions, BFGSOptions, LBFGSBOptions, BrentOptions
OptimizeResult
numgrad, numgrad!, numgrad_with_cache!, NumericalGradientCache
numerical_hessian
scaler, descaler
```

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

### Key keyword arguments

- `gradient=nothing`: analytic gradient function `gradient(x) -> vector`
- `lower=-Inf`, `upper=Inf`: bounds (`:lbfgsb`, `:brent`)
- `max_iterations`: iteration/evaluation budget
- `param_scale`: parameter scaling vector
- `step_sizes`: finite-difference steps for numerical gradients
- `fn_scale`: objective scaling (negative to maximize)
- `trace`: verbosity (`0` silent)
- `report_interval`: trace cadence
- `reltol`, `abstol`, `gtol`
- `factr`, `pgtol`, `memory_size` (`:lbfgsb`)
- `alpha`, `beta`, `gamma` (`:nelder_mead`)
- `hessian=false`: include numerical Hessian at solution

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

Key options:

- `abstol`, `reltol`, `alpha`, `beta`, `gamma`
- `maxit`, `trace`, `invalid_penalty`
- optional bound projection: `project_to_bounds=true`, `lower`, `upper`

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

---

### `lbfgsb`

```julia
result = lbfgsb(f, g, x0; lower=..., upper=..., options=LBFGSBOptions(...))
```

- Requires gradient function `g(x) -> vector` (for direct `lbfgsb` calls)
- Supports `mask` (masked variables are fixed at initial values)

Returns:

```julia
(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals, message)
```

`fail` conventions:

- `0`: converged
- `1`: iteration limit reached
- `52`: abnormal/invalid state (for example infeasible bounds or non-finite values)

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

---

## Numerical Derivatives

### Gradient

```julia
numgrad(f, x; step_sizes=fill(1e-3, length(x)))
```

For repeated calls, prefer a cache:

```julia
cache = NumericalGradientCache(length(x))
g = numgrad_with_cache!(cache, f, x, step_sizes)
```

Bound-aware finite differences are used internally when needed.

### Hessian

```julia
H = numerical_hessian(f, x)
```

If gradient is available:

```julia
H = numerical_hessian(f, x, grad)
```

---

## Parameter Scaling Helpers

```julia
x_scaled = scaler(x, scale)      # x ./ scale
x_orig   = descaler(x_scaled, scale)  # x_scaled .* scale
```

Use scaling when parameters are on very different magnitudes.

---

## Practical Notes

- Exact function/gradient call counts can differ from older Durbyn releases because
  solver internals are now delegated to `Optim.jl`.
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
- Nocedal, J. and Wright, S. J. (2006). *Numerical Optimization*.
- `Optim.jl` documentation: <https://julianlsolvers.github.io/Optim.jl/stable/>
