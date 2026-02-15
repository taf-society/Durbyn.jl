# Optimization Module

The Optimize module provides numerical optimization algorithms for minimizing objective functions. These solvers are used internally throughout Durbyn.jl for model fitting (e.g., maximum likelihood estimation in ETS, ARIMA, and BATS models) and are also available for general-purpose optimization tasks.

---

## Overview

The module implements four optimization algorithms accessible through a unified interface:

| Algorithm | Function | Type | Use Case |
|-----------|----------|------|----------|
| **Nelder-Mead** | `nelder_mead` | Derivative-free | General purpose, no gradient needed |
| **BFGS** | `bfgs` | Quasi-Newton | Fast convergence with gradients |
| **L-BFGS-B** | `lbfgsb` | Bounded quasi-Newton | Box-constrained optimization |
| **Brent** | `brent` | 1D derivative-free | Scalar optimization |

### Exported Functions and Types

```julia
# Main optimization functions
export optimize, nelder_mead, bfgs, lbfgsb, brent

# Options types
export NelderMeadOptions, BFGSOptions, LBFGSBOptions, BrentOptions

# Supporting functions
export numgrad, numgrad!, numgrad_with_cache!, NumericalGradientCache
export numerical_hessian, bfgs_hessian_update!, BFGSWorkspace
export scaler, descaler
```

---

## Unified Interface (`optimize`)

The `optimize` function provides a unified interface for all solvers, allowing easy switching between methods.

```julia
optimize(x0, fn; grad=nothing, method="Nelder-Mead", lower=-Inf, upper=Inf,
         control=Dict(), hessian=false, kwargs...)
```

### Arguments

- `x0::Vector{Float64}`: Initial parameter vector
- `fn::Function`: Objective function to minimize, called as `fn(x; kwargs...)`

### Keyword Arguments

- `grad::Union{Function,Nothing}`: Gradient function, called as `grad(x; kwargs...)`. If `nothing`, numerical gradients are computed automatically.
- `method::String`: Optimization method:
  - `"Nelder-Mead"` (default) - Derivative-free simplex
  - `"BFGS"` - Quasi-Newton with line search
  - `"L-BFGS-B"` - Limited-memory BFGS with box constraints
  - `"Brent"` - 1D optimization (scalar `x0` only)
- `lower`, `upper`: Bounds for L-BFGS-B and Brent methods
- `control::Dict`: Control parameters (see below)
- `hessian::Bool`: If `true`, compute Hessian at solution

### Control Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trace` | 0 | Verbosity level (0=silent, >0=verbose) |
| `fnscale` | 1.0 | Function scaling factor |
| `parscale` | ones(n) | Parameter scaling vector |
| `ndeps` | 1e-3 | Step sizes for numerical derivatives |
| `maxit` | 500/100 | Maximum iterations (500 for NM, 100 for others) |
| `abstol` | -Inf | Absolute convergence tolerance |
| `reltol` | sqrt(eps) | Relative convergence tolerance |
| `gtol` | 0.0 | Gradient norm tolerance (BFGS only) |
| `alpha` | 1.0 | Nelder-Mead reflection coefficient |
| `beta` | 0.5 | Nelder-Mead contraction coefficient |
| `gamma` | 2.0 | Nelder-Mead expansion coefficient |
| `REPORT` | 10 | Reporting frequency for BFGS |
| `lmm` | 5 | L-BFGS-B memory parameter |
| `factr` | 1e7 | L-BFGS-B tolerance factor |
| `pgtol` | 0.0 | L-BFGS-B projected gradient tolerance |

### Returns

Named tuple with fields:
- `par::Vector{Float64}`: Optimal parameters
- `value::Float64`: Function value at optimum
- `counts::NamedTuple`: `(function_=n, gradient=m)` evaluation counts
- `convergence::Int`: Status code (0=success, 1=maxit reached)
- `message`: Convergence message (method-dependent)
- `hessian`: Hessian matrix at solution (if requested)

### Examples

```julia
using Durbyn.Optimize

# Define Rosenbrock function
rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

# Analytical gradient
rosenbrock_grad(x) = [
    -400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]),
    200*(x[2]-x[1]^2)
]

# Nelder-Mead (no gradient needed)
result = optimize([-1.2, 1.0], rosenbrock)
println("Optimal: $(result.par), Value: $(result.value)")

# BFGS with analytical gradient
result = optimize([-1.2, 1.0], rosenbrock; grad=rosenbrock_grad, method="BFGS")

# BFGS with numerical gradient (automatic)
result = optimize([-1.2, 1.0], rosenbrock; method="BFGS")

# L-BFGS-B with box constraints
result = optimize([0.5, 0.5], rosenbrock; method="L-BFGS-B",
                  lower=[0.0, 0.0], upper=[2.0, 2.0])

# With control parameters
result = optimize([-1.2, 1.0], rosenbrock; method="BFGS",
                  control=Dict("trace" => 1, "maxit" => 500, "gtol" => 1e-6))

# Request Hessian at solution
result = optimize([-1.2, 1.0], rosenbrock; grad=rosenbrock_grad,
                  method="BFGS", hessian=true)
println("Hessian:\n$(result.hessian)")

# 1D optimization with Brent's method
f1d(x) = (x[1] - 2)^2
result = optimize([0.0], f1d; method="Brent", lower=-5.0, upper=5.0)
```

---

## Nelder-Mead Simplex (`nelder_mead`)

A derivative-free optimization method that searches for a minimum using a simplex -- a polytope with ``n+1`` vertices in ``n`` dimensions. The algorithm adaptively reshapes the simplex through reflection, expansion, contraction, and shrink operations.

**Reference:** Nelder, J. A. and Mead, R. (1965). *A simplex method for function minimization*. Computer Journal, 7, 308--313. Implementation follows the compact formulation in Nash (1990).

### Algorithm

The algorithm performs four operations on the simplex:
1. **Reflection**: Reflect the worst point through the centroid
2. **Expansion**: If reflection improves the best, expand further
3. **Contraction**: If reflection fails, contract toward the best
4. **Shrink**: If still no improvement, shrink simplex about the best point

### Mathematical Details

Given simplex vertices ``\{x_1, ..., x_{n+1}\}`` ordered by function values ``f(x_1) \leq ... \leq f(x_{n+1})``:

- **Centroid**: ``\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i``
- **Reflection**: ``x_r = \bar{x} + \alpha(\bar{x} - x_{n+1})``
- **Expansion**: ``x_e = \bar{x} + \gamma(x_r - \bar{x})``
- **Contraction**: ``x_c = \bar{x} + \beta(x_{n+1} - \bar{x})``

Default coefficients: ``\alpha = 1.0``, ``\beta = 0.5``, ``\gamma = 2.0``

### Usage

```julia
nelder_mead(f, x0, options::NelderMeadOptions)
```

**Options:**
```julia
NelderMeadOptions(;
    abstol = -Inf,           # Absolute tolerance on function value
    reltol = sqrt(eps()),    # Relative tolerance
    alpha = 1.0,             # Reflection coefficient
    beta = 0.5,              # Contraction coefficient
    gamma = 2.0,             # Expansion coefficient
    trace = false,           # Print diagnostics
    maxit = 500,             # Maximum function evaluations
    invalid_penalty = 1e35,  # Penalty for non-finite values
    project_to_bounds = false,
    lower = nothing,
    upper = nothing,
    init_step_cap = nothing
)
```

**Returns:** Named tuple `(x_opt, f_opt, fncount, fail)`
- `fail=0`: Converged successfully
- `fail=1`: Exceeded maximum iterations
- `fail=10`: Degenerate simplex (shrink failure)

### Example

```julia
using Durbyn.Optimize

f(x) = (x[1] - 1)^2 + (x[2] - 2)^2

opts = NelderMeadOptions(trace=true, maxit=1000)
result = nelder_mead(f, [0.0, 0.0], opts)
println("Optimum: $(result.x_opt)")
```

---

## BFGS Quasi-Newton (`bfgs`)

The Broyden--Fletcher--Goldfarb--Shanno (BFGS) algorithm is a quasi-Newton method that iteratively builds an approximation to the inverse Hessian matrix using gradient information. It achieves superlinear convergence on smooth problems without requiring explicit second-derivative computation.

**Reference:** Nocedal, J. and Wright, S. J. (1999). *Numerical Optimization*. Springer. See also the original papers by Broyden (1970), Fletcher (1970), Goldfarb (1970), and Shanno (1970).

### Algorithm

BFGS iteratively updates the inverse Hessian approximation ``B_k`` using the formula:

```math
B_{k+1} = B_k + \frac{(1 + c^T B_k c / D_1) t t^T}{D_1} - \frac{t (B_k c)^T + (B_k c) t^T}{D_1}
```

where:
- ``t = x_{k+1} - x_k`` (parameter step)
- ``c = \nabla f_{k+1} - \nabla f_k`` (gradient difference)
- ``D_1 = t^T c``

### Features

- **Armijo line search** with backtracking
- **Periodic Hessian restarts** every 2n gradient evaluations
- **Parameter masking** to freeze variables
- **Automatic numerical gradients** if analytical gradient not provided
- **Gradient norm convergence** criterion

### Usage

```julia
bfgs(f, g, x0; mask=nothing, options=BFGSOptions(), step_sizes=1e-3*ones(n),
     numgrad_cache=nothing, extra=nothing)
```

**Function Signatures:**
- `f(n, x, extra)` - Objective function
- `g(n, x, grad, extra)` - Gradient function (modifies `grad` in-place), or `nothing`

**Options:**
```julia
BFGSOptions(;
    abstol = -Inf,           # Absolute tolerance
    reltol = sqrt(eps()),    # Relative tolerance
    gtol = 0.0,              # Gradient norm tolerance
    trace = false,           # Print progress
    maxit = 100,             # Maximum iterations
    report_interval = 10     # Reporting frequency
)
```

**Gradient Norm Convergence (`gtol`):**

When `gtol > 0`, convergence is declared if:
```math
\|\nabla f(x)\| < \text{gtol} \times \max(1, |f(x)|)
```

This provides a first-order optimality condition. Recommended values: `1e-5` to `1e-8`.

**Returns:** Named tuple `(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals)`

### Example

```julia
using Durbyn.Optimize

# Internal function signature: f(n, x, extra)
rosenbrock_internal(n, x, extra) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

# Gradient modifies grad in-place
function rosenbrock_grad_internal(n, x, grad, extra)
    grad[1] = -400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1])
    grad[2] = 200*(x[2]-x[1]^2)
    return nothing
end

opts = BFGSOptions(trace=true, gtol=1e-6)
result = bfgs(rosenbrock_internal, rosenbrock_grad_internal, [-1.2, 1.0]; options=opts)

# With numerical gradients (g=nothing)
result = bfgs(rosenbrock_internal, nothing, [-1.2, 1.0]; options=opts)
```

---

## L-BFGS-B Bounded Optimization (`lbfgsb`)

A limited-memory variant of BFGS that supports box constraints on variables. Instead of storing the full inverse Hessian approximation, it retains only the last ``memory_size`` iterations of gradient information, making it memory-efficient for large-scale problems.

**Reference:** Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM Journal on Scientific Computing, 16, 1190--1208. See also Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997). *Algorithm 778: L-BFGS-B*.

### Features

- **Box constraints**: Lower and upper bounds on variables
- **Limited memory**: Stores only `memory_size` gradient steps (default: 10)
- **Wolfe line search** with zoom refinement
- **Projected gradient** for convergence checking
- **Parameter masking** by setting `lower[i] = upper[i] = x0[i]`

### Usage

```julia
lbfgsb(f, g, x0; mask=nothing, lower=-Inf, upper=Inf, options=LBFGSBOptions())
```

**Options:**
```julia
LBFGSBOptions(;
    memory_size = 10,    # Memory size (number of stored iterations)
    ftol_factor = 1e7,   # Tolerance factor: f_tol = ftol_factor * eps()
    pg_tol = 1e-5,       # Projected gradient infinity-norm tolerance
    maxit = 1000,        # Maximum iterations
    print_level = 0      # Print level (0=silent, >0=verbose)
)
```

**Convergence Criterion:**

The algorithm converges when:
```math
\|\text{proj}(\nabla f(x))\|_\infty < \text{pg\_tol}
```

where the projected gradient accounts for variables at their bounds.

**Returns:** Named tuple `(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals)`

### Example

```julia
using Durbyn.Optimize

rosenbrock(n, x, extra) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

opts = LBFGSBOptions(memory_size=5, pg_tol=1e-6, print_level=1)
result = lbfgsb(rosenbrock, nothing, [0.5, 0.5];
                lower=[0.0, 0.0], upper=[2.0, 2.0], options=opts)

println("Bounded optimum: $(result.x_opt)")
```

---

## Brent's 1D Method (`brent`)

A derivative-free algorithm for one-dimensional optimization that combines golden section search with parabolic interpolation. Golden section steps provide guaranteed progress, while parabolic interpolation accelerates convergence near the minimum.

**Reference:** Brent, R. P. (1973). *Algorithms for Minimization Without Derivatives*. Prentice-Hall. Implementation follows the compact formulation in Nash (1990).

### Convergence

For unimodal functions with positive second derivative at the minimum:
- **Superlinear convergence** with order approximately 1.324
- Error bound: ``< 3\epsilon|x_{\min}| + \text{tol}``

### Usage

```julia
brent(f, lower, upper; options=BrentOptions())
```

**Options:**
```julia
BrentOptions(;
    tol = 1.5e-8,    # Interval tolerance
    trace = false,    # Print diagnostics
    maxit = 1000      # Maximum iterations
)
```

**Returns:** Named tuple `(x_opt, f_opt, n_iter, fail, fn_evals)`

### Example

```julia
using Durbyn.Optimize

f(x) = (x - 3.5)^2 + 2*sin(x)

opts = BrentOptions(tol=1e-10, trace=true)
result = brent(f, 0.0, 10.0; options=opts)

println("Minimum at x = $(result.x_opt), f(x) = $(result.f_opt)")
```

---

## Numerical Gradient Computation

The module provides efficient numerical gradient computation using central finite differences.

### `numgrad`

```julia
numgrad(f, n, x, extra, step_sizes; usebounds=false, lower=nothing, upper=nothing)
```

Computes the gradient using central differences:
```math
\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon_i e_i) - f(x - \epsilon_i e_i)}{2\epsilon_i}
```

### `numgrad_with_cache!`

For repeated gradient evaluations, use a pre-allocated cache:

```julia
cache = NumericalGradientCache(n)
numgrad_with_cache!(cache, f, n, x, extra, step_sizes)
```

This eliminates memory allocations during iterative optimization.

### Example

```julia
using Durbyn.Optimize

f(n, x, extra) = x[1]^2 + x[2]^2
x = [1.0, 2.0]
step_sizes = [1e-6, 1e-6]

# Single evaluation
g = numgrad(f, 2, x, nothing, step_sizes)

# With cache for repeated evaluations
cache = NumericalGradientCache(2)
for i in 1:1000
    numgrad_with_cache!(cache, f, 2, x, nothing, step_sizes)
    # cache.gradient contains the gradient
end
```

---

## Hessian Computation

The `numerical_hessian` function computes the Hessian matrix at a given point using finite differences.

**Reference:** Nocedal, J. and Wright, S. J. (1999). *Numerical Optimization*. Springer. Chapter 8, finite difference formulas.

```julia
numerical_hessian(fn, x, grad=nothing; fnscale=1.0, parscale=ones(n), step_sizes=1e-3*ones(n))
```

**Method:**
- If `grad=nothing`: Uses second-order finite differences of the objective function
- If `grad` provided: Computes Hessian from gradient via finite differences

**Returns:** Symmetric Hessian matrix (n x n)

### Example

```julia
using Durbyn.Optimize

rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

# At the optimum
x_opt = [1.0, 1.0]
H = numerical_hessian(rosenbrock, x_opt)
println("Hessian at optimum:\n$H")

# Eigenvalues (should be positive for minimum)
using LinearAlgebra
eigvals(H)
```

---

## Parameter Scaling

Scaling parameters can improve numerical conditioning when parameters have different magnitudes.

### `scaler` and `descaler`

```julia
x_scaled = scaler(x, scale)       # x_scaled = x ./ scale
x_original = descaler(x_scaled, scale)  # x_original = x_scaled .* scale
```

### Example

```julia
# Parameters with different scales
par = [1e-6, 1e6]
scale = [1e-6, 1e6]

# Scale for optimization
par_scaled = scaler(par, scale)  # [1.0, 1.0]

# After optimization, recover original scale
par_opt = descaler(par_scaled_opt, scale)
```

---

## Practical Guidelines

### Choosing an Optimization Method

| Scenario | Recommended Method |
|----------|-------------------|
| No gradient available | Nelder-Mead |
| Smooth function, gradient available | BFGS |
| Box constraints needed | L-BFGS-B |
| Large-scale problem (many parameters) | L-BFGS-B |
| 1D optimization | Brent |
| Non-smooth or noisy function | Nelder-Mead |

### Tips for Better Convergence

1. **Parameter scaling**: Scale parameters to similar magnitudes
   ```julia
   control = Dict("parscale" => [1e-3, 1e3])
   ```

2. **Good starting point**: Provide reasonable initial values

3. **Check gradients**: Verify analytical gradients match numerical
   ```julia
   grad_analytical = my_gradient(x)
   grad_numerical = numgrad(f, n, x, nothing, 1e-6*ones(n))
   @assert isapprox(grad_analytical, grad_numerical, rtol=1e-4)
   ```

4. **Adjust tolerances**: Tighten tolerances for more precision
   ```julia
   control = Dict("reltol" => 1e-10, "gtol" => 1e-8)
   ```

5. **Monitor convergence**: Use `trace` to diagnose issues
   ```julia
   control = Dict("trace" => 1)
   ```

---

## Complete Example: Maximum Likelihood Estimation

```julia
using Durbyn.Optimize
using Distributions

# Generate data from Normal(mu=5, sigma=2)
data = rand(Normal(5.0, 2.0), 100)

# Negative log-likelihood (to minimize)
function neg_loglik(params)
    mu, log_sigma = params
    sigma = exp(log_sigma)  # Ensure sigma > 0
    n = length(data)
    ll = -n/2 * log(2*pi) - n*log_sigma - sum((data .- mu).^2) / (2*sigma^2)
    return -ll  # Return negative for minimization
end

# Analytical gradient
function neg_loglik_grad(params)
    mu, log_sigma = params
    sigma = exp(log_sigma)
    n = length(data)

    d_mu = sum(data .- mu) / sigma^2
    d_log_sigma = n - sum((data .- mu).^2) / sigma^2

    return [-d_mu, -d_log_sigma]
end

# Optimize with BFGS
result = optimize([0.0, 0.0], neg_loglik;
                  grad=neg_loglik_grad,
                  method="BFGS",
                  hessian=true)

# Extract estimates
mu_hat = result.par[1]
sigma_hat = exp(result.par[2])

println("Estimated mu: $mu_hat (true: 5.0)")
println("Estimated sigma: $sigma_hat (true: 2.0)")

# Standard errors from Hessian
using LinearAlgebra
se = sqrt.(diag(inv(result.hessian)))
println("SE(mu): $(se[1])")
println("SE(log_sigma): $(se[2])")
```

---

## References

- Nelder, J. A. and Mead, R. (1965). *A simplex method for function minimization*. Computer Journal, 7, 308--313.
- Broyden, C. G. (1970). *The convergence of a class of double-rank minimization algorithms*. Journal of the Institute of Mathematics and Its Applications, 6, 76--90.
- Fletcher, R. (1970). *A new approach to variable metric algorithms*. Computer Journal, 13, 317--322.
- Goldfarb, D. (1970). *A family of variable metric methods derived by variational means*. Mathematics of Computation, 24, 23--26.
- Shanno, D. F. (1970). *Conditioning of quasi-Newton methods for function minimization*. Mathematics of Computation, 24, 647--656.
- Byrd, R. H., Lu, P., Nocedal, J., and Zhu, C. (1995). *A limited memory algorithm for bound constrained optimization*. SIAM Journal on Scientific Computing, 16, 1190--1208.
- Zhu, C., Byrd, R. H., Lu, P., and Nocedal, J. (1997). *Algorithm 778: L-BFGS-B, Fortran subroutines for large-scale bound-constrained optimization*. ACM Transactions on Mathematical Software, 23, 550--560.
- Brent, R. P. (1973). *Algorithms for Minimization Without Derivatives*. Prentice-Hall.
- Nash, J. C. (1990). *Compact Numerical Methods for Computers: Linear Algebra and Function Minimisation*. 2nd ed. Adam Hilger.
- Nocedal, J. and Wright, S. J. (1999). *Numerical Optimization*. Springer.
