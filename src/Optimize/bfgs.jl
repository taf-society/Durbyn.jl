"""
    BFGS Quasi-Newton Optimizer

Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton optimization
algorithm. BFGS iteratively builds an approximation to the inverse Hessian using
gradient information, achieving superlinear convergence on smooth problems.

Features an optional gradient norm convergence check (`gtol` parameter) based on
the first-order necessary condition for optimality: `||∇f(x)|| < gtol * max(1, |f(x)|)`.
This is disabled by default (`gtol = 0`).

Supports both analytical and numerical gradients (via `numgrad_with_cache!`).

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Algorithm 6.1.
  Springer.
- Broyden, C. G. (1970). The convergence of a class of double-rank minimization
  algorithms. *J. Inst. Math. Appl.*, 6, 76–90.
- Fletcher, R. (1970). A new approach to variable metric algorithms.
  *The Computer Journal*, 13(3), 317–322.
"""

# Armijo sufficient decrease parameter c₁ (Nocedal & Wright (2006), Eq. 3.6a)
const SUFFICIENT_DECREASE = 1.0e-4

# Backtracking contraction factor ρ ∈ (0,1) (Nocedal & Wright (2006), Algorithm 3.1)
const BACKTRACK_FACTOR = 0.2

"""
    BFGSOptions

Options for BFGS quasi-Newton optimization.

Fields:
- `abstol::Float64` — Absolute convergence tolerance (default: -Inf, disabled)
- `reltol::Float64` — Relative convergence tolerance (default: √eps)
- `gtol::Float64` — Gradient norm tolerance for first-order optimality (default: 0, disabled).
  When `gtol > 0`, convergence is declared if `||∇f(x)|| < gtol * max(1, |f(x)|)`.
  Based on the first-order necessary condition for optimality (∇f(x*) = 0).
  Helps convergence on flat surfaces.
- `trace::Bool` — Print iteration progress (default: false)
- `maxit::Int` — Maximum iterations (default: 100)
- `report_interval::Int` — Reporting frequency when trace=true (default: 10)
"""
Base.@kwdef struct BFGSOptions
    abstol::Float64 = -Inf
    reltol::Float64 = sqrt(eps(Float64))
    gtol::Float64 = 0.0
    trace::Bool = false
    maxit::Int = 100
    report_interval::Int = 10
end


"""
    BFGSWorkspace

Pre-allocated workspace used internally by the BFGS optimizer to avoid
memory allocations during iterative updates.

Fields:
- `gradient::Vector{Float64}` — Current gradient vector (length `n0`)
- `step::Vector{Float64}` — Search direction (length `n`)
- `x::Vector{Float64}` — Current parameter vector (length `n`)
- `grad_diff::Vector{Float64}` — Gradient difference vector (length `n`)
- `inv_hessian::Matrix{Float64}` — Approximate inverse Hessian matrix (`n × n`)
- `Bc::Vector{Float64}` — Temporary buffer for B*c product in Hessian updates (length `n`)
- `gradient_new::Vector{Float64}` — Buffer for storing the new gradient (length `n0`)

The workspace is automatically managed by the optimizer, but it can also be
created and reused manually if desired.
"""
mutable struct BFGSWorkspace
    gradient::Vector{Float64}
    step::Vector{Float64}
    x::Vector{Float64}
    grad_diff::Vector{Float64}
    inv_hessian::Matrix{Float64}
    Bc::Vector{Float64}
    gradient_new::Vector{Float64}

    function BFGSWorkspace(n0::Int, n::Int)
        new(
            zeros(n0),
            zeros(n),
            zeros(n),
            zeros(n),
            zeros(n, n),
            zeros(n),
            zeros(n0)
        )
    end
end

"""
    bfgs_hessian_update!(inv_hessian, step, grad_diff, Bc, curvature)

Perform the standard **BFGS inverse Hessian update** (Nocedal & Wright (2006), Eq. 6.17):

```math
H_{k+1} = (I - ρ_k s_k y_k^T) H_k (I - ρ_k y_k s_k^T) + ρ_k s_k s_k^T
```

where `s_k = step`, `y_k = grad_diff`, and `ρ_k = 1 / (y_k^T s_k)`.

Arguments:
- `inv_hessian::AbstractMatrix` — Inverse Hessian approximation (`n × n`, symmetric)
- `step::AbstractVector` — Step vector sₖ = x_{k+1} - x_k (length `n`)
- `grad_diff::AbstractVector` — Gradient difference yₖ = ∇f_{k+1} - ∇f_k (length `n`)
- `Bc::AbstractVector` — Temporary buffer (`n`), used for storing `H_k * y_k`
- `curvature::Float64` — Curvature condition `y_k^T s_k` (must be positive)

This operation updates the inverse Hessian approximation in-place using the
standard BFGS formula. The matrix `inv_hessian` is assumed to be symmetric and stored in
full form (lower triangle).

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Eq. 6.17.
  Springer.
"""
@inline function bfgs_hessian_update!(
    inv_hessian::AbstractMatrix,
    step::AbstractVector,
    grad_diff::AbstractVector,
    Bc::AbstractVector,
    curvature::Float64
)
    n = length(step)

    # Compute Bc = H_k * y_k (using lower-triangular storage)
    @inbounds for i in 1:n
        s = 0.0
        @simd for j in 1:i
            s += inv_hessian[i, j] * grad_diff[j]
        end
        @simd for j in (i+1):n
            s += inv_hessian[j, i] * grad_diff[j]
        end
        Bc[i] = s
    end

    # D2 = 1 + y_k^T H_k y_k / (s_k^T y_k)
    D2 = 0.0
    @inbounds @simd for i in 1:n
        D2 += Bc[i] * grad_diff[i]
    end
    D2 = 1.0 + D2 / curvature

    inv_D1 = 1.0 / curvature
    @inbounds for i in 1:n
        ti_scaled = step[i] * inv_D1
        xi_scaled = Bc[i] * inv_D1

        @simd for j in 1:i
            inv_hessian[i, j] += D2 * ti_scaled * step[j] - xi_scaled * step[j] - ti_scaled * Bc[j]
        end
    end
end

"""
    _armijo_backtrack!(x_current, workspace, active_indices, n_active, f, f_best, dir_deriv)

Perform Armijo backtracking line search (Nocedal & Wright (2006), Algorithm 3.1).

Returns `(steplength, fval, stagnant, fn_evals)` where `stagnant` indicates
the step had no numerically detectable effect on the parameters.
"""
function _armijo_backtrack!(
    x_current::Vector{Float64},
    workspace::BFGSWorkspace,
    active_indices::Vector{Int},
    n_active::Int,
    f::Function,
    f_best::Float64,
    dir_deriv::Float64
)
    steplength = 1.0
    fval = f_best
    fn_evals = 0

    while true
        @inbounds for i in 1:n_active
            x_current[active_indices[i]] = workspace.x[i] + steplength * workspace.step[i]
        end

        # Check if step is negligible relative to x
        max_rel_step = 0.0
        @inbounds for i in 1:n_active
            dx = abs(steplength * workspace.step[i])
            max_rel_step = max(max_rel_step, dx / max(abs(workspace.x[i]), 1.0))
        end
        if max_rel_step < eps(Float64)
            return steplength, fval, true, fn_evals
        end

        fval = f(x_current)
        fn_evals += 1

        if isfinite(fval) && (fval <= f_best + dir_deriv * steplength * SUFFICIENT_DECREASE)
            return steplength, fval, false, fn_evals
        end
        steplength *= BACKTRACK_FACTOR
    end
end

"""
    bfgs(f, g, x0; mask=trues(length(x0)), options=BFGSOptions(), step_sizes=nothing,
         numgrad_cache=nothing) -> NamedTuple

Minimize a function using the BFGS quasi-Newton algorithm with Armijo backtracking
line search and periodic inverse Hessian restarts.

# Arguments

- `f::Function`: Objective function, signature `f(x::Vector)` → `Float64`.
- `g::Union{Function, Nothing}`: Gradient function, signature `g(grad::Vector, x::Vector)`
  modifies `grad` in-place and returns `nothing`. Pass `nothing` for numerical gradients.
- `x0::Vector{Float64}`: Initial parameter vector.
- `mask::BitVector`: Logical mask for active parameters (default: all active).
- `options::BFGSOptions`: Optimization options (tolerances, iteration limits, etc.).
- `step_sizes::Union{Nothing, Vector{Float64}}`: Step sizes for numerical differentiation.
- `numgrad_cache::Union{Nothing, NumericalGradientCache}`: Pre-allocated cache for numerical gradients.

# Returns

A named tuple `(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals)`.

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Algorithm 6.1.
  Springer.
- Nash, J. C. (1990). *Compact Numerical Methods for Computers*, 2nd ed. Adam Hilger.

# Examples

```julia
rosenbrock(x) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2

function rosenbrock_grad!(g, x)
    g[1] = -400.0 * (x[2] - x[1]^2) * x[1] - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * (x[2] - x[1]^2)
    nothing
end

result = bfgs(rosenbrock, rosenbrock_grad!, [-1.2, 1.0])
```
"""
function bfgs(
    f::Function,
    g::Union{Function,Nothing},
    x0::Vector{Float64};
    mask = trues(length(x0)),
    options::BFGSOptions = BFGSOptions(),
    step_sizes::Union{Nothing,Vector{Float64}} = nothing,
    numgrad_cache::Union{Nothing,NumericalGradientCache} = nothing,
)
    abstol = options.abstol
    reltol = options.reltol
    gtol = options.gtol
    trace = options.trace
    maxit = options.maxit
    report_interval = options.report_interval

    converged_gradient = false

    n_total = length(x0)
    active_indices = findall(mask)
    n_active = length(active_indices)
    x_current = copy(x0)

    if maxit <= 0
        fail = 0
        f_best = f(x_current)
        fn_eval_count = 0
        grad_eval_count = 0
        return (
            x_opt = copy(x_current),
            f_opt = f_best,
            n_iter = 0,
            fail = fail,
            fn_evals = fn_eval_count,
            gr_evals = grad_eval_count
        )
    end

    if report_interval <= 0
        throw(ArgumentError("report_interval must be positive for BFGS"))
    end

    workspace = BFGSWorkspace(n_total, n_active)

    if isnothing(g)
        ndeps_actual = isnothing(step_sizes) ? fill(1e-3, n_total) : step_sizes
        if length(ndeps_actual) != n_total
            throw(ArgumentError("ndeps must have length $n_total"))
        end

        if isnothing(numgrad_cache)
            numgrad_cache = NumericalGradientCache(n_total)
        end

        gfunc = function(g_out, x)
            g_temp = numgrad_with_cache!(numgrad_cache, f, x, ndeps_actual)
            g_out .= g_temp
            nothing
        end
    else
        gfunc = g
    end

    fval = f(x_current)
    if !isfinite(fval)
        error("BFGS: initial objective value is not finite")
    end
    if trace
        println("initial  value $fval ")
    end

    f_best = fval
    fn_eval_count = 1
    grad_eval_count = 1

    gfunc(workspace.gradient, x_current)

    @inbounds for i in 1:n_active
        if !isfinite(workspace.gradient[active_indices[i]])
            return (x_opt=copy(x_current), f_opt=f_best, n_iter=0,
                fail=1, fn_evals=fn_eval_count, gr_evals=grad_eval_count)
        end
    end

    iter = 1
    last_restart_eval = grad_eval_count

    while true
        # Reset inverse Hessian to identity when needed (periodic restart)
        if last_restart_eval == grad_eval_count
            fill!(workspace.inv_hessian, 0.0)
            @inbounds for i in 1:n_active
                workspace.inv_hessian[i, i] = 1.0
            end
        end

        # Save current active parameters and gradient
        @inbounds for i in 1:n_active
            workspace.x[i] = x_current[active_indices[i]]
            workspace.grad_diff[i] = workspace.gradient[active_indices[i]]
        end

        # Compute search direction: p = -H * g (Nocedal & Wright, Eq. 6.18)
        dir_deriv = 0.0
        @inbounds for i in 1:n_active
            s = 0.0
            @simd for j in 1:i
                s -= workspace.inv_hessian[i, j] * workspace.gradient[active_indices[j]]
            end
            @simd for j in (i+1):n_active
                s -= workspace.inv_hessian[j, i] * workspace.gradient[active_indices[j]]
            end
            workspace.step[i] = s
            dir_deriv += s * workspace.gradient[active_indices[i]]
        end

        stagnant = true
        if dir_deriv < 0.0
            # Armijo backtracking line search (Nocedal & Wright, Algorithm 3.1)
            steplength, fval, stagnant, bt_evals = _armijo_backtrack!(
                x_current, workspace, active_indices, n_active,
                f, f_best, dir_deriv)
            fn_eval_count += bt_evals

            enough = (fval > abstol) && abs(fval - f_best) > reltol * (abs(f_best) + reltol)
            if !enough
                stagnant = true
                f_best = fval
            end

            if !stagnant
                f_best = fval
                gfunc(workspace.gradient, x_current)
                grad_eval_count += 1
                iter += 1

                has_nonfinite_grad = false
                @inbounds for i in 1:n_active
                    if !isfinite(workspace.gradient[active_indices[i]])
                        has_nonfinite_grad = true
                        break
                    end
                end
                if has_nonfinite_grad
                    return (x_opt=copy(x_current), f_opt=f_best, n_iter=iter,
                        fail=1, fn_evals=fn_eval_count, gr_evals=grad_eval_count)
                end

                # BFGS update: compute sₖ and yₖ, then curvature sₖᵀyₖ
                curvature = 0.0
                @inbounds for i in 1:n_active
                    workspace.step[i] = steplength * workspace.step[i]
                    workspace.grad_diff[i] = workspace.gradient[active_indices[i]] - workspace.grad_diff[i]
                    curvature += workspace.step[i] * workspace.grad_diff[i]
                end

                if curvature > 0.0
                    # Curvature condition satisfied — update inverse Hessian
                    bfgs_hessian_update!(
                        workspace.inv_hessian,
                        workspace.step,
                        workspace.grad_diff,
                        workspace.Bc,
                        curvature
                    )
                else
                    # Skip update, schedule restart
                    last_restart_eval = grad_eval_count
                end
            else
                if last_restart_eval < grad_eval_count
                    stagnant = false
                    last_restart_eval = grad_eval_count
                end
            end
        else
            # Search direction is not a descent direction — restart
            stagnant = false
            if last_restart_eval == grad_eval_count
                stagnant = true
            else
                last_restart_eval = grad_eval_count
            end
        end

        if trace && (iter % report_interval == 0)
            println("iter", lpad(iter, 4), " value ", fval)
        end

        # Gradient norm convergence check (first-order optimality)
        if gtol > 0.0
            grad_norm_sq = 0.0
            @inbounds for i in 1:n_active
                grad_norm_sq += workspace.gradient[active_indices[i]]^2
            end
            grad_norm = sqrt(grad_norm_sq)
            if grad_norm < gtol * max(1.0, abs(f_best))
                converged_gradient = true
                if trace
                    println("converged: gradient norm ", grad_norm, " < ", gtol * max(1.0, abs(f_best)))
                end
                break
            end
        end

        if iter >= maxit
            break
        end
        # Periodic restart after 2n gradient evaluations without improvement
        if grad_eval_count - last_restart_eval > 2 * n_active
            last_restart_eval = grad_eval_count
        end
        if stagnant && last_restart_eval == grad_eval_count
            break
        end
    end

    if trace
        println("final  value ", f_best, " ")
        if iter < maxit || converged_gradient
            println("converged")
        else
            println("stopped after ", iter, " iterations")
        end
    end

    fail = (iter < maxit || converged_gradient) ? 0 : 1

    return (
        x_opt = copy(x_current),
        f_opt = f_best,
        n_iter = iter,
        fail = fail,
        fn_evals = fn_eval_count,
        gr_evals = grad_eval_count
    )
end
