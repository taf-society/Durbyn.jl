using LinearAlgebra: mul!

"""
    BFGS Quasi-Newton Optimizer

Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton optimization
algorithm with Strong Wolfe line search. BFGS iteratively builds an approximation
to the inverse Hessian using gradient information, achieving superlinear convergence
on smooth problems.

The inverse Hessian approximation is stored as a full symmetric `n × n` matrix and
updated via the standard BFGS formula (N&W Eq. 6.17). The line search satisfies the
strong Wolfe conditions (N&W Algorithms 3.5/3.6), ensuring the curvature condition
`yᵀs > 0` needed for positive-definite Hessian updates.

Convergence is tested by:
1. Gradient norm: `‖∇f(x)‖_∞ < gtol` (N&W Section 6.1, first-order optimality)
2. Relative function change: `|f_new - f_old| / max(1, |f_old|) < reltol`
3. Absolute function value: `f < abstol`

Supports both analytical and numerical gradients (via `numgrad_with_cache!`).

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed.,
  Algorithm 6.1, Eq. 6.17. Springer.
- Broyden, C. G. (1970). The convergence of a class of double-rank minimization
  algorithms. *J. Inst. Math. Appl.*, 6, 76–90.
- Fletcher, R. (1970). A new approach to variable metric algorithms.
  *The Computer Journal*, 13(3), 317–322.
"""

"""
    BFGSOptions

Options for BFGS quasi-Newton optimization.

Fields:
- `abstol::Float64` — Absolute convergence tolerance (default: -Inf, disabled)
- `reltol::Float64` — Relative convergence tolerance (default: √eps)
- `gtol::Float64` — Gradient infinity-norm tolerance for first-order optimality (default: 0, disabled).
  When `gtol > 0`, convergence is declared if `‖∇f(x)‖_∞ < gtol`.
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

Pre-allocated workspace for the BFGS optimizer.

Fields:
- `gradient::Vector{Float64}` — Current gradient vector (length `n_total`)
- `search_dir::Vector{Float64}` — Search direction p = -H*g (length `n_active`)
- `x_active::Vector{Float64}` — Active parameter snapshot (length `n_active`)
- `grad_diff::Vector{Float64}` — Gradient difference yₖ = ∇f_{k+1} - ∇f_k (length `n_active`)
- `inv_hessian::Matrix{Float64}` — Inverse Hessian approximation Hₖ (`n_active × n_active`)
- `Hy::Vector{Float64}` — Buffer for H*y product (length `n_active`)
- `d_full::Vector{Float64}` — Full search direction in original space (length `n_total`)
- `x_trial::Vector{Float64}` — Line-search trial point buffer (length `n_total`)
- `g_trial::Vector{Float64}` — Line-search trial gradient buffer (length `n_total`)
- `g_best::Vector{Float64}` — Best line-search gradient snapshot (length `n_total`)
- `g_saved::Vector{Float64}` — Zoom-phase saved gradient buffer (length `n_total`)
"""
mutable struct BFGSWorkspace
    gradient::Vector{Float64}
    search_dir::Vector{Float64}
    x_active::Vector{Float64}
    grad_diff::Vector{Float64}
    inv_hessian::Matrix{Float64}
    Hy::Vector{Float64}
    d_full::Vector{Float64}
    x_trial::Vector{Float64}
    g_trial::Vector{Float64}
    g_best::Vector{Float64}
    g_saved::Vector{Float64}

    function BFGSWorkspace(n_total::Int, n_active::Int)
        new(
            zeros(n_total),
            zeros(n_active),
            zeros(n_active),
            zeros(n_active),
            zeros(n_active, n_active),
            zeros(n_active),
            zeros(n_total),
            zeros(n_total),
            zeros(n_total),
            zeros(n_total),
            zeros(n_total)
        )
    end
end

"""
    bfgs_hessian_update!(H, s, y, Hy, sTy)

Standard BFGS inverse Hessian update (Nocedal & Wright (2006), Eq. 6.17):

    Hₖ₊₁ = (I - ρ s yᵀ) Hₖ (I - ρ y sᵀ) + ρ s sᵀ

where ρ = 1/(yᵀs). Implemented as the mathematically equivalent form:

    Hₖ₊₁ = Hₖ + (1 + yᵀHy / sᵀy) * (s sᵀ)/(sᵀy) - (s (Hy)ᵀ + Hy sᵀ)/(sᵀy)

This avoids forming the intermediate matrices and updates H in-place.

Arguments:
- `H::Matrix{Float64}` — Inverse Hessian approximation (symmetric, full storage)
- `s::Vector{Float64}` — Step vector sₖ = x_{k+1} - x_k
- `y::Vector{Float64}` — Gradient difference yₖ = ∇f_{k+1} - ∇f_k
- `Hy::Vector{Float64}` — Buffer; on exit contains Hₖ * yₖ
- `sTy::Float64` — Curvature sᵀy (must be positive)

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Eq. 6.17.
  Springer.
"""
@inline function bfgs_hessian_update!(
    H::Matrix{Float64},
    s::Vector{Float64},
    y::Vector{Float64},
    Hy::Vector{Float64},
    sTy::Float64
)
    n = length(s)

    # Compute Hy = H * y (BLAS dgemv for n ≥ 4, hand loop for tiny problems)
    mul!(Hy, H, y)

    # scale = (1 + yᵀHy / sᵀy) / sᵀy
    yTHy = 0.0
    @inbounds @simd for i in 1:n
        yTHy += y[i] * Hy[i]
    end
    scale = (1.0 + yTHy / sTy) / sTy
    inv_sTy = 1.0 / sTy

    # H += scale * s*sᵀ - inv_sTy * (s*Hyᵀ + Hy*sᵀ)
    @inbounds for j in 1:n
        @simd for i in 1:n
            H[i, j] += scale * s[i] * s[j] - inv_sTy * (s[i] * Hy[j] + Hy[i] * s[j])
        end
    end
end


"""
    bfgs(f, g, x0; mask=trues(length(x0)), options=BFGSOptions(),
         step_sizes=nothing, numgrad_cache=nothing) -> NamedTuple

Minimize a function using the BFGS quasi-Newton algorithm with Strong Wolfe
line search (Nocedal & Wright (2006), Algorithm 6.1).

# Arguments

- `f::Function`: Objective function, signature `f(x::Vector)` → `Float64`.
- `g::Union{Function, Nothing}`: Gradient function, signature `g(grad::Vector, x::Vector)`
  modifies `grad` in-place and returns `nothing`. Pass `nothing` for numerical gradients.
- `x0::Vector{Float64}`: Initial parameter vector.
- `mask::BitVector`: Logical mask for active parameters (default: all active).
- `options::BFGSOptions`: Optimization options.
- `step_sizes::Union{Nothing, Vector{Float64}}`: Step sizes for numerical differentiation.
- `numgrad_cache::Union{Nothing, NumericalGradientCache}`: Pre-allocated cache for numerical gradients.

# Returns

A named tuple `(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals)`.
- `fail`: 0 = converged, 1 = reached max iterations or stagnated.

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Algorithm 6.1.
  Springer.

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

    n_total = length(x0)
    active_indices = findall(mask)
    n_active = length(active_indices)
    x_current = copy(x0)

    if maxit <= 0
        f_val = f(x_current)
        return (
            x_opt = copy(x_current),
            f_opt = f_val,
            n_iter = 0,
            fail = 0,
            fn_evals = 1,
            gr_evals = 0
        )
    end

    if report_interval <= 0
        throw(ArgumentError("report_interval must be positive for BFGS"))
    end

    workspace = BFGSWorkspace(n_total, n_active)

    # Set up gradient function (analytical or numerical)
    if isnothing(g)
        ndeps_actual = isnothing(step_sizes) ? fill(1e-3, n_total) : step_sizes
        if length(ndeps_actual) != n_total
            throw(ArgumentError("step_sizes must have length $n_total"))
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

    # Initial function and gradient evaluation
    fval = f(x_current)
    if !isfinite(fval)
        error("BFGS: initial objective value is not finite")
    end

    fn_eval_count = 1
    gr_eval_count = 1
    gfunc(workspace.gradient, x_current)

    # Check initial gradient finiteness
    @inbounds for i in 1:n_active
        if !isfinite(workspace.gradient[active_indices[i]])
            return (x_opt=copy(x_current), f_opt=fval, n_iter=0,
                fail=1, fn_evals=fn_eval_count, gr_evals=gr_eval_count)
        end
    end

    if trace
        println("BFGS: initial value ", fval)
    end

    # Initialize inverse Hessian to identity (N&W Algorithm 6.1, H₀ = I)
    @inbounds for i in 1:n_active
        workspace.inv_hessian[i, i] = 1.0
    end

    # Line search gradient wrapper: writes into workspace buffer, returns it.
    # Non-finite gradients (e.g. at singularities) are replaced with NaN to
    # let the line search detect failure rather than throwing.
    ls_g = x_ls -> begin
        gfunc(workspace.g_trial, x_ls)
        @inbounds for i in eachindex(workspace.g_trial)
            if !isfinite(workspace.g_trial[i])
                fill!(workspace.g_trial, NaN)
                break
            end
        end
        return workspace.g_trial
    end

    f_best = fval
    converged = false

    iter = 0
    while iter < maxit
        iter += 1

        # Save current active parameters and gradient, and extract active
        # gradient into Hy buffer for search direction computation
        @inbounds for i in 1:n_active
            idx = active_indices[i]
            workspace.x_active[i] = x_current[idx]
            gi = workspace.gradient[idx]
            workspace.grad_diff[i] = gi
            workspace.Hy[i] = gi
        end

        # Compute search direction: p = -H * g (N&W Eq. 6.18)
        mul!(workspace.search_dir, workspace.inv_hessian, workspace.Hy)

        # Build full-dimensional search direction and negate
        fill!(workspace.d_full, 0.0)
        @inbounds for i in 1:n_active
            workspace.search_dir[i] = -workspace.search_dir[i]
            workspace.d_full[active_indices[i]] = workspace.search_dir[i]
        end

        # Strong Wolfe line search (N&W Algorithm 3.5)
        ok, alpha, f_new, g_new, x_new, ls_fe, ls_ge = _strong_wolfe_line_search!(
            x_current, f_best, workspace.gradient, workspace.d_full, nothing, nothing,
            f, ls_g;
            c1=1e-4, c2=0.9, iter=iter, boxed=false,
            xtrial_buf=workspace.x_trial, gtrial_buf=workspace.g_trial,
            gbest_buf=workspace.g_best, gsave_buf=workspace.g_saved)
        fn_eval_count += ls_fe
        gr_eval_count += ls_ge

        if !ok || !isfinite(f_new)
            # Line search failed — reset Hessian to identity and retry once
            fill!(workspace.inv_hessian, 0.0)
            @inbounds for i in 1:n_active
                workspace.inv_hessian[i, i] = 1.0
            end

            # Recompute steepest descent direction
            @inbounds for i in 1:n_active
                workspace.search_dir[i] = -workspace.gradient[active_indices[i]]
                workspace.d_full[active_indices[i]] = workspace.search_dir[i]
            end

            ok, alpha, f_new, g_new, x_new, ls_fe2, ls_ge2 = _strong_wolfe_line_search!(
                x_current, f_best, workspace.gradient, workspace.d_full, nothing, nothing,
                f, ls_g;
                c1=1e-4, c2=0.9, iter=iter, boxed=false,
                xtrial_buf=workspace.x_trial, gtrial_buf=workspace.g_trial,
                gbest_buf=workspace.g_best, gsave_buf=workspace.g_saved)
            fn_eval_count += ls_fe2
            gr_eval_count += ls_ge2

            if !ok || !isfinite(f_new)
                if trace
                    println("BFGS: line search failed at iteration ", iter)
                end
                break
            end
        end

        # Accept step
        copyto!(x_current, x_new)
        f_best = f_new
        copyto!(workspace.gradient, g_new)

        # Check for non-finite gradient
        has_nonfinite = false
        @inbounds for i in 1:n_active
            if !isfinite(workspace.gradient[active_indices[i]])
                has_nonfinite = true
                break
            end
        end
        if has_nonfinite
            return (x_opt=copy(x_current), f_opt=f_best, n_iter=iter,
                fail=1, fn_evals=fn_eval_count, gr_evals=gr_eval_count)
        end

        if trace && (iter % report_interval == 0)
            println("BFGS: iteration ", iter, ", value ", f_best)
        end

        # Test convergence: gradient infinity-norm (N&W Section 6.1)
        if gtol > 0.0
            grad_inf_norm = 0.0
            @inbounds for i in 1:n_active
                grad_inf_norm = max(grad_inf_norm, abs(workspace.gradient[active_indices[i]]))
            end
            if grad_inf_norm < gtol
                converged = true
                if trace
                    println("BFGS: converged (gradient norm ", grad_inf_norm, " < ", gtol, ")")
                end
                break
            end
        end

        # Test convergence: relative function change
        rel_change = abs(f_new - fval) / max(1.0, abs(fval))
        if f_best <= abstol || rel_change < reltol
            converged = true
            if trace
                println("BFGS: converged (relative change ", rel_change, " < ", reltol, ")")
            end
            break
        end

        fval = f_new

        # Compute sₖ and yₖ for BFGS update
        sTy = 0.0
        @inbounds for i in 1:n_active
            # sₖ = x_{k+1} - x_k (for active parameters)
            workspace.x_active[i] = x_current[active_indices[i]] - workspace.x_active[i]
            # yₖ = ∇f_{k+1} - ∇f_k
            workspace.grad_diff[i] = workspace.gradient[active_indices[i]] - workspace.grad_diff[i]
            sTy += workspace.x_active[i] * workspace.grad_diff[i]
        end

        if sTy > 0.0
            # Curvature condition satisfied — update inverse Hessian (N&W Eq. 6.17)
            bfgs_hessian_update!(
                workspace.inv_hessian,
                workspace.x_active,      # s
                workspace.grad_diff,      # y
                workspace.Hy,
                sTy
            )
        else
            # Curvature condition violated — reset to identity
            fill!(workspace.inv_hessian, 0.0)
            @inbounds for i in 1:n_active
                workspace.inv_hessian[i, i] = 1.0
            end
        end
    end

    if trace
        println("BFGS: final value ", f_best)
        if converged
            println("BFGS: converged")
        else
            println("BFGS: stopped after ", iter, " iterations")
        end
    end

    fail = converged ? 0 : 1

    return (
        x_opt = copy(x_current),
        f_opt = f_best,
        n_iter = iter,
        fail = fail,
        fn_evals = fn_eval_count,
        gr_evals = gr_eval_count
    )
end
