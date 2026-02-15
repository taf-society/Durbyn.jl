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

- Nocedal, J. & Wright, S. J. (1999). *Numerical Optimization*, Chapter 6.
  Springer.
- Broyden, C. G. (1970). The convergence of a class of double-rank minimization
  algorithms. *J. Inst. Math. Appl.*, 6, 76–90.
- Fletcher, R. (1970). A new approach to variable metric algorithms.
  *The Computer Journal*, 13(3), 317–322.
"""

const RELTEST = 10.0
const ACCTOL = 1.0e-4
const STEPREDN = 0.2

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

Perform the standard **BFGS inverse Hessian update**:

```math
B_{k+1} = B_k + \frac{(1 + c' B_k c / D_1)}{D_1} t t' - \frac{1}{D_1} (t (B_k c)' + (B_k c) t')
```

where `step` is the parameter step, `grad_diff` is the gradient difference, and
`curvature = step' * grad_diff`.

Arguments:
- `inv_hessian::AbstractMatrix` — Inverse Hessian approximation (`n × n`, symmetric)
- `step::AbstractVector` — Step direction vector (length `n`)
- `grad_diff::AbstractVector` — Gradient difference vector (length `n`)
- `Bc::AbstractVector` — Temporary buffer (`n`), used for storing `inv_hessian * grad_diff`
- `curvature::Float64` — Dot product `step' * grad_diff`

This operation updates the inverse Hessian approximation in-place using the
standard BFGS formula. The matrix `inv_hessian` is assumed to be symmetric and stored in
full form.
"""
@inline function bfgs_hessian_update!(
    inv_hessian::AbstractMatrix,
    step::AbstractVector,
    grad_diff::AbstractVector,
    Bc::AbstractVector,
    curvature::Float64
)
    n = length(step)

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
    bfgs(f, g, x0; mask=trues(length(x0)), options=BFGSOptions(), step_sizes=nothing,
         numgrad_cache=nothing, extra=nothing) -> NamedTuple

Minimize a function using the BFGS quasi-Newton algorithm with Armijo backtracking
line search and periodic inverse Hessian restarts.

# Arguments

- `f::Function`: Objective function, signature `f(n::Int, x::Vector, extra)` → `Float64`.
- `g::Union{Function, Nothing}`: Gradient function, signature `g(n::Int, x::Vector, grad::Vector, extra)`
  modifies `grad` in-place and returns `nothing`. Pass `nothing` for numerical gradients.
- `x0::Vector{Float64}`: Initial parameter vector.
- `mask::BitVector`: Logical mask for active parameters (default: all active).
- `options::BFGSOptions`: Optimization options (tolerances, iteration limits, etc.).
- `step_sizes::Union{Nothing, Vector{Float64}}`: Step sizes for numerical differentiation.
- `numgrad_cache::Union{Nothing, NumericalGradientCache}`: Pre-allocated cache for numerical gradients.
- `extra`: External data passed to `f` and `g` (default: `nothing`).

# Returns

A named tuple `(x_opt, f_opt, n_iter, fail, fn_evals, gr_evals)`.

# References

- Nocedal, J. & Wright, S. J. (1999). *Numerical Optimization*, Chapter 6. Springer.
- Nash, J. C. (1990). *Compact Numerical Methods for Computers*, 2nd ed. Adam Hilger.

# Examples

```julia
rosenbrock(n, x, ex) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2

function rosenbrock_grad!(n, x, g, ex)
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
    extra = nothing
)
    abstol = options.abstol
    reltol = options.reltol
    gtol = options.gtol
    trace = options.trace
    maxit = options.maxit
    report_interval = options.report_interval

    converged_gradient = false

    n0 = length(x0)
    l = findall(mask)
    n = length(l)
    b = copy(x0)

    if maxit <= 0
        fail = 0
        Fmin = f(n0, b, extra)
        fncount = 0
        grcount = 0
        return (
            x_opt = copy(b),
            f_opt = Fmin,
            n_iter = 0,
            fail = fail,
            fn_evals = fncount,
            gr_evals = grcount
        )
    end

    if report_interval <= 0
        error("REPORT must be > 0 (method = \"BFGS\")")
    end

    workspace = BFGSWorkspace(n0, n)

    if isnothing(g)
        ndeps_actual = isnothing(step_sizes) ? fill(1e-3, n0) : step_sizes
        if length(ndeps_actual) != n0
            error("ndeps must have length $n0")
        end

        if isnothing(numgrad_cache)
            numgrad_cache = NumericalGradientCache(n0)
        end

        gfunc = function(n, x, g_out, ex_arg)
            g_temp = numgrad_with_cache!(numgrad_cache, f, n, x, ex_arg, ndeps_actual)
            g_out .= g_temp
            nothing
        end
    else
        gfunc = g
    end

    fval = f(n0, b, extra)
    if !isfinite(fval)
        error("initial value in 'vmmin' is not finite")
    end
    if trace
        println("initial  value $fval ")
    end

    Fmin = fval
    funcount = 1
    gradcount = 1

    gfunc(n0, b, workspace.gradient, extra)

    @inbounds for i in 1:n
        if !isfinite(workspace.gradient[l[i]])
            return (x_opt=copy(b), f_opt=Fmin, n_iter=0,
                fail=1, fn_evals=funcount, gr_evals=gradcount)
        end
    end

    iter = 0
    iter += 1
    ilast = gradcount

    while true
        if ilast == gradcount
            fill!(workspace.inv_hessian, 0.0)
            @inbounds for i in 1:n
                workspace.inv_hessian[i, i] = 1.0
            end
        end

        @inbounds for i in 1:n
            workspace.x[i] = b[l[i]]
            workspace.grad_diff[i] = workspace.gradient[l[i]]
        end

        gradproj = 0.0
        @inbounds for i in 1:n
            s = 0.0
            @simd for j in 1:i
                s -= workspace.inv_hessian[i, j] * workspace.gradient[l[j]]
            end
            @simd for j in (i+1):n
                s -= workspace.inv_hessian[j, i] * workspace.gradient[l[j]]
            end
            workspace.step[i] = s
            gradproj += s * workspace.gradient[l[i]]
        end

        if gradproj < 0.0
            steplength = 1.0
            accpoint = false
            count = n
            while true
                count = 0
                @inbounds for i in 1:n
                    b[l[i]] = workspace.x[i] + steplength * workspace.step[i]
                    if RELTEST + workspace.x[i] == RELTEST + b[l[i]]
                        count += 1
                    end
                end
                if count < n
                    fval = f(n0, b, extra)
                    funcount += 1
                    accpoint = isfinite(fval) && (fval <= Fmin + gradproj * steplength * ACCTOL)
                    if !accpoint
                        steplength *= STEPREDN
                    end
                end
                if count == n || accpoint
                    break
                end
            end

            enough = (fval > abstol) && abs(fval - Fmin) > reltol * (abs(Fmin) + reltol)
            if !enough
                count = n
                Fmin = fval
            end

            if count < n
                Fmin = fval
                gfunc(n0, b, workspace.gradient, extra)
                gradcount += 1
                iter += 1

                has_nonfinite_grad = false
                @inbounds for i in 1:n
                    if !isfinite(workspace.gradient[l[i]])
                        has_nonfinite_grad = true
                        break
                    end
                end
                if has_nonfinite_grad
                    return (x_opt=copy(b), f_opt=Fmin, n_iter=iter,
                        fail=1, fn_evals=funcount, gr_evals=gradcount)
                end

                D1 = 0.0
                @inbounds for i in 1:n
                    workspace.step[i] = steplength * workspace.step[i]
                    workspace.grad_diff[i] = workspace.gradient[l[i]] - workspace.grad_diff[i]
                    D1 += workspace.step[i] * workspace.grad_diff[i]
                end

                if D1 > 0.0
                    bfgs_hessian_update!(
                        workspace.inv_hessian,
                        workspace.step,
                        workspace.grad_diff,
                        workspace.Bc,
                        D1
                    )
                else
                    ilast = gradcount
                end
            else
                if ilast < gradcount
                    count = 0
                    ilast = gradcount
                end
            end
        else
            count = 0
            if ilast == gradcount
                count = n
            else
                ilast = gradcount
            end
        end

        if trace && (iter % report_interval == 0)
            println("iter", lpad(iter, 4), " value ", fval)
        end

        if gtol > 0.0
            grad_norm_sq = 0.0
            @inbounds for i in 1:n
                grad_norm_sq += workspace.gradient[l[i]]^2
            end
            grad_norm = sqrt(grad_norm_sq)
            if grad_norm < gtol * max(1.0, abs(Fmin))
                converged_gradient = true
                if trace
                    println("converged: gradient norm ", grad_norm, " < ", gtol * max(1.0, abs(Fmin)))
                end
                break
            end
        end

        if iter >= maxit
            break
        end
        if gradcount - ilast > 2 * n
            ilast = gradcount
        end
        if count == n && ilast == gradcount
            break
        end
    end

    if trace
        println("final  value ", Fmin, " ")
        if iter < maxit || converged_gradient
            println("converged")
        else
            println("stopped after ", iter, " iterations")
        end
    end

    fail = (iter < maxit || converged_gradient) ? 0 : 1

    return (
        x_opt = copy(b),
        f_opt = Fmin,
        n_iter = iter,
        fail = fail,
        fn_evals = funcount,
        gr_evals = gradcount
    )
end