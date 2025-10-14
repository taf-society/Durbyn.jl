"""
    BFGS Quasi-Newton Optimizer

Implements a general-purpose **BFGS quasi-Newton optimization algorithm** for
unconstrained or bound-constrained problems. Suitable for use in nonlinear
optimization tasks where the objective function may or may not provide an
analytical gradient.

This implementation supports both analytical and numerical gradients (via
`numgrad_with_cache!`) and is compatible with standard gradient-based
optimizers and model fitting workflows.
"""

"""
    BFGSOptions

Options for BFGS quasi-Newton optimization.

# Fields
- `abstol::Float64` — Absolute convergence tolerance (default: -Inf, disabled)
- `reltol::Float64` — Relative convergence tolerance (default: √eps)
- `trace::Bool` — Print iteration progress (default: false)
- `maxit::Int` — Maximum iterations (default: 100)
- `nREPORT::Int` — Reporting frequency when trace=true (default: 10)
"""
Base.@kwdef struct BFGSOptions
    abstol::Float64 = -Inf
    reltol::Float64 = sqrt(eps(Float64))
    trace::Bool = false
    maxit::Int = 100
    nREPORT::Int = 10
end


"""
    BFGSWorkspace

Pre-allocated workspace used internally by the BFGS optimizer to avoid
memory allocations during iterative updates.

# Fields
- `gvec::Vector{Float64}` — Current gradient vector (length `n0`)
- `t::Vector{Float64}` — Search direction (length `n`)
- `X::Vector{Float64}` — Current parameter vector (length `n`)
- `c::Vector{Float64}` — Gradient difference vector (length `n`)
- `B::Matrix{Float64}` — Approximate inverse Hessian matrix (`n × n`)
- `X_cache::Vector{Float64}` — Temporary buffer used in Hessian updates (length `n`)
- `gnew::Vector{Float64}` — Buffer for storing the new gradient (length `n0`)

The workspace is automatically managed by the optimizer, but it can also be
created and reused manually if desired.
"""
mutable struct BFGSWorkspace
    gvec::Vector{Float64}
    t::Vector{Float64}
    X::Vector{Float64}
    c::Vector{Float64}
    B::Matrix{Float64}
    X_cache::Vector{Float64}
    gnew::Vector{Float64}

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
    bfgs_hessian_update!(B, t, c, X_cache, D1)

Perform the standard **BFGS inverse Hessian update**:

```math
B_{k+1} = B_k + \frac{(1 + c' B_k c / D_1)}{D_1} t t' - \frac{1}{D_1} (t (B_k c)' + (B_k c) t')
````

where `t` is the parameter step, `c` is the gradient difference, and
`D1 = t' * c`.

# Arguments
- `B::AbstractMatrix` — Inverse Hessian approximation (`n × n`, symmetric)
- `t::AbstractVector` — Step direction vector (length `n`)
- `c::AbstractVector` — Gradient difference vector (length `n`)
- `X_cache::AbstractVector` — Temporary buffer (`n`), used for storing `B * c`
- `D1::Float64` — Dot product `t' * c`

# Notes
This operation updates the inverse Hessian approximation in-place using the
standard BFGS formula. The matrix `B` is assumed to be symmetric and stored in
full form.
"""
@inline function bfgs_hessian_update!(
    B::AbstractMatrix,
    t::AbstractVector,
    c::AbstractVector,
    X_cache::AbstractVector,
    D1::Float64
)
    n = length(t)

    # Compute B * c → X_cache (BLAS-level-2, optimized for symmetric lower triangular)
    @inbounds for i in 1:n
        s = 0.0
        # Lower triangular part: B[i,j] for j ≤ i
        @simd for j in 1:i
            s += B[i, j] * c[j]
        end
        # Upper triangular part (symmetric): B[j,i] for j > i
        @simd for j in (i+1):n
            s += B[j, i] * c[j]
        end
        X_cache[i] = s
    end

    # Compute D2 = c' * B * c / D1
    D2 = 0.0
    @inbounds @simd for i in 1:n
        D2 += X_cache[i] * c[i]
    end
    D2 = 1.0 + D2 / D1

    # Symmetric rank-2 update: B += (D2 * t * t' - X_cache * t' - t * X_cache') / D1
    inv_D1 = inv(D1)
    @inbounds for i in 1:n
        ti_scaled = t[i] * inv_D1
        xi_scaled = X_cache[i] * inv_D1

        # Only update lower triangular part (B is symmetric)
        @simd for j in 1:i
            B[i, j] += D2 * ti_scaled * t[j] - xi_scaled * t[j] - ti_scaled * X_cache[j]
        end
    end
end

""""
    bfgsmin(f, g, x0; mask=trues(length(x0)), options=BFGSOptions(), ndeps=nothing, 
    numgrad_cache=nothing) -> NamedTuple

Perform unconstrained optimization using the **BFGS quasi-Newton method**.

# Arguments
- `f::Function` — Objective function, called as `f(n, x, ex)`
- `g::Union{Function, Nothing}` — Gradient function, or `nothing` to use numerical gradients
- `x0::Vector{Float64}` — Initial parameter vector
- `mask::BitVector` — Logical mask for active parameters (default: all active)
- `options::BFGSOptions` — Optimization options (tolerances, iteration limits, etc.)
- `ndeps::Union{Nothing, Vector{Float64}}` — Step sizes for numerical differentiation (used if `g` is `nothing`)
- `numgrad_cache::Union{Nothing, NumericalGradientCache}` — Optional pre-allocated cache for numerical gradients

# Returns
A named tuple containing:
- `x_opt` — Optimal parameter vector
- `f_opt` — Objective function value at the optimum
- `n_iter` — Number of iterations performed
- `fail` — Status flag (`0` if converged, `1` otherwise)
- `fn_evals` — Number of function evaluations
- `gr_evals` — Number of gradient evaluations

# Behavior
If no analytical gradient is supplied, numerical gradients are computed
using central differences with an optional cache for efficiency.
The algorithm terminates when the gradient norm falls below tolerance
or when the iteration limit is reached.

# Example
```julia
rosenbrock(n, x, ex) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
rosenbrock_grad(n, x, ex) = [
    -400 * (x[2] - x[1]^2) * x[1] - 2 * (1 - x[1]),
    200 * (x[2] - x[1]^2)
]

x0 = [-1.2, 1.0]
result = bfgsmin(rosenbrock, rosenbrock_grad, x0)
println("Optimal x: ", result.x_opt)
```
"""
function bfgsmin(
    f::Function,
    g::Union{Function,Nothing},
    x0::Vector{Float64};
    mask = trues(length(x0)),
    options::BFGSOptions = BFGSOptions(),
    ndeps::Union{Nothing,Vector{Float64}} = nothing,
    numgrad_cache::Union{Nothing,NumericalGradientCache} = nothing
)
    # Extract options
    abstol = options.abstol
    reltol = options.reltol
    trace = options.trace
    maxit = options.maxit
    nREPORT = options.nREPORT

    n0 = length(x0)
    l = findall(mask)
    n = length(l)
    b = copy(x0)

    # Pre-allocate workspace
    workspace = BFGSWorkspace(n0, n)

    # Setup gradient function (analytical or numerical with cache)
    if isnothing(g)
        # Use numerical gradients with cache
        ndeps_actual = isnothing(ndeps) ? fill(1e-3, n0) : ndeps
        if length(ndeps_actual) != n0
            error("ndeps must have length $n0")
        end

        # Use provided cache or create new one
        if isnothing(numgrad_cache)
            numgrad_cache = NumericalGradientCache(n0)
        end

        # Gradient function using cached numgrad
        gfunc = (n, x, ex) -> numgrad_with_cache!(numgrad_cache, f, n, x, ex, ndeps_actual)
    else
        gfunc = g
    end

    fval = f(n0, b, nothing)
    if !isfinite(fval)
        error("Initial value in 'bfgsmin' is not finite")
    end
    if trace
        println("initial value $fval")
    end

    Fmin = fval
    funcount = 1
    gradcount = 1
    workspace.gvec .= gfunc(n0, b, nothing)
    iter = 1
    ilast = gradcount
    fail = 1
    count = n

    while true
        # Restart Hessian if required
        if ilast == gradcount
            @inbounds for i in 1:n, j in 1:n
                workspace.B[i, j] = (i == j) ? 1.0 : 0.0
            end
        end

        # Extract active parameters and gradients
        @inbounds for i in 1:n
            workspace.X[i] = b[l[i]]
            workspace.c[i] = workspace.gvec[l[i]]
        end

        # Compute search direction: t = -B * g
        gradproj = 0.0
        @inbounds for i in 1:n
            s = 0.0
            # Lower triangular
            @simd for j in 1:i
                s -= workspace.B[i, j] * workspace.gvec[l[j]]
            end
            # Upper triangular (symmetric)
            @simd for j in (i+1):n
                s -= workspace.B[j, i] * workspace.gvec[l[j]]
            end
            workspace.t[i] = s
            gradproj += s * workspace.gvec[l[i]]
        end

        if gradproj < 0.0
            # Line search
            steplength = 1.0
            accpoint = false
            while true
                count = 0
                @inbounds for i in 1:n
                    tmp = workspace.X[i] + steplength * workspace.t[i]
                    if 1.0 + workspace.X[i] == 1.0 + tmp
                        count += 1
                    end
                    b[l[i]] = tmp
                end
                if count < n
                    fval = f(n0, b, nothing)
                    funcount += 1
                    accpoint = isfinite(fval) && (fval <= Fmin + gradproj * steplength * 1e-4)
                    if !accpoint
                        steplength *= 0.2
                    end
                end
                if count == n || accpoint
                    break
                end
            end

            # Check convergence
            enough = (fval > abstol) && abs(fval - Fmin) > reltol * (abs(Fmin) + reltol)
            if !enough
                count = n
                Fmin = fval
            end

            if count < n
                Fmin = fval
                workspace.gnew .= gfunc(n0, b, nothing)
                gradcount += 1
                iter += 1

                # Compute step and gradient difference
                D1 = 0.0
                @inbounds for i in 1:n
                    workspace.t[i] = steplength * workspace.t[i]
                    workspace.c[i] = workspace.gnew[l[i]] - workspace.c[i]
                    D1 += workspace.t[i] * workspace.c[i]
                end

                # BFGS update (if D1 > 0)
                if D1 > 0
                    # Optimized Hessian update with pre-allocated cache
                    bfgs_hessian_update!(
                        workspace.B,
                        workspace.t,
                        workspace.c,
                        workspace.X_cache,
                        D1
                    )
                else
                    ilast = gradcount
                end

                workspace.gvec .= workspace.gnew
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

        if trace && (iter % nREPORT == 0)
            println("iter $iter value $fval")
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
        println("final value $Fmin")
        if iter < maxit
            println("converged")
        else
            println("stopped after $iter iterations")
        end
    end
    fail = (iter < maxit) ? 0 : 1

    return (
        x_opt = copy(b),
        f_opt = Fmin,
        n_iter = iter,
        fail = fail,
        fn_evals = funcount,
        gr_evals = gradcount
    )
end