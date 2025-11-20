"""
    BFGS Quasi-Newton Optimizer

Line-for-line port of the C `vmmin` from R's `stats::optim`.

Implements the BFGS quasi-Newton optimization algorithm exactly as in the C code,
including all edge cases, constants, and iteration logic.

This implementation supports both analytical and numerical gradients (via
`numgrad_with_cache!`) and is compatible with standard gradient-based
optimizers and model fitting workflows.
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

Fields:
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

Arguments:
- `B::AbstractMatrix` — Inverse Hessian approximation (`n × n`, symmetric)
- `t::AbstractVector` — Step direction vector (length `n`)
- `c::AbstractVector` — Gradient difference vector (length `n`)
- `X_cache::AbstractVector` — Temporary buffer (`n`), used for storing `B * c`
- `D1::Float64` — Dot product `t' * c`

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

    @inbounds for i in 1:n
        s = 0.0
        @simd for j in 1:i
            s += B[i, j] * c[j]
        end
        @simd for j in (i+1):n
            s += B[j, i] * c[j]
        end
        X_cache[i] = s
    end

    D2 = 0.0
    @inbounds @simd for i in 1:n
        D2 += X_cache[i] * c[i]
    end
    D2 = 1.0 + D2 / D1

    inv_D1 = 1.0 / D1
    @inbounds for i in 1:n
        ti_scaled = t[i] * inv_D1
        xi_scaled = X_cache[i] * inv_D1

        @simd for j in 1:i
            B[i, j] += D2 * ti_scaled * t[j] - xi_scaled * t[j] - ti_scaled * X_cache[j]
        end
    end
end

""""
    bfgsmin(f, g, x0; mask=trues(length(x0)), options=BFGSOptions(), ndeps=nothing,
    numgrad_cache=nothing, ex=nothing) -> NamedTuple

Line-for-line port of the C `vmmin` BFGS quasi-Newton optimizer from R's `stats::optim`.

This implementation exactly reproduces the C code behavior including:
- Step-change test using RELTEST = 10.0 (from R source optim.c line 106)
- Armijo line search with ACCTOL = 1e-4, STEPREDN = 0.2
- BFGS update gated on D1 > 0
- Periodic Hessian restarts (every 2n gradients)
- Exact iteration counting and convergence checks
- maxit <= 0 and nREPORT <= 0 handling

Arguments:
- `f::Function` — Objective function, signature: `f(n::Int, x::Vector, ex)` returns `Float64`
- `g::Union{Function, Nothing}` — Gradient function, signature: `g(n::Int, x::Vector, grad::Vector, ex)`
  modifies `grad` in-place and returns `nothing`. Use `nothing` for numerical gradients.
- `x0::Vector{Float64}` — Initial parameter vector
- `mask::BitVector` — Logical mask for active parameters (default: all active)
- `options::BFGSOptions` — Optimization options (tolerances, iteration limits, etc.)
- `ndeps::Union{Nothing, Vector{Float64}}` — Step sizes for numerical differentiation (used if `g` is `nothing`)
- `numgrad_cache::Union{Nothing, NumericalGradientCache}` — Optional pre-allocated cache for numerical gradients
- `ex` — External data passed to `f` and `g` (default: `nothing`). Use closures if you need to capture data.

Returns a named tuple containing:
- `x_opt` — Optimal parameter vector
- `f_opt` — Objective function value at the optimum
- `n_iter` — Number of iterations performed
- `fail` — Status flag (`0` if converged, `1` otherwise)
- `fn_evals` — Number of function evaluations
- `gr_evals` — Number of gradient evaluations

Example:
```julia
rosenbrock(n, x, ex) = 100.0 * (x[2] - x[1]^2)^2 + (1.0 - x[1])^2

function rosenbrock_grad!(n, x, g, ex)
    g[1] = -400.0 * (x[2] - x[1]^2) * x[1] - 2.0 * (1.0 - x[1])
    g[2] = 200.0 * (x[2] - x[1]^2)
    nothing
end

x0 = [-1.2, 1.0]
result = bfgsmin(rosenbrock, rosenbrock_grad!, x0)
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
    numgrad_cache::Union{Nothing,NumericalGradientCache} = nothing,
    ex = nothing
)
    abstol = options.abstol
    reltol = options.reltol
    trace = options.trace
    maxit = options.maxit
    nREPORT = options.nREPORT

    n0 = length(x0)
    l = findall(mask)
    n = length(l)
    b = copy(x0)

    if maxit <= 0
        fail = 0
        Fmin = f(n0, b, ex)
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

    if nREPORT <= 0
        error("REPORT must be > 0 (method = \"BFGS\")")
    end

    workspace = BFGSWorkspace(n0, n)

    if isnothing(g)
        ndeps_actual = isnothing(ndeps) ? fill(1e-3, n0) : ndeps
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

    fval = f(n0, b, ex)
    if !isfinite(fval)
        error("initial value in 'vmmin' is not finite")
    end
    if trace
        println("initial  value $fval ")
    end

    Fmin = fval
    funcount = 1
    gradcount = 1

    gfunc(n0, b, workspace.gvec, ex)

    iter = 0
    iter += 1
    ilast = gradcount

    while true
        if ilast == gradcount
            fill!(workspace.B, 0.0)
            @inbounds for i in 1:n
                workspace.B[i, i] = 1.0
            end
        end

        @inbounds for i in 1:n
            workspace.X[i] = b[l[i]]
            workspace.c[i] = workspace.gvec[l[i]]
        end

        gradproj = 0.0
        @inbounds for i in 1:n
            s = 0.0
            @simd for j in 1:i
                s -= workspace.B[i, j] * workspace.gvec[l[j]]
            end
            @simd for j in (i+1):n
                s -= workspace.B[j, i] * workspace.gvec[l[j]]
            end
            workspace.t[i] = s
            gradproj += s * workspace.gvec[l[i]]
        end

        if gradproj < 0.0
            steplength = 1.0
            accpoint = false
            count = n
            while true
                count = 0
                @inbounds for i in 1:n
                    b[l[i]] = workspace.X[i] + steplength * workspace.t[i]
                    if RELTEST + workspace.X[i] == RELTEST + b[l[i]]
                        count += 1
                    end
                end
                if count < n
                    fval = f(n0, b, ex)
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
                gfunc(n0, b, workspace.gvec, ex)
                gradcount += 1
                iter += 1

                D1 = 0.0
                @inbounds for i in 1:n
                    workspace.t[i] = steplength * workspace.t[i]
                    workspace.c[i] = workspace.gvec[l[i]] - workspace.c[i]
                    D1 += workspace.t[i] * workspace.c[i]
                end

                if D1 > 0.0
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
            println("iter", lpad(iter, 4), " value ", fval)
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
        if iter < maxit
            println("converged")
        else
            println("stopped after ", iter, " iterations")
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