"""
    optimize(x0, fn; grad=nothing, method=:nelder_mead, lower=-Inf, upper=Inf,
             control=Dict(), hessian=false, kwargs...)

Unified interface for general-purpose optimization.

Dispatches to the appropriate solver based on the `method` argument, handling
parameter/function scaling and returning results in a consistent format.

# Arguments

- `x0::Vector{Float64}`: Initial parameter vector.
- `fn::Function`: Objective function to minimize, called as `fn(x; kwargs...)`.

# Keyword Arguments

- `grad::Union{Function,Nothing}=nothing`: Gradient function, called as `grad(x; kwargs...)`.
  If `nothing`, numerical gradients are computed for methods that need them.
- `method::Symbol=:nelder_mead`: Optimization method:
  - `:nelder_mead` — derivative-free simplex
  - `:bfgs` — quasi-Newton with line search
  - `:lbfgsb` — limited-memory BFGS with box constraints
  - `:brent` — 1D optimization (scalar `x0` only)
- `lower`, `upper`: Bounds for L-BFGS-B and Brent methods.
- `control::Dict`: Control parameters (trace, fnscale, parscale, ndeps, maxit,
  abstol, reltol, gtol, alpha, beta, gamma, REPORT, lmm, factr, pgtol).
- `hessian::Bool`: If `true`, compute Hessian at solution.
- `kwargs...`: Additional arguments passed to `fn` and `grad`.

# Returns

Named tuple with fields:
- `par::Vector{Float64}`: Optimal parameters found
- `value::Float64`: Function value at optimum
- `counts::NamedTuple`: `(function_=n, gradient=m)` evaluation counts
- `convergence::Int`: Status code (0=success, 1=maxit reached)
- `message::Union{String,Nothing}`: Convergence message (method-dependent)
- `hessian::Union{Matrix{Float64},Nothing}`: Hessian at solution (if requested)

# Examples

```julia
using Durbyn.Optimize

rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
rosenbrock_grad(x) = [-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]), 200*(x[2]-x[1]^2)]

result = optimize([-1.2, 1.0], rosenbrock)
result = optimize([-1.2, 1.0], rosenbrock; grad=rosenbrock_grad, method=:bfgs)
result = optimize([0.5, 0.5], rosenbrock; method=:lbfgsb,
                  lower=[0.0, 0.0], upper=[2.0, 2.0])

f1d(x) = (x[1] - 2)^2
result = optimize([0.0], f1d; method=:brent, lower=-5.0, upper=5.0)
```
"""

# --- Helpers ---

_to_scalar(val::Number) = Float64(val)
_to_scalar(::Nothing) = throw(ArgumentError("objective function in optimize evaluates to length 0 not 1"))
_to_scalar(::Missing) = throw(ArgumentError("objective function in optimize evaluates to length 0 not 1"))
function _to_scalar(val)
    length(val) == 1 || throw(ArgumentError("objective function in optimize evaluates to length $(length(val)) not 1"))
    Float64(first(val))
end

function _repeat_to_length(x::Vector{Float64}, n::Int)
    lx = length(x)
    lx == n && return x
    lx == 0 && return fill(NaN, n)
    return Float64[x[mod1(i, lx)] for i in 1:n]
end


# --- Internal typed configuration (replaces string-keyed Dict for internal use) ---

struct _OptimConfig
    trace::Int
    fnscale::Float64
    parscale::Vector{Float64}
    ndeps::Vector{Float64}
    maxit::Int
    abstol::Float64
    reltol::Float64
    gtol::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    report_interval::Int
    memory_size::Int
    ftol_factor::Float64
    pg_tol::Float64
    warn_1d_nm::Bool
end

# Default values for each control key
const _CONTROL_DEFAULTS = Dict{String,Any}(
    "trace" => 0,
    "fnscale" => 1.0,
    "parscale" => :auto,    # sentinel — will be filled to ones(npar)
    "ndeps" => :auto,       # sentinel — will be filled to fill(1e-3, npar)
    "maxit" => :auto,       # sentinel — method-dependent
    "abstol" => -Inf,
    "reltol" => sqrt(eps(Float64)),
    "gtol" => 0.0,
    "alpha" => 1.0,
    "beta" => 0.5,
    "gamma" => 2.0,
    "REPORT" => 10,
    "lmm" => 5,
    "factr" => 1e7,
    "pgtol" => 0.0,
    "warn.1d.NelderMead" => true
)

"""Parse user-supplied control Dict into typed _OptimConfig, validating and warning as needed."""
function _parse_control(control::Dict, npar::Int, method::Symbol)
    # Stringify keys for uniform handling
    merged = Dict{String,Any}(string(k) => v for (k, v) in _CONTROL_DEFAULTS)
    user_str = Dict{String,Any}(string(k) => v for (k, v) in control)

    unknown = setdiff(keys(user_str), keys(merged))
    if !isempty(unknown)
        @warn "unknown names in control: $(join(unknown, ", "))"
    end

    merge!(merged, user_str)

    # Resolve sentinels
    if merged["parscale"] === :auto
        merged["parscale"] = ones(npar)
    end
    if merged["ndeps"] === :auto
        merged["ndeps"] = fill(1e-3, npar)
    end
    if merged["maxit"] === :auto
        merged["maxit"] = (method === :nelder_mead ? 500 : 100)
    end

    trace_val = merged["trace"]
    if trace_val < 0
        @warn "trace should be non-negative"
    end

    # Validate and normalize parscale
    ps = merged["parscale"]
    if method !== :brent
        ps = ps isa Number ? fill(Float64(ps), npar) : Float64.(ps)
        if length(ps) != npar
            throw(ArgumentError("parscale must have length $npar"))
        end
    else
        ps = ones(npar)
    end

    # Validate and normalize ndeps
    nd = merged["ndeps"]
    nd = nd isa Number ? fill(Float64(nd), npar) : Float64.(nd)
    if (method === :bfgs || method === :lbfgsb) && length(nd) != npar
        throw(ArgumentError("ndeps must have length $npar"))
    end

    if method === :lbfgsb && any(haskey.(Ref(control), ["reltol", "abstol"]))
        @warn "method :lbfgsb uses 'factr' (and 'pgtol') instead of 'reltol' and 'abstol'"
    end

    return _OptimConfig(
        Int(trace_val),
        Float64(merged["fnscale"]),
        ps,
        nd,
        Int(merged["maxit"]),
        Float64(merged["abstol"]),
        Float64(merged["reltol"]),
        Float64(merged["gtol"]),
        Float64(merged["alpha"]),
        Float64(merged["beta"]),
        Float64(merged["gamma"]),
        Int(merged["REPORT"]),
        Int(merged["lmm"]),
        Float64(merged["factr"]),
        Float64(merged["pgtol"]),
        Bool(merged["warn.1d.NelderMead"]),
    )
end


"""Build scaled objective and gradient closures that apply parscale/fnscale transformations."""
function _build_scaled_fns(fn, grad, npar::Int, cfg::_OptimConfig; kwargs...)
    fnscale = cfg.fnscale
    parscale = cfg.parscale
    needs_parscale = any(parscale .!= 1.0)

    # Objective: f_scaled(x_scaled) = fn(x_scaled .* parscale; kwargs...) / fnscale
    fn_scaled = if needs_parscale && fnscale != 1.0
        x -> _to_scalar(fn(x .* parscale; kwargs...)) / fnscale
    elseif needs_parscale
        x -> _to_scalar(fn(x .* parscale; kwargs...))
    elseif fnscale != 1.0
        x -> _to_scalar(fn(x; kwargs...)) / fnscale
    else
        x -> _to_scalar(fn(x; kwargs...))
    end

    # Gradient: g_scaled(x_scaled) = grad(x_scaled .* parscale; kwargs...) .* parscale / fnscale
    grad_scaled = if !isnothing(grad)
        validate_grad = g -> begin
            (g isa AbstractVector && length(g) == npar) ||
                throw(ArgumentError("gradient evaluated to length $(g isa AbstractVector ? length(g) : 0), expected $npar"))
            g
        end
        if fnscale != 1.0 || needs_parscale
            x -> (validate_grad(grad(x .* parscale; kwargs...)) .* parscale) / fnscale
        else
            x -> validate_grad(grad(x; kwargs...))
        end
    else
        nothing
    end

    return fn_scaled, grad_scaled
end


# --- Main entry point ---

function optimize(x0::AbstractVector{<:Real}, fn::Function;
               grad::Union{Function,Nothing}=nothing,
               method::Symbol=:nelder_mead,
               lower::Union{Real,AbstractVector{<:Real}}=-Inf,
               upper::Union{Real,AbstractVector{<:Real}}=Inf,
               control::Dict=Dict(),
               hessian::Bool=false,
               kwargs...)

    x0 = Float64.(x0)
    lower = lower isa AbstractVector ? Float64.(lower) : Float64(lower)
    upper = upper isa AbstractVector ? Float64.(upper) : Float64(upper)
    npar = length(x0)

    # Validate method
    valid_methods = [:nelder_mead, :bfgs, :lbfgsb, :brent]
    _check_arg(method, valid_methods, "method")

    # Auto-switch to L-BFGS-B when bounds are specified with an incompatible method
    if (any(lower .> -Inf) || any(upper .< Inf)) && !(method === :lbfgsb || method === :brent)
        @warn "finite bounds require method :lbfgsb or :brent; switching to :lbfgsb"
        method = :lbfgsb
    end

    # Brent requires exactly 1 parameter
    if method === :brent && npar != 1
        throw(ArgumentError("Brent method requires exactly one parameter"))
    end

    # Expand scalar bounds to vectors
    lower_vec = lower isa Float64 ? fill(lower, npar) : _repeat_to_length(lower, npar)
    upper_vec = upper isa Float64 ? fill(upper, npar) : _repeat_to_length(upper, npar)

    # Brent requires finite bounds with lower < upper
    if method === :brent
        if !all(isfinite, lower_vec) || !all(isfinite, upper_vec)
            throw(ArgumentError("method = :brent requires finite 'lower' and 'upper' bounds"))
        end
        if lower_vec[1] >= upper_vec[1]
            throw(ArgumentError("lower bound must be strictly less than upper bound"))
        end
    end

    # Parse control parameters into typed config
    cfg = _parse_control(control, npar, method)

    # Warn about unreliable 1D Nelder-Mead
    if method === :nelder_mead && npar == 1 && cfg.warn_1d_nm
        @warn "one-dimensional optimization by Nelder-Mead is unreliable: use :brent instead"
    end

    # Warn about gradient-method ndeps mismatch (non-analytical gradient only)
    if (method === :bfgs || method === :lbfgsb) && isnothing(grad) && length(cfg.ndeps) != npar
        throw(ArgumentError("ndeps must have length $npar"))
    end

    # Build scaled objective and gradient
    fn_scaled, grad_scaled = _build_scaled_fns(fn, grad, npar, cfg; kwargs...)
    x0_scaled = x0 ./ cfg.parscale

    # Dispatch to solver
    result = _run_solver(method, x0_scaled, fn_scaled, grad_scaled, cfg, lower_vec, upper_vec)

    # Compute Hessian at solution if requested
    hess = if hessian
        _compute_hessian(result.par, fn, grad, cfg; kwargs...)
    else
        nothing
    end

    return (
        par = result.par,
        value = result.value,
        counts = (function_=result.fn_evals, gradient=result.gr_evals),
        convergence = result.fail,
        message = result.message,
        hessian = hess
    )
end


# --- Solver dispatch ---

function _run_solver(method::Symbol, x0_scaled, fn_scaled, grad_scaled, cfg::_OptimConfig,
                     lower_vec, upper_vec)
    if method === :nelder_mead
        _run_nelder_mead(x0_scaled, fn_scaled, cfg)
    elseif method === :bfgs
        _run_bfgs(x0_scaled, fn_scaled, grad_scaled, cfg)
    elseif method === :lbfgsb
        _run_lbfgsb(x0_scaled, fn_scaled, grad_scaled, cfg, lower_vec, upper_vec)
    elseif method === :brent
        _run_brent(x0_scaled[1], fn_scaled, cfg, lower_vec[1], upper_vec[1])
    end
end


function _run_nelder_mead(par, fn, cfg::_OptimConfig)
    opts = NelderMeadOptions(
        abstol = cfg.abstol,
        reltol = cfg.reltol,
        alpha = cfg.alpha,
        beta = cfg.beta,
        gamma = cfg.gamma,
        trace = cfg.trace > 0,
        maxit = cfg.maxit
    )
    result = nelder_mead(fn, par, opts)
    return (
        par = result.x_opt .* cfg.parscale,
        value = result.f_opt * cfg.fnscale,
        fn_evals = result.fncount,
        gr_evals = nothing,
        fail = result.fail,
        message = nothing
    )
end


function _run_bfgs(par, fn, gr, cfg::_OptimConfig)
    gr_internal = isnothing(gr) ? nothing : (grad, x) -> (grad .= gr(x); nothing)

    opts = BFGSOptions(
        abstol = cfg.abstol,
        reltol = cfg.reltol,
        gtol = cfg.gtol,
        trace = cfg.trace > 0,
        maxit = cfg.maxit,
        report_interval = cfg.report_interval
    )

    result = bfgs(fn, gr_internal, par; options=opts, step_sizes=cfg.ndeps)

    return (
        par = result.x_opt .* cfg.parscale,
        value = result.f_opt * cfg.fnscale,
        fn_evals = result.fn_evals,
        gr_evals = result.gr_evals,
        fail = result.fail,
        message = nothing
    )
end


function _run_lbfgsb(par, fn, gr, cfg::_OptimConfig, lower, upper)
    npar = length(par)

    if npar == 0
        f_val = fn(Float64[])
        return (
            par = Float64[],
            value = f_val * cfg.fnscale,
            fn_evals = 1,
            gr_evals = 0,
            fail = 0,
            message = "NOTHING TO DO"
        )
    end

    lower_scaled = lower ./ cfg.parscale
    upper_scaled = upper ./ cfg.parscale

    fn_count = Ref(0)
    gr_count = Ref(0)

    fn_internal = x -> begin
        fn_count[] += 1
        val = fn(x)
        if !isfinite(val)
            error("L-BFGS-B needs finite values of 'fn'")
        end
        val
    end

    gr_internal = if !isnothing(gr)
        x -> begin
            gr_count[] += 1
            gval = gr(x)
            for i in eachindex(gval)
                if !isfinite(gval[i])
                    error("non-finite value supplied by optimize")
                end
            end
            gval
        end
    else
        ndeps = cfg.ndeps
        cache = NumericalGradientCache(npar)
        x -> begin
            gr_count[] += 1
            numgrad_bounded!(cache.gradient, cache.x_trial, fn_internal, x, ndeps,
                             lower_scaled, upper_scaled)
            return cache.gradient
        end
    end

    opts = LBFGSBOptions(
        memory_size = cfg.memory_size,
        ftol_factor = cfg.ftol_factor,
        pg_tol = cfg.pg_tol,
        maxit = cfg.maxit,
        print_level = cfg.trace > 0 ? cfg.report_interval : 0
    )

    result = lbfgsb(fn_internal, gr_internal, par;
                    lower=lower_scaled, upper=upper_scaled, options=opts)

    convergence = if result.fail == 0
        0
    elseif result.fail == 52
        52
    elseif result.fail == 1 && result.n_iter >= opts.maxit
        1
    else
        51
    end

    return (
        par = result.x_opt .* cfg.parscale,
        value = result.f_opt * cfg.fnscale,
        fn_evals = result.fn_evals,
        gr_evals = result.gr_evals,
        fail = convergence,
        message = result.message
    )
end


function _run_brent(par, fn, cfg::_OptimConfig, lower, upper)
    opts = BrentOptions(
        tol = cfg.reltol,
        trace = cfg.trace > 0
    )

    result = brent(fn, lower, upper; options=opts)

    return (
        par = [result.x_opt],
        value = result.f_opt * cfg.fnscale,
        fn_evals = nothing,
        gr_evals = nothing,
        fail = 0,
        message = nothing
    )
end


function _compute_hessian(par, fn, grad, cfg::_OptimConfig; kwargs...)
    fn_wrapper(x) = fn(x; kwargs...)
    grad_wrapper = isnothing(grad) ? nothing : (x -> grad(x; kwargs...))
    return numerical_hessian(fn_wrapper, par, grad_wrapper;
                        fnscale=cfg.fnscale, parscale=cfg.parscale, step_sizes=cfg.ndeps)
end
