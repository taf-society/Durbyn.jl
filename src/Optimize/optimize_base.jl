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

    valid_methods = [:nelder_mead, :bfgs, :lbfgsb, :brent]
    _check_arg(method, valid_methods, "method")

    if (any(lower .> -Inf) || any(upper .< Inf)) && !(method === :lbfgsb || method === :brent)
        @warn "finite bounds require method :lbfgsb or :brent; switching to :lbfgsb"
        method = :lbfgsb
    end

    if method === :brent && npar != 1
        throw(ArgumentError("method = :brent is only available for one-dimensional optimization"))
    end

    lower_vec = lower isa Float64 ? fill(lower, npar) : _repeat_to_length(lower, npar)
    upper_vec = upper isa Float64 ? fill(upper, npar) : _repeat_to_length(upper, npar)

    if method === :brent
        if !all(isfinite, lower_vec) || !all(isfinite, upper_vec)
            throw(ArgumentError("method = :brent requires finite 'lower' and 'upper' bounds"))
        end
        if lower_vec[1] >= upper_vec[1]
            throw(ArgumentError("'xmin' not less than 'xmax'"))
        end
    end

    con = Dict{String,Any}(
        "trace" => 0,
        "fnscale" => 1.0,
        "parscale" => ones(npar),
        "ndeps" => fill(1e-3, npar),
        "maxit" => (method === :nelder_mead ? 500 : 100),
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

    control_str = Dict{String,Any}(string(k) => v for (k, v) in control)

    known_keys = keys(con)
    unknown = setdiff(keys(control_str), known_keys)
    if !isempty(unknown)
        @warn "unknown names in control: $(join(unknown, ", "))"
    end

    merge!(con, control_str)

    if con["trace"] < 0
        @warn "trace should be non-negative"
    end

    if method === :brent
        con["parscale"] = ones(npar)
    else
        ps = con["parscale"]
        con["parscale"] = ps isa Number ? fill(Float64(ps), npar) : Float64.(ps)
        if length(con["parscale"]) != npar
            throw(ArgumentError("'parscale' is of the wrong length"))
        end
    end

    nd = con["ndeps"]
    con["ndeps"] = nd isa Number ? fill(Float64(nd), npar) : Float64.(nd)
    if (method === :bfgs || method === :lbfgsb) && isnothing(grad) && length(con["ndeps"]) != npar
        throw(ArgumentError("'ndeps' is of the wrong length"))
    end

    if method === :lbfgsb && any(haskey.(Ref(control), ["reltol", "abstol"]))
        @warn "method :lbfgsb uses 'factr' (and 'pgtol') instead of 'reltol' and 'abstol'"
    end

    if method === :nelder_mead && npar == 1 && con["warn.1d.NelderMead"]
        @warn "one-dimensional optimization by Nelder-Mead is unreliable: use :brent instead"
    end

    fnscale = con["fnscale"]
    parscale = con["parscale"]

    fn_scaled = if any(parscale .!= 1.0)
        if fnscale != 1.0
            x -> _to_scalar(fn(x .* parscale; kwargs...)) / fnscale
        else
            x -> _to_scalar(fn(x .* parscale; kwargs...))
        end
    elseif fnscale != 1.0
        x -> _to_scalar(fn(x; kwargs...)) / fnscale
    else
        x -> _to_scalar(fn(x; kwargs...))
    end

    grad_scaled = if !isnothing(grad)
        _check_grad = g -> begin
            (g isa AbstractVector && length(g) == npar) ||
                throw(ArgumentError("gradient in optimize evaluated to length $(g isa AbstractVector ? length(g) : 0) not $npar"))
            g
        end
        if fnscale != 1.0 || any(parscale .!= 1.0)
            x -> (_check_grad(grad(x .* parscale; kwargs...)) .* parscale) / fnscale
        else
            x -> _check_grad(grad(x; kwargs...))
        end
    else
        nothing
    end

    x0_scaled = x0 ./ parscale

    result = if method === :nelder_mead
        _dispatch_nelder_mead(x0_scaled, fn_scaled, con, lower_vec, upper_vec, parscale)
    elseif method === :bfgs
        _dispatch_bfgs(x0_scaled, fn_scaled, grad_scaled, con, parscale)
    elseif method === :lbfgsb
        _dispatch_lbfgsb(x0_scaled, fn_scaled, grad_scaled, con, lower_vec, upper_vec, parscale)
    elseif method === :brent
        _dispatch_brent(x0_scaled[1], fn_scaled, con, lower_vec[1], upper_vec[1], parscale[1])
    end

    hess = if hessian
        _compute_hessian(result.par, fn, grad, con, parscale; kwargs...)
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


function _dispatch_nelder_mead(par, fn, con, lower, upper, parscale)
    opts = NelderMeadOptions(
        abstol = con["abstol"],
        reltol = con["reltol"],
        alpha = con["alpha"],
        beta = con["beta"],
        gamma = con["gamma"],
        trace = con["trace"] > 0,
        maxit = con["maxit"]
    )

    result = nelder_mead(fn, par, opts)

    return (
        par = result.x_opt .* parscale,
        value = result.f_opt * con["fnscale"],
        fn_evals = result.fncount,
        gr_evals = nothing,
        fail = result.fail,
        message = nothing
    )
end


function _dispatch_bfgs(par, fn, gr, con, parscale)
    gr_internal = isnothing(gr) ? nothing : (grad, x) -> (grad .= gr(x); nothing)

    opts = BFGSOptions(
        abstol = con["abstol"],
        reltol = con["reltol"],
        gtol = con["gtol"],
        trace = con["trace"] > 0,
        maxit = con["maxit"],
        report_interval = con["REPORT"]
    )

    result = bfgs(fn, gr_internal, par;
                  options=opts, step_sizes=con["ndeps"])

    return (
        par = result.x_opt .* parscale,
        value = result.f_opt * con["fnscale"],
        fn_evals = result.fn_evals,
        gr_evals = result.gr_evals,
        fail = result.fail,
        message = nothing
    )
end


function _dispatch_lbfgsb(par, fn, gr, con, lower, upper, parscale)
    npar = length(par)

    if npar == 0
        f_val = fn(Float64[])
        return (
            par = Float64[],
            value = f_val * con["fnscale"],
            fn_evals = 1,
            gr_evals = 0,
            fail = 0,
            message = "NOTHING TO DO"
        )
    end

    lower_scaled = lower ./ parscale
    upper_scaled = upper ./ parscale

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
        npar_local = length(par)
        ndeps = con["ndeps"]
        cache = NumericalGradientCache(npar_local)
        x -> begin
            gr_count[] += 1
            numgrad_bounded!(cache.gradient, cache.x_trial, fn_internal, x, ndeps,
                             lower_scaled, upper_scaled)
            return cache.gradient
        end
    end

    opts = LBFGSBOptions(
        memory_size = con["lmm"],
        ftol_factor = con["factr"],
        pg_tol = con["pgtol"],
        maxit = con["maxit"],
        print_level = con["trace"] > 0 ? con["REPORT"] : 0
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
        par = result.x_opt .* parscale,
        value = result.f_opt * con["fnscale"],
        fn_evals = result.fn_evals,
        gr_evals = result.gr_evals,
        fail = convergence,
        message = result.message
    )
end


function _dispatch_brent(par, fn, con, lower, upper, parscale)
    opts = BrentOptions(
        tol = con["reltol"],
        trace = con["trace"] > 0
    )

    result = brent(fn, lower, upper; options=opts)

    return (
        par = [result.x_opt],
        value = result.f_opt * con["fnscale"],
        fn_evals = nothing,
        gr_evals = nothing,
        fail = 0,
        message = nothing
    )
end


function _compute_hessian(par, fn, grad, con, parscale; kwargs...)
    npar = length(par)
    fnscale = con["fnscale"]
    ndeps = con["ndeps"]

    fn_wrapper(x) = fn(x; kwargs...)
    grad_wrapper = isnothing(grad) ? nothing : (x -> grad(x; kwargs...))

    return numerical_hessian(fn_wrapper, par, grad_wrapper;
                        fnscale=fnscale, parscale=parscale, step_sizes=ndeps)
end
