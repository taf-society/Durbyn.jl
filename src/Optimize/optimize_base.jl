"""
    OptimizeResult

Result of an optimization run.

# Fields
- `minimizer::Vector{Float64}` — Parameter vector at the optimum
- `minimum::Float64` — Objective function value at the optimum
- `converged::Bool` — Whether the optimizer converged
- `iterations::Int` — Number of iterations performed
- `f_calls::Int` — Number of objective function evaluations (0 if unavailable)
- `g_calls::Int` — Number of gradient evaluations (0 if unavailable)
- `message::String` — Convergence or diagnostic message
"""
struct OptimizeResult
    minimizer::Vector{Float64}
    minimum::Float64
    converged::Bool
    iterations::Int
    f_calls::Int
    g_calls::Int
    message::String
end


# --- Helpers ---

_to_scalar(val::Number) = Float64(val)
_to_scalar(::Nothing) = throw(ArgumentError("objective function returned nothing instead of a scalar"))
_to_scalar(::Missing) = throw(ArgumentError("objective function returned missing instead of a scalar"))
function _to_scalar(val)
    length(val) == 1 || throw(ArgumentError("objective function returned $(length(val)) values instead of 1"))
    Float64(first(val))
end

function _repeat_to_length(x::Vector{Float64}, n::Int)
    lx = length(x)
    lx == n && return x
    lx == 0 && return fill(NaN, n)
    return Float64[x[mod1(i, lx)] for i in 1:n]
end


"""
    optimize(fn, x0, method=:nelder_mead; kwargs...)

Minimize a scalar-valued function using the specified optimization method.

# Arguments

- `fn::Function`: Objective function to minimize, called as `fn(x; kwargs...)`.
- `x0::AbstractVector{<:Real}`: Initial parameter vector.
- `method::Symbol=:nelder_mead`: Optimization method:
  - `:nelder_mead` — derivative-free simplex
  - `:bfgs` — quasi-Newton with Strong Wolfe line search
  - `:lbfgsb` — limited-memory BFGS with box constraints
  - `:brent` — 1D optimization (scalar `x0` only)

# Keyword Arguments

- `gradient::Union{Function,Nothing}=nothing`: Gradient function `gradient(x; kwargs...) → Vector`.
  If `nothing`, numerical gradients are computed automatically.
- `lower`, `upper`: Bounds for `:lbfgsb` and `:brent` methods.
- `max_iterations::Int`: Maximum iterations (default: 500 for Nelder-Mead, 100 otherwise).
- `param_scale::Union{Nothing,Vector{Float64}}=nothing`: Parameter scaling factors.
- `step_sizes::Union{Nothing,Vector{Float64}}=nothing`: Step sizes for numerical differentiation.
- `fn_scale::Float64=1.0`: Function scaling (negative to maximize).
- `trace::Int=0`: Verbosity level (0=silent).
- `report_interval::Int=10`: Reporting frequency when trace > 0.
- `reltol::Float64=√eps`: Relative convergence tolerance.
- `abstol::Float64=-Inf`: Absolute convergence tolerance.
- `gtol::Float64=0.0`: Gradient norm tolerance.
- `factr::Float64=1e7`: L-BFGS-B function tolerance factor.
- `pgtol::Float64=0.0`: L-BFGS-B projected gradient tolerance.
- `alpha::Float64=1.0`: Nelder-Mead reflection coefficient.
- `beta::Float64=0.5`: Nelder-Mead contraction coefficient.
- `gamma::Float64=2.0`: Nelder-Mead expansion coefficient.
- `memory_size::Int=5`: L-BFGS-B memory size.
- `hessian::Bool=false`: If `true`, compute Hessian at solution.
- `kwargs...`: Additional arguments passed to `fn` and `gradient`.

# Returns

An `OptimizeResult` with fields: `minimizer`, `minimum`, `converged`, `iterations`,
`f_calls`, `g_calls`, `message`.

If `hessian=true`, returns a `NamedTuple` with an additional `hessian` field.

# Examples

```julia
using Durbyn.Optimize

rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
result = optimize(rosenbrock, [-1.2, 1.0])
result = optimize(rosenbrock, [-1.2, 1.0], :bfgs)
result = optimize(rosenbrock, [0.5, 0.5], :lbfgsb; lower=[0.0, 0.0], upper=[2.0, 2.0])
```
"""
function optimize(fn::Function, x0::AbstractVector{<:Real},
               method::Symbol=:nelder_mead;
               gradient::Union{Function,Nothing}=nothing,
               lower::Union{Real,AbstractVector{<:Real}}=-Inf,
               upper::Union{Real,AbstractVector{<:Real}}=Inf,
               max_iterations::Union{Int,Nothing}=nothing,
               param_scale::Union{Nothing,AbstractVector{<:Real}}=nothing,
               step_sizes::Union{Nothing,AbstractVector{<:Real}}=nothing,
               fn_scale::Float64=1.0,
               trace::Int=0,
               report_interval::Int=10,
               reltol::Float64=sqrt(eps(Float64)),
               abstol::Float64=-Inf,
               gtol::Float64=0.0,
               factr::Float64=1e7,
               pgtol::Float64=0.0,
               alpha::Float64=1.0,
               beta::Float64=0.5,
               gamma::Float64=2.0,
               memory_size::Int=5,
               hessian::Bool=false,
               kwargs...)

    # Fast path for Brent — bypass all wrapper overhead
    if method === :brent && !hessian
        length(x0) == 1 || throw(ArgumentError("Brent method requires exactly one parameter"))
        _lb = lower isa AbstractVector ? (length(lower) >= 1 ? Float64(lower[1]) : -Inf) : Float64(lower)
        _ub = upper isa AbstractVector ? (length(upper) >= 1 ? Float64(upper[1]) : Inf) : Float64(upper)
        (isfinite(_lb) && isfinite(_ub)) || throw(ArgumentError("method = :brent requires finite 'lower' and 'upper' bounds"))
        _lb < _ub || throw(ArgumentError("lower bound must be strictly less than upper bound"))

        _maxit_b = isnothing(max_iterations) ? 1000 : max_iterations
        _ps = isnothing(param_scale) ? 1.0 : Float64(first(param_scale))
        if isempty(kwargs)
            return _brent_fast(fn, _lb, _ub, fn_scale, reltol, trace > 0, _maxit_b, _ps)
        else
            return _brent_fast_kw(fn, _lb, _ub, fn_scale, reltol, trace > 0, _maxit_b, _ps, kwargs)
        end
    end

    x0 = Float64.(x0)
    lower = lower isa AbstractVector ? Float64.(lower) : Float64(lower)
    upper = upper isa AbstractVector ? Float64.(upper) : Float64(upper)
    npar = length(x0)

    # Validate method
    valid_methods = [:nelder_mead, :bfgs, :lbfgsb, :brent]
    _check_arg(method, valid_methods, "method")

    # Resolve default max_iterations (method-dependent)
    maxit = if isnothing(max_iterations)
        method === :nelder_mead ? 500 : 100
    else
        max_iterations
    end

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

    # Warn about unreliable 1D Nelder-Mead
    if method === :nelder_mead && npar == 1
        @warn "one-dimensional optimization by Nelder-Mead is unreliable: use :brent instead"
    end

    # Resolve param_scale
    parscale = if isnothing(param_scale)
        ones(npar)
    else
        ps = Float64.(param_scale)
        length(ps) == npar || throw(ArgumentError("param_scale must have length $npar"))
        ps
    end

    # Resolve step_sizes
    ndeps = if isnothing(step_sizes)
        fill(1e-3, npar)
    else
        nd = Float64.(step_sizes)
        if (method === :bfgs || method === :lbfgsb) && length(nd) != npar
            throw(ArgumentError("step_sizes must have length $npar"))
        end
        nd
    end

    if trace < 0
        @warn "trace should be non-negative"
    end

    # Build scaled objective and gradient
    fn_scaled, grad_scaled = _build_scaled_fns(fn, gradient, npar, parscale, fn_scale; kwargs...)
    x0_scaled = x0 ./ parscale

    # Dispatch to solver
    result = _run_solver(method, x0_scaled, fn_scaled, grad_scaled,
                         parscale, fn_scale, ndeps, maxit, abstol, reltol, gtol,
                         alpha, beta, gamma, trace, report_interval,
                         memory_size, factr, pgtol,
                         lower_vec, upper_vec)

    # Compute Hessian at solution if requested
    if hessian
        hess = _compute_hessian(result.minimizer, fn, gradient, parscale, fn_scale, ndeps; kwargs...)
        return (
            minimizer = result.minimizer,
            minimum = result.minimum,
            converged = result.converged,
            iterations = result.iterations,
            f_calls = result.f_calls,
            g_calls = result.g_calls,
            message = result.message,
            hessian = hess
        )
    end

    return result
end


"""Build scaled objective and gradient closures that apply parscale/fnscale transformations."""
function _build_scaled_fns(fn, grad, npar::Int, parscale::Vector{Float64}, fnscale::Float64; kwargs...)
    needs_parscale = any(p -> p != 1.0, parscale)
    x_scaled_fn = needs_parscale ? Vector{Float64}(undef, npar) : Float64[]
    x_scaled_grad = needs_parscale ? Vector{Float64}(undef, npar) : Float64[]

    scale_params! = function (dst::Vector{Float64}, x::AbstractVector, scale::Vector{Float64})
        @inbounds @simd for i in eachindex(dst, x, scale)
            dst[i] = x[i] * scale[i]
        end
        return dst
    end

    fn_scaled = if needs_parscale && fnscale != 1.0
        x -> begin
            scale_params!(x_scaled_fn, x, parscale)
            _to_scalar(fn(x_scaled_fn; kwargs...)) / fnscale
        end
    elseif needs_parscale
        x -> begin
            scale_params!(x_scaled_fn, x, parscale)
            _to_scalar(fn(x_scaled_fn; kwargs...))
        end
    elseif fnscale != 1.0
        x -> _to_scalar(fn(x; kwargs...)) / fnscale
    else
        x -> _to_scalar(fn(x; kwargs...))
    end

    grad_scaled = if !isnothing(grad)
        g_buffer = Vector{Float64}(undef, npar)
        validate_grad! = (dst, g) -> begin
            (g isa AbstractVector && length(g) == npar) ||
                throw(ArgumentError("gradient returned length $(g isa AbstractVector ? length(g) : 0), expected $npar"))
            copyto!(dst, g)
        end
        if needs_parscale && fnscale != 1.0
            inv_fnscale = inv(fnscale)
            x -> begin
                scale_params!(x_scaled_grad, x, parscale)
                raw = grad(x_scaled_grad; kwargs...)
                validate_grad!(g_buffer, raw)
                @inbounds @simd for i in eachindex(g_buffer)
                    g_buffer[i] *= parscale[i] * inv_fnscale
                end
                g_buffer
            end
        elseif needs_parscale
            x -> begin
                scale_params!(x_scaled_grad, x, parscale)
                raw = grad(x_scaled_grad; kwargs...)
                validate_grad!(g_buffer, raw)
                @inbounds @simd for i in eachindex(g_buffer)
                    g_buffer[i] *= parscale[i]
                end
                g_buffer
            end
        elseif fnscale != 1.0
            inv_fnscale = inv(fnscale)
            x -> begin
                raw = grad(x; kwargs...)
                validate_grad!(g_buffer, raw)
                @inbounds @simd for i in eachindex(g_buffer)
                    g_buffer[i] *= inv_fnscale
                end
                g_buffer
            end
        else
            x -> begin
                raw = grad(x; kwargs...)
                validate_grad!(g_buffer, raw)
                g_buffer
            end
        end
    else
        nothing
    end

    return fn_scaled, grad_scaled
end


# --- Solver dispatch ---

function _run_solver(method::Symbol, x0_scaled, fn_scaled, grad_scaled,
                     parscale, fnscale, ndeps, maxit, abstol, reltol, gtol,
                     alpha, beta, gamma, trace, report_interval,
                     memory_size, factr, pgtol,
                     lower_vec, upper_vec)
    if method === :nelder_mead
        _run_nelder_mead(x0_scaled, fn_scaled, parscale, fnscale,
                         maxit, abstol, reltol, alpha, beta, gamma, trace)
    elseif method === :bfgs
        _run_bfgs(x0_scaled, fn_scaled, grad_scaled, parscale, fnscale,
                  ndeps, maxit, abstol, reltol, gtol, trace, report_interval)
    elseif method === :lbfgsb
        _run_lbfgsb(x0_scaled, fn_scaled, grad_scaled, parscale, fnscale,
                    ndeps, maxit, trace, report_interval,
                    memory_size, factr, pgtol,
                    lower_vec, upper_vec)
    elseif method === :brent
        _run_brent(x0_scaled[1], fn_scaled, fnscale, reltol, trace, maxit,
                   lower_vec[1], upper_vec[1])
    end
end


function _run_nelder_mead(par, fn, parscale, fnscale,
                          maxit, abstol, reltol, alpha, beta, gamma, trace)
    opts = NelderMeadOptions(
        abstol = abstol,
        reltol = reltol,
        alpha = alpha,
        beta = beta,
        gamma = gamma,
        trace = trace > 0,
        maxit = maxit
    )
    result = nelder_mead(fn, par, opts)
    return OptimizeResult(
        result.x_opt .* parscale,
        result.f_opt * fnscale,
        result.fail == 0,
        maxit,    # nelder_mead doesn't return iteration count
        result.fncount,
        0,
        result.fail == 0 ? "converged" : "maximum iterations reached"
    )
end


function _run_bfgs(par, fn, gr, parscale, fnscale,
                   ndeps, maxit, abstol, reltol, gtol, trace, report_interval)
    gr_internal = if isnothing(gr)
        nothing
    else
        (grad, x) -> begin
            g_vec = gr(x)
            copyto!(grad, g_vec)
            nothing
        end
    end

    opts = BFGSOptions(
        abstol = abstol,
        reltol = reltol,
        gtol = gtol,
        trace = trace > 0,
        maxit = maxit,
        report_interval = report_interval
    )

    result = bfgs(fn, gr_internal, par; options=opts, step_sizes=ndeps)

    return OptimizeResult(
        result.x_opt .* parscale,
        result.f_opt * fnscale,
        result.fail == 0,
        result.n_iter,
        result.fn_evals,
        result.gr_evals,
        result.fail == 0 ? "converged" : "maximum iterations reached"
    )
end


function _run_lbfgsb(par, fn, gr, parscale, fnscale,
                     ndeps, maxit, trace, report_interval,
                     memory_size, factr, pgtol,
                     lower, upper)
    npar = length(par)

    if npar == 0
        f_val = fn(Float64[])
        return OptimizeResult(
            Float64[],
            f_val * fnscale,
            true,
            0,
            1,
            0,
            "NOTHING TO DO"
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
            error("L-BFGS-B requires a finite objective value; got $(val)")
        end
        val
    end

    gr_internal = if !isnothing(gr)
        x -> begin
            gr_count[] += 1
            gval = gr(x)
            for i in eachindex(gval)
                if !isfinite(gval[i])
                    error("gradient contains non-finite value at index $i")
                end
            end
            gval
        end
    else
        cache = NumericalGradientCache(npar)
        x -> begin
            gr_count[] += 1
            numgrad_bounded!(cache.gradient, cache.x_trial, fn_internal, x, ndeps,
                             lower_scaled, upper_scaled)
            return cache.gradient
        end
    end

    opts = LBFGSBOptions(
        memory_size = memory_size,
        ftol_factor = factr,
        pg_tol = pgtol,
        maxit = maxit,
        print_level = trace > 0 ? report_interval : 0
    )

    result = lbfgsb(fn_internal, gr_internal, par;
                    lower=lower_scaled, upper=upper_scaled, options=opts)

    is_converged = result.fail == 0
    msg = if result.fail == 0
        isnothing(result.message) ? "converged" : result.message
    elseif result.fail == 52
        isnothing(result.message) ? "no feasible solution" : result.message
    elseif result.fail == 1 && result.n_iter >= opts.maxit
        "maximum iterations reached"
    else
        isnothing(result.message) ? "abnormal termination" : result.message
    end

    return OptimizeResult(
        result.x_opt .* parscale,
        result.f_opt * fnscale,
        is_converged,
        result.n_iter,
        result.fn_evals,
        result.gr_evals,
        msg
    )
end


function _run_brent(par, fn, fnscale, reltol, trace, maxit, lower, upper)
    opts = BrentOptions(
        tol = reltol,
        trace = trace > 0,
        maxit = maxit
    )

    result = brent(fn, lower, upper; options=opts)

    return OptimizeResult(
        [result.x_opt],
        result.f_opt * fnscale,
        result.fail == 0,
        result.n_iter,
        result.fn_evals,
        0,
        result.fail == 0 ? "converged" : "maximum iterations reached"
    )
end


"""Function barrier for Brent fast path — Julia specializes on concrete `fn` type.

Passes the scalar `x::Float64` directly to `fn(x)` — works for both scalar-style
(`f(x) = x^2`) and vector-style (`f(x) = x[1]^2`) callbacks because Julia's
`getindex(::Number, 1)` returns the number itself.
"""
function _brent_fast(fn, lb::Float64, ub::Float64, fn_scale::Float64,
                     reltol::Float64, do_trace::Bool, maxit::Int, ps::Float64)
    inv_ps = inv(ps)
    fn_brent = if ps == 1.0 && fn_scale == 1.0
        x -> _to_scalar(fn(x))
    elseif ps == 1.0
        _inv_fns = inv(fn_scale)
        x -> _to_scalar(fn(x)) * _inv_fns
    elseif fn_scale == 1.0
        x -> _to_scalar(fn(x * ps))
    else
        _inv_fns = inv(fn_scale)
        x -> _to_scalar(fn(x * ps)) * _inv_fns
    end
    result = brent(fn_brent, lb * inv_ps, ub * inv_ps;
                   options=BrentOptions(tol=reltol, trace=do_trace, maxit=maxit))
    return OptimizeResult(
        [result.x_opt * ps], result.f_opt * fn_scale,
        result.fail == 0, result.n_iter, result.fn_evals, 0,
        result.fail == 0 ? "converged" : "maximum iterations reached")
end

function _brent_fast_kw(fn, lb::Float64, ub::Float64, fn_scale::Float64,
                        reltol::Float64, do_trace::Bool, maxit::Int,
                        ps::Float64, kw)
    inv_ps = inv(ps)
    fn_brent = if ps == 1.0 && fn_scale == 1.0
        x -> _to_scalar(fn(x; kw...))
    elseif ps == 1.0
        _inv_fns = inv(fn_scale)
        x -> _to_scalar(fn(x; kw...)) * _inv_fns
    elseif fn_scale == 1.0
        x -> _to_scalar(fn(x * ps; kw...))
    else
        _inv_fns = inv(fn_scale)
        x -> _to_scalar(fn(x * ps; kw...)) * _inv_fns
    end
    result = brent(fn_brent, lb * inv_ps, ub * inv_ps;
                   options=BrentOptions(tol=reltol, trace=do_trace, maxit=maxit))
    return OptimizeResult(
        [result.x_opt * ps], result.f_opt * fn_scale,
        result.fail == 0, result.n_iter, result.fn_evals, 0,
        result.fail == 0 ? "converged" : "maximum iterations reached")
end


function _compute_hessian(par, fn, grad, parscale, fnscale, ndeps; kwargs...)
    fn_wrapper(x) = fn(x; kwargs...)
    grad_wrapper = isnothing(grad) ? nothing : (x -> grad(x; kwargs...))
    return numerical_hessian(fn_wrapper, par, grad_wrapper;
                        fnscale=fnscale, parscale=parscale, step_sizes=ndeps)
end
