"""
    optim(par, fn; gr=nothing, method="Nelder-Mead", lower=-Inf, upper=Inf,
          control=Dict(), hessian=false, kwargs...)

General-purpose optimization interface matching R's `optim()` function.

# Arguments

- `par::Vector{Float64}`: Initial parameter vector
- `fn::Function`: Objective function to minimize. Called as `fn(par; kwargs...)`

# Keyword Arguments

- `gr::Union{Function,Nothing}=nothing`: Gradient function. Called as `gr(par; kwargs...)`.
  If `nothing`, numerical gradients will be computed for methods that need them.

- `method::String="Nelder-Mead"`: Optimization method. Options:
  - `"Nelder-Mead"`: Nelder-Mead simplex (derivative-free)
  - `"BFGS"`: Quasi-Newton method (requires gradient or uses numerical)
  - `"L-BFGS-B"`: Limited-memory BFGS with box constraints
  - `"Brent"`: Brent's method for 1D optimization (requires scalar par)

- `lower::Union{Float64,Vector{Float64}}=-Inf`: Lower bounds (for L-BFGS-B and Brent)

- `upper::Union{Float64,Vector{Float64}}=Inf`: Upper bounds (for L-BFGS-B and Brent)

- `control::Dict`: Control parameters. Supported keys:
  - `trace::Int=0`: Verbosity level (0=silent, >0=verbose)
  - `fnscale::Float64=1.0`: Scaling for objective (fn will be divided by this)
  - `parscale::Vector{Float64}`: Parameter scaling (default: ones)
  - `ndeps::Vector{Float64}`: Step sizes for numerical derivatives (default: 1e-3)
  - `maxit::Int`: Maximum iterations (default: method-dependent)
  - `abstol::Float64`: Absolute convergence tolerance
  - `reltol::Float64`: Relative convergence tolerance (default: √eps)
  - `gtol::Float64=0.0`: **Julia enhancement** - Gradient norm tolerance for BFGS.
    When gtol > 0, convergence is declared if ||∇f(x)|| < gtol * max(1, |f(x)|).
    Based on first-order optimality condition. Recommended: 1e-5 to 1e-8.
  - `alpha::Float64=1.0`: Nelder-Mead reflection coefficient
  - `beta::Float64=0.5`: Nelder-Mead contraction coefficient
  - `gamma::Float64=2.0`: Nelder-Mead expansion coefficient
  - `REPORT::Int=10`: Reporting frequency for BFGS
  - `lmm::Int=5`: L-BFGS-B memory parameter
  - `factr::Float64=1e7`: L-BFGS-B tolerance factor
  - `pgtol::Float64=0.0`: L-BFGS-B projected gradient tolerance
  - `type::Int=1`: CG type (reserved for future CG implementation)

- `hessian::Bool=false`: If `true`, compute Hessian at solution

- `kwargs...`: Additional arguments passed to `fn` and `gr`

# Returns

Named tuple with fields:
- `par::Vector{Float64}`: Optimal parameters found
- `value::Float64`: Function value at optimum
- `counts::NamedTuple`: Function and gradient evaluation counts
- `convergence::Int`: Convergence code (0=success, 1=maxit reached)
- `message::Union{String,Nothing}`: Convergence message (method-dependent)
- `hessian::Union{Matrix{Float64},Nothing}`: Hessian at solution (if requested)

# Examples

```julia
# Rosenbrock function
rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
rosenbrock_grad(x) = [-400*x[1]*(x[2]-x[1]^2) - 2*(1-x[1]), 200*(x[2]-x[1]^2)]

# Nelder-Mead (no gradient needed)
result = optim([-1.2, 1.0], rosenbrock)

# BFGS with analytical gradient
result = optim([-1.2, 1.0], rosenbrock; gr=rosenbrock_grad, method="BFGS")

# BFGS with numerical gradient
result = optim([-1.2, 1.0], rosenbrock; method="BFGS")

# L-BFGS-B with bounds
result = optim([0.0, 0.0], rosenbrock; method="L-BFGS-B",
               lower=[0.0, 0.0], upper=[2.0, 2.0])

# With control parameters
result = optim([-1.2, 1.0], rosenbrock; method="BFGS",
               control=Dict("trace" => 1, "maxit" => 500))

# With Hessian
result = optim([-1.2, 1.0], rosenbrock; gr=rosenbrock_grad,
               method="BFGS", hessian=true)

# 1D optimization with Brent
f1d(x) = (x - 2)^2
result = optim([0.0], f1d; method="Brent", lower=-5.0, upper=5.0)
```

# Notes

This function provides a unified interface matching R's `optim()`. Method-specific
implementations are in separate files (nmmin.jl, bfgs.jl, lbfgsbmin.jl, fmin.jl).

Parameter and function scaling follow R's convention:
- Internal optimization uses `par/parscale`
- Function values are scaled by `1/fnscale`
- This improves conditioning when parameters have different scales
"""
# R's rep_len: recycle or truncate a vector to length n.
# Empty input → NaN fill (matches R's rep_len(numeric(0), n) → NA vector).
function _rep_len(x::Vector{Float64}, n::Int)
    lx = length(x)
    lx == n && return x
    lx == 0 && return fill(NaN, n)
    return Float64[x[mod1(i, lx)] for i in 1:n]
end

function optim(par::AbstractVector{<:Real}, fn::Function;
               gr::Union{Function,Nothing}=nothing,
               method::String="Nelder-Mead",
               lower::Union{Real,AbstractVector{<:Real}}=-Inf,
               upper::Union{Real,AbstractVector{<:Real}}=Inf,
               control::Dict=Dict(),
               hessian::Bool=false,
               kwargs...)

    # Coerce to Float64 (R's as.double)
    par = Float64.(par)
    lower = lower isa AbstractVector ? Float64.(lower) : Float64(lower)
    upper = upper isa AbstractVector ? Float64.(upper) : Float64(upper)

    npar = length(par)

    # Validate method
    valid_methods = ["Nelder-Mead", "BFGS", "L-BFGS-B", "Brent"]
    if !(method in valid_methods)
        error("method must be one of: $(join(valid_methods, ", "))")
    end

    # Handle bounds
    if (any(lower .> -Inf) || any(upper .< Inf)) && !(method in ["L-BFGS-B", "Brent"])
        @warn "bounds can only be used with method L-BFGS-B or Brent, switching to L-BFGS-B"
        method = "L-BFGS-B"
    end

    # Check 1D for Brent
    if method == "Brent" && npar != 1
        error("method = \"Brent\" is only available for one-dimensional optimization")
    end

    # Expand bounds to vectors first (R uses rep_len; empty → NaN like R's NA)
    lower_vec = lower isa Float64 ? fill(lower, npar) : _rep_len(lower, npar)
    upper_vec = upper isa Float64 ? fill(upper, npar) : _rep_len(upper, npar)

    # Brent requires finite bounds (matching R's error).
    # Checked after expansion so empty vectors become NaN → caught here.
    if method == "Brent"
        if !all(isfinite, lower_vec) || !all(isfinite, upper_vec)
            error("method = \"Brent\" requires finite 'lower' and 'upper' bounds")
        end
        # R's optimize.c:267 checks xmin < xmax
        if lower_vec[1] >= upper_vec[1]
            error("'xmin' not less than 'xmax'")
        end
    end

    # Default control parameters (matching R, with Julia enhancements)
    con = Dict{String,Any}(
        "trace" => 0,
        "fnscale" => 1.0,
        "parscale" => ones(npar),
        "ndeps" => fill(1e-3, npar),
        "maxit" => (method == "Nelder-Mead" ? 500 : 100),
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
        "type" => 1,
        "warn.1d.NelderMead" => true
    )

    # Validate control parameters before merging
    known_keys = keys(con)
    unknown = setdiff(keys(control), known_keys)
    if !isempty(unknown)
        @warn "unknown names in control: $(join(unknown, ", "))"
    end

    # Override with user control
    merge!(con, control)

    if con["trace"] < 0
        @warn "read the documentation for 'trace' more carefully"
    end

    # Validate parscale and ndeps length (R's optim.c errors on mismatch)
    ps = con["parscale"]
    con["parscale"] = ps isa Number ? fill(Float64(ps), npar) : Float64.(ps)
    if length(con["parscale"]) != npar
        error("'parscale' is of the wrong length")
    end
    nd = con["ndeps"]
    con["ndeps"] = nd isa Number ? fill(Float64(nd), npar) : Float64.(nd)
    if length(con["ndeps"]) != npar
        error("'ndeps' is of the wrong length")
    end

    # Warnings for method-specific parameters
    if method == "L-BFGS-B" && any(haskey.(Ref(control), ["reltol", "abstol"]))
        @warn "method L-BFGS-B uses 'factr' (and 'pgtol') instead of 'reltol' and 'abstol'"
    end

    if method == "Nelder-Mead" && npar == 1 && con["warn.1d.NelderMead"]
        @warn "one-dimensional optimization by Nelder-Mead is unreliable: use \"Brent\" instead"
    end

    # Create scaled function wrappers
    fnscale = con["fnscale"]
    parscale = con["parscale"]

    fn_scaled = if fnscale != 1.0 || any(parscale .!= 1.0)
        x -> begin
            val = fn(x .* parscale; kwargs...)
            # R's optim.c:82-84 validates objective returns scalar
            val isa Number || error("objective function in optim evaluates to length $(length(val)) not 1")
            val / fnscale
        end
    else
        x -> begin
            val = fn(x; kwargs...)
            val isa Number || error("objective function in optim evaluates to length $(length(val)) not 1")
            val
        end
    end

    gr_scaled = if !isnothing(gr)
        if fnscale != 1.0 || any(parscale .!= 1.0)
            x -> begin
                g = gr(x .* parscale; kwargs...)
                # R's optim.c:109-111 validates gradient length every call
                length(g) == npar || error("gradient in optim evaluated to length $(length(g)) not $npar")
                (g .* parscale) / fnscale
            end
        else
            x -> begin
                g = gr(x; kwargs...)
                length(g) == npar || error("gradient in optim evaluated to length $(length(g)) not $npar")
                g
            end
        end
    else
        nothing
    end

    # Scale initial parameters
    par_scaled = par ./ parscale

    # Call appropriate method
    result = if method == "Nelder-Mead"
        _optim_neldermead(par_scaled, fn_scaled, con, lower_vec, upper_vec, parscale)
    elseif method == "BFGS"
        _optim_bfgs(par_scaled, fn_scaled, gr_scaled, con, parscale)
    elseif method == "L-BFGS-B"
        _optim_lbfgsb(par_scaled, fn_scaled, gr_scaled, con, lower_vec, upper_vec, parscale)
    elseif method == "Brent"
        _optim_brent(par_scaled[1], fn_scaled, con, lower_vec[1], upper_vec[1], parscale[1])
    end

    # Compute Hessian if requested
    hess = if hessian
        _compute_hessian(result.par, fn, gr, con, parscale; kwargs...)
    else
        nothing
    end

    # Return result in R's format
    return (
        par = result.par,
        value = result.value,
        counts = (function_=result.fn_evals, gradient=result.gr_evals),
        convergence = result.fail,
        message = result.message,
        hessian = hess
    )
end


# Method-specific implementations

function _optim_neldermead(par, fn, con, lower, upper, parscale)
    opts = NelderMeadOptions(
        abstol = con["abstol"],
        intol = con["reltol"],
        alpha = con["alpha"],
        beta = con["beta"],
        gamma = con["gamma"],
        trace = con["trace"] > 0,
        maxit = con["maxit"]
    )

    result = nmmin(fn, par, opts)

    return (
        par = result.x_opt .* parscale,
        value = result.f_opt * con["fnscale"],
        fn_evals = result.fncount,
        gr_evals = 0,
        fail = result.fail,
        message = nothing
    )
end


function _optim_bfgs(par, fn, gr, con, parscale)
    # Convert to internal function signature: f(n, x, ex)
    fn_internal(n, x, ex) = fn(x)
    gr_internal = isnothing(gr) ? nothing : (n, x, grad, ex) -> (grad .= gr(x); nothing)

    opts = BFGSOptions(
        abstol = con["abstol"],
        reltol = con["reltol"],
        gtol = con["gtol"],
        trace = con["trace"] > 0,
        maxit = con["maxit"],
        nREPORT = con["REPORT"]
    )

    result = bfgsmin(fn_internal, gr_internal, par;
                     options=opts, ndeps=con["ndeps"])

    return (
        par = result.x_opt .* parscale,
        value = result.f_opt * con["fnscale"],
        fn_evals = result.fn_evals,
        gr_evals = result.gr_evals,
        fail = result.fail,
        message = nothing
    )
end


function _optim_lbfgsb(par, fn, gr, con, lower, upper, parscale)
    npar = length(par)

    # R's optim.c:653-659 special-cases n==0
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

    # Scale bounds
    lower_scaled = lower ./ parscale
    upper_scaled = upper ./ parscale

    # Track evaluation counts externally so they survive exceptions
    fn_count = Ref(0)
    gr_count = Ref(0)

    # Convert to internal function signature.
    # R's optim.c:684 checks !R_FINITE(f) after every fn evaluation inside
    # the L-BFGS-B loop and hard-errors. We match this by checking in the wrapper.
    fn_internal(n, x, ex) = begin
        fn_count[] += 1
        val = fn(x)
        if !isfinite(val)
            error("L-BFGS-B needs finite values of 'fn'")
        end
        val
    end

    # L-BFGS-B requires a gradient function - use numerical gradients if not provided
    gr_internal = if !isnothing(gr)
        (n, x, ex) -> begin
            gr_count[] += 1
            gr(x)
        end
    else
        # Create bounded numerical gradient function (respects box constraints)
        npar_local = length(par)
        ndeps = con["ndeps"]
        cache = NumericalGradientCache(npar_local)
        (n, x, ex) -> begin
            gr_count[] += 1
            numgrad_bounded!(cache.df, cache.xtrial, fn_internal, n, x, ex, ndeps,
                             lower_scaled, upper_scaled)
            return cache.df
        end
    end

    opts = LBFGSBOptions(
        m = con["lmm"],
        factr = con["factr"],
        pgtol = con["pgtol"],
        maxit = con["maxit"],
        iprint = con["trace"] > 0 ? con["REPORT"] : 0
    )

    result = try
        lbfgsbmin(fn_internal, gr_internal, par;
                  l=lower_scaled, u=upper_scaled, options=opts)
    catch e
        if e isa ErrorException && e.msg == "L-BFGS-B needs finite values of 'fn'"
            # R-compatible error: convergence=52 with message
            # Use external counters to preserve actual evaluation counts
            (x_opt=copy(par), f_opt=NaN, n_iter=0,
             fail=52, fn_evals=fn_count[], gr_evals=gr_count[],
             message="ERROR: L-BFGS-B NEEDS FINITE VALUES OF FN")
        else
            rethrow(e)
        end
    end

    # Translate fail codes to R-compatible convergence codes:
    # 0 = converged, 1 = maxit reached, 51 = warning, 52 = error
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


function _optim_brent(par, fn, con, lower, upper, parscale)
    # fn is already scalar function
    opts = FminOptions(
        tol = con["reltol"],
        trace = con["trace"] > 0,
        maxit = con["maxit"]
    )

    result = fmin(x -> fn([x]), lower/parscale, upper/parscale; options=opts)

    # R's optim() Brent path always returns convergence=0 and counts=NA
    # (R delegates to optimize() which has no convergence/counts concept)
    return (
        par = [result.x_opt * parscale],
        value = result.f_opt * con["fnscale"],
        fn_evals = nothing,
        gr_evals = nothing,
        fail = 0,
        message = nothing
    )
end


function _compute_hessian(par, fn, gr, con, parscale; kwargs...)
    npar = length(par)
    fnscale = con["fnscale"]
    ndeps = con["ndeps"]

    fn_wrapper(x) = fn(x; kwargs...)
    gr_wrapper = isnothing(gr) ? nothing : (x -> gr(x; kwargs...))

    return optim_hessian(fn_wrapper, par, gr_wrapper;
                        fnscale=fnscale, parscale=parscale, ndeps=ndeps)
end
