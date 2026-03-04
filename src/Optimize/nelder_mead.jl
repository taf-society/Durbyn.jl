"""
    NelderMeadOptions(; abstol=-Inf, reltol=sqrt(eps(Float64)), alpha=1.0, beta=0.5,
                        gamma=2.0, trace=false, maxit=500, invalid_penalty=1e35,
                        project_to_bounds=false, lower=nothing, upper=nothing,
                        init_step_cap=nothing)

Configuration for `nelder_mead`, which now delegates to `Optim.jl`.

# Keyword Arguments

- `abstol::Float64`: Absolute objective tolerance.
- `reltol::Float64`: Relative objective tolerance.
- `alpha::Float64`: Reflection coefficient.
- `beta::Float64`: Contraction coefficient (also used as shrink coefficient).
- `gamma::Float64`: Expansion coefficient.
- `trace::Bool`: Print optimizer trace output.
- `maxit::Int`: Maximum objective evaluations (plus one initial feasibility check).
- `invalid_penalty::Float64`: Replacement value for non-finite objective evaluations.
- `project_to_bounds::Bool`: Enable box constraints if both `lower` and `upper` are provided.
- `lower::Union{Nothing,AbstractVector{<:Real}}`: Lower bounds.
- `upper::Union{Nothing,AbstractVector{<:Real}}`: Upper bounds.
- `init_step_cap::Union{Nothing,Float64}`: Optional cap for simplex edge initialization.
"""
struct NelderMeadOptions
    abstol::Float64
    reltol::Float64
    alpha::Float64
    beta::Float64
    gamma::Float64
    trace::Bool
    maxit::Int
    invalid_penalty::Float64
    project_to_bounds::Bool
    lower::Union{Nothing,AbstractVector{<:Real}}
    upper::Union{Nothing,AbstractVector{<:Real}}
    init_step_cap::Union{Nothing,Float64}
end

NelderMeadOptions(;
    abstol = -Inf,
    reltol = sqrt(eps(Float64)),
    alpha = 1.0,
    beta = 0.5,
    gamma = 2.0,
    trace = false,
    maxit = 500,
    invalid_penalty = 1.0e35,
    project_to_bounds = false,
    lower = nothing,
    upper = nothing,
    init_step_cap = nothing,
) = NelderMeadOptions(
    abstol,
    reltol,
    alpha,
    beta,
    gamma,
    trace,
    maxit,
    invalid_penalty,
    project_to_bounds,
    lower,
    upper,
    init_step_cap,
)

struct _DurbynSimplexer <: Optim.Simplexer
    init_step_cap::Union{Nothing,Float64}
    lower::Union{Nothing,Vector{Float64}}
    upper::Union{Nothing,Vector{Float64}}
end

@inline function _nm_clamp!(x::AbstractVector{<:Real},
    lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    @inbounds for i in eachindex(x, lower, upper)
        if x[i] < lower[i]
            x[i] = lower[i]
        elseif x[i] > upper[i]
            x[i] = upper[i]
        end
    end
    return x
end

function _nm_bounds(options::NelderMeadOptions, n::Int)
    has_bounds = options.project_to_bounds &&
                 !isnothing(options.lower) &&
                 !isnothing(options.upper)
    if !has_bounds
        return nothing, nothing
    end

    lower = Float64.(options.lower)
    upper = Float64.(options.upper)
    length(lower) == n || throw(ArgumentError("lower must have length $n"))
    length(upper) == n || throw(ArgumentError("upper must have length $n"))
    all(lower .<= upper) || throw(ArgumentError("all lower bounds must be <= upper bounds"))
    return lower, upper
end

function Optim.simplexer(S::_DurbynSimplexer, x0::Tx) where {Tx<:AbstractVector}
    n = length(x0)
    simplex = Tx[copy(x0) for _ in 1:n+1]

    step = 0.0
    @inbounds for i in 1:n
        step = max(step, 0.1 * abs(x0[i]))
    end
    step == 0.0 && (step = 0.1)

    @inbounds for j in 1:n
        base = simplex[j+1][j]
        trystep = step
        trial = base
        while trial == base
            trial = base + trystep
            if !isnothing(S.init_step_cap)
                trial = base + min(trystep, S.init_step_cap)
            end
            if !isnothing(S.lower)
                trial = clamp(trial, S.lower[j], S.upper[j])
            end
            trystep *= 10.0
            if !isnothing(S.init_step_cap) && trystep > S.init_step_cap
                trial = nextfloat(base)
                break
            end
        end
        simplex[j+1][j] = trial
    end

    return simplex
end

"""
    nelder_mead(f, x0, options::NelderMeadOptions)

Minimize a multivariate scalar objective with Nelder-Mead via `Optim.jl`.

Returns `(x_opt, f_opt, fncount, fail)`:
- `fail == 0`: converged.
- `fail == 1`: stopped before convergence.
"""
function nelder_mead(f::Function, x0::AbstractVector{<:Real}, options::NelderMeadOptions)
    x_init = Float64.(x0)
    n = length(x_init)
    lower, upper = _nm_bounds(options, n)
    has_bounds = !isnothing(lower)

    if has_bounds
        _nm_clamp!(x_init, lower, upper)
    end

    if options.maxit <= 0
        fmin = Float64(f(x_init))
        return (x_opt = copy(x_init), f_opt = fmin, fncount = 0, fail = 0)
    end

    f0 = Float64(f(x_init))
    isfinite(f0) || error("function cannot be evaluated at initial parameters")

    fncount = Ref(1)
    invalid_penalty = options.invalid_penalty

    objective = function (x)
        x_eval = has_bounds ? copy(x) : x
        if has_bounds
            _nm_clamp!(x_eval, lower, upper)
        end
        fx = Float64(f(x_eval))
        fncount[] += 1
        return isfinite(fx) ? fx : invalid_penalty
    end

    params = Optim.FixedParameters(
        α = options.alpha,
        β = options.gamma,
        γ = options.beta,
        δ = options.beta,
    )
    simplexer = _DurbynSimplexer(options.init_step_cap, lower, upper)
    method = Optim.NelderMead(parameters = params, initial_simplex = simplexer)
    opt_options = Optim.Options(
        iterations = options.maxit,
        f_calls_limit = options.maxit,
        f_abstol = options.abstol,
        f_reltol = options.reltol,
        show_trace = options.trace,
        show_every = 1,
        allow_f_increases = false,
    )

    result = if has_bounds
        Optim.optimize(
            objective,
            lower,
            upper,
            x_init,
            Optim.Fminbox(method),
            opt_options,
        )
    else
        Optim.optimize(objective, x_init, method, opt_options)
    end

    x_opt = Vector{Float64}(Optim.minimizer(result))
    if has_bounds
        _nm_clamp!(x_opt, lower, upper)
    end

    fail = Optim.converged(result) ? 0 : 1
    total_calls = max(fncount[], Optim.f_calls(result))
    return (
        x_opt = x_opt,
        f_opt = Float64(Optim.minimum(result)),
        fncount = total_calls,
        fail = fail,
    )
end
