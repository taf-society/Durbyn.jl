using LinearAlgebra: mul!

"""
    BFGSOptions

Options for BFGS optimization.
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

Compatibility workspace type kept for public API stability.
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
            zeros(n_total),
        )
    end
end

"""
    bfgs_hessian_update!(H, s, y, Hy, sTy)

In-place BFGS inverse-Hessian update (compatibility helper).
"""
@inline function bfgs_hessian_update!(
    H::Matrix{Float64},
    s::Vector{Float64},
    y::Vector{Float64},
    Hy::Vector{Float64},
    sTy::Float64,
)
    n = length(s)
    mul!(Hy, H, y)

    yTHy = 0.0
    @inbounds @simd for i in 1:n
        yTHy += y[i] * Hy[i]
    end
    scale = (1.0 + yTHy / sTy) / sTy
    inv_sTy = 1.0 / sTy

    @inbounds for j in 1:n
        @simd for i in 1:n
            H[i, j] += scale * s[i] * s[j] - inv_sTy * (s[i] * Hy[j] + Hy[i] * s[j])
        end
    end
end

struct _BFGSObjectiveNonFinite <: Exception
    value::Float64
end

@inline function _write_full_from_active!(
    x_full::Vector{Float64},
    x_base::Vector{Float64},
    active::Vector{Int},
    x_active::AbstractVector{<:Real},
)
    copyto!(x_full, x_base)
    @inbounds for k in eachindex(active)
        x_full[active[k]] = x_active[k]
    end
    return x_full
end

"""
    bfgs(f, g, x0; mask=trues(length(x0)), options=BFGSOptions(),
         step_sizes=nothing, numgrad_cache=nothing) -> NamedTuple

Public BFGS wrapper that delegates to `Optim.BFGS`.
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
    length(mask) == n_total || throw(ArgumentError("mask length must equal x0 length"))
    report_interval > 0 || throw(ArgumentError("report_interval must be positive for BFGS"))

    x_start = copy(x0)
    active = findall(mask)
    n_active = length(active)

    f0 = Float64(f(x_start))
    isfinite(f0) || error("BFGS: initial objective value is not finite")

    if maxit <= 0 || n_active == 0
        return (
            x_opt = copy(x_start),
            f_opt = f0,
            n_iter = 0,
            fail = 0,
            fn_evals = 1,
            gr_evals = 0,
        )
    end

    x_active0 = x_start[active]
    x_obj = copy(x_start)
    x_grad = copy(x_start)
    grad_full = zeros(Float64, n_total)

    f_calls = Ref(1)
    g_calls = Ref(0)

    f_full = function (x_full)
        fx = Float64(f(x_full))
        f_calls[] += 1
        isfinite(fx) || throw(_BFGSObjectiveNonFinite(fx))
        return fx
    end

    f_wrapped = function (x_active)
        _write_full_from_active!(x_obj, x_start, active, x_active)
        return f_full(x_obj)
    end

    if isnothing(g)
        ndeps = isnothing(step_sizes) ? fill(1e-3, n_total) : step_sizes
        length(ndeps) == n_total || throw(ArgumentError("step_sizes must have length $n_total"))
        cache = isnothing(numgrad_cache) ? NumericalGradientCache(n_total) : numgrad_cache

        g_wrapped! = function (G, x_active)
            _write_full_from_active!(x_grad, x_start, active, x_active)
            g_calls[] += 1
            numgrad_with_cache!(cache, f_full, x_grad, ndeps)
            @inbounds for k in eachindex(active)
                G[k] = cache.gradient[active[k]]
            end
            if !all(isfinite, G)
                fill!(G, NaN)
            end
            return nothing
        end

        opt_options = Optim.Options(
            iterations = maxit,
            f_abstol = abstol,
            f_reltol = reltol,
            g_abstol = max(gtol, 0.0),
            show_trace = trace,
            show_every = report_interval,
            allow_f_increases = false,
        )

        result = try
            Optim.optimize(f_wrapped, g_wrapped!, x_active0, Optim.BFGS(), opt_options; inplace = true)
        catch err
            if err isa _BFGSObjectiveNonFinite
                return (
                    x_opt = copy(x_start),
                    f_opt = f0,
                    n_iter = 0,
                    fail = 1,
                    fn_evals = f_calls[],
                    gr_evals = g_calls[],
                )
            end
            rethrow()
        end

        x_opt = copy(x_start)
        x_opt_active = Optim.minimizer(result)
        @inbounds for k in eachindex(active)
            x_opt[active[k]] = x_opt_active[k]
        end

        return (
            x_opt = x_opt,
            f_opt = Float64(Optim.minimum(result)),
            n_iter = Optim.iterations(result),
            fail = Optim.converged(result) ? 0 : 1,
            fn_evals = max(f_calls[], Optim.f_calls(result)),
            gr_evals = max(g_calls[], Optim.g_calls(result)),
        )
    else
        g_wrapped! = function (G, x_active)
            _write_full_from_active!(x_grad, x_start, active, x_active)
            fill!(grad_full, 0.0)
            g(grad_full, x_grad)
            g_calls[] += 1
            @inbounds for k in eachindex(active)
                G[k] = grad_full[active[k]]
            end
            if !all(isfinite, G)
                fill!(G, NaN)
            end
            return nothing
        end

        opt_options = Optim.Options(
            iterations = maxit,
            f_abstol = abstol,
            f_reltol = reltol,
            g_abstol = max(gtol, 0.0),
            show_trace = trace,
            show_every = report_interval,
            allow_f_increases = false,
        )

        result = try
            Optim.optimize(f_wrapped, g_wrapped!, x_active0, Optim.BFGS(), opt_options; inplace = true)
        catch err
            if err isa _BFGSObjectiveNonFinite
                return (
                    x_opt = copy(x_start),
                    f_opt = f0,
                    n_iter = 0,
                    fail = 1,
                    fn_evals = f_calls[],
                    gr_evals = g_calls[],
                )
            end
            rethrow()
        end

        x_opt = copy(x_start)
        x_opt_active = Optim.minimizer(result)
        @inbounds for k in eachindex(active)
            x_opt[active[k]] = x_opt_active[k]
        end

        return (
            x_opt = x_opt,
            f_opt = Float64(Optim.minimum(result)),
            n_iter = Optim.iterations(result),
            fail = Optim.converged(result) ? 0 : 1,
            fn_evals = max(f_calls[], Optim.f_calls(result)),
            gr_evals = max(g_calls[], Optim.g_calls(result)),
        )
    end
end
