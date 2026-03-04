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
    bfgs(f, g, x0; mask=trues(length(x0)), options=BFGSOptions()) -> NamedTuple

Public BFGS wrapper that delegates to `Optim.BFGS`.
"""
function bfgs(
    f::Function,
    g::Union{Function,Nothing},
    x0::Vector{Float64};
    mask = trues(length(x0)),
    options::BFGSOptions = BFGSOptions(),
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
        if isnothing(g)
            Optim.optimize(
                f_wrapped,
                x_active0,
                Optim.BFGS(),
                opt_options;
                autodiff = Optim.ADTypes.AutoFiniteDiff(),
            )
        else
            x_grad = copy(x_start)
            grad_full = zeros(Float64, n_total)

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

            Optim.optimize(f_wrapped, g_wrapped!, x_active0, Optim.BFGS(), opt_options; inplace = true)
        end
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

    gr_calls_result = try
        Optim.g_calls(result)
    catch
        g_calls[]
    end

    return (
        x_opt = x_opt,
        f_opt = Float64(Optim.minimum(result)),
        n_iter = Optim.iterations(result),
        fail = Optim.converged(result) ? 0 : 1,
        fn_evals = max(f_calls[], Optim.f_calls(result)),
        gr_evals = max(g_calls[], gr_calls_result),
    )
end
