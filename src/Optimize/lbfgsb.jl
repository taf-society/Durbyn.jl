"""
    LBFGSBOptions

Options for L-BFGS-B optimization.
"""
struct LBFGSBOptions
    memory_size::Int
    ftol_factor::Float64
    pg_tol::Float64
    maxit::Int
    print_level::Int
end

LBFGSBOptions(; memory_size::Int = 10, ftol_factor::Float64 = 1e7,
    pg_tol::Float64 = 1e-5, maxit::Int = 1000, print_level::Int = 0) =
    LBFGSBOptions(memory_size, ftol_factor, pg_tol, maxit, print_level)

struct _LBFGSBNonFinite <: Exception
    what::Symbol
end

@inline function _project_bounds!(
    x::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
)
    @inbounds for i in eachindex(x, lower, upper)
        xi = x[i]
        if xi < lower[i]
            x[i] = lower[i]
        elseif xi > upper[i]
            x[i] = upper[i]
        end
    end
    return x
end

"""
    lbfgsb(f, g, x0; mask=trues(length(x0)), lower=nothing, upper=nothing, options=LBFGSBOptions())

Public L-BFGS-B wrapper that delegates to `Optim.Fminbox(Optim.LBFGS(...))`.
"""
function lbfgsb(
    f::Function,
    g::Union{Function,Nothing},
    x0::Vector{Float64};
    mask = trues(length(x0)),
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
    options::LBFGSBOptions = LBFGSBOptions(),
)
    n = length(x0)
    length(mask) == n || throw(ArgumentError("mask length must equal x0 length"))

    lb = isnothing(lower) ? fill(-Inf, n) : copy(lower)
    ub = isnothing(upper) ? fill(+Inf, n) : copy(upper)
    length(lb) == n || throw(ArgumentError("lower must have length $n"))
    length(ub) == n || throw(ArgumentError("upper must have length $n"))

    @inbounds for i in 1:n
        if !mask[i]
            lb[i] = x0[i]
            ub[i] = x0[i]
        end
        if isnan(lb[i])
            lb[i] = -Inf
        end
        if isnan(ub[i])
            ub[i] = Inf
        end
    end

    @inbounds for i in 1:n
        if lb[i] > ub[i]
            return (
                x_opt = copy(x0),
                f_opt = Inf,
                n_iter = 0,
                fail = 52,
                fn_evals = 0,
                gr_evals = 0,
                message = "Error: no feasible solution (lower > upper)",
            )
        end
    end

    has_finite_bounds = any(isfinite, lb) || any(isfinite, ub)
    x_start = copy(x0)
    if has_finite_bounds
        _project_bounds!(x_start, lb, ub)
    end

    f0 = Float64(f(x_start))
    if !isfinite(f0)
        return (
            x_opt = copy(x_start),
            f_opt = f0,
            n_iter = 0,
            fail = 52,
            fn_evals = 1,
            gr_evals = 0,
            message = "Error: objective function returned non-finite value",
        )
    end

    f_calls = Ref(1)
    g_calls = Ref(0)

    if !isnothing(g)
        g0 = g(x_start)
        if length(g0) != n
            throw(ArgumentError("gradient must have length $n"))
        end
        if !all(isfinite, g0)
            return (
                x_opt = copy(x_start),
                f_opt = f0,
                n_iter = 0,
                fail = 52,
                fn_evals = 1,
                gr_evals = 1,
                message = "Error: gradient contains non-finite values",
            )
        end
        g_calls[] = 1
    end

    f_wrapped = function (x)
        fx = Float64(f(x))
        f_calls[] += 1
        isfinite(fx) || throw(_LBFGSBNonFinite(:objective))
        return fx
    end

    g_wrapped! = if isnothing(g)
        nothing
    else
        function (G, x)
            gv = g(x)
            g_calls[] += 1
            if length(gv) != n
                throw(ArgumentError("gradient must have length $n"))
            end
            if !all(isfinite, gv)
                throw(_LBFGSBNonFinite(:gradient))
            end
            copyto!(G, gv)
            return nothing
        end
    end

    opt_options = Optim.Options(
        iterations = options.maxit,
        outer_iterations = options.maxit,
        f_reltol = options.ftol_factor * eps(Float64),
        g_abstol = options.pg_tol,
        show_trace = options.print_level > 0,
        show_every = 1,
        allow_f_increases = false,
        allow_outer_f_increases = false,
    )

    result = try
        if has_finite_bounds
            if isnothing(g_wrapped!)
                Optim.optimize(
                    f_wrapped,
                    lb,
                    ub,
                    x_start,
                    Optim.Fminbox(Optim.LBFGS(m = options.memory_size)),
                    opt_options;
                    autodiff = Optim.ADTypes.AutoFiniteDiff(),
                )
            else
                Optim.optimize(
                    f_wrapped,
                    g_wrapped!,
                    lb,
                    ub,
                    x_start,
                    Optim.Fminbox(Optim.LBFGS(m = options.memory_size)),
                    opt_options;
                    inplace = true,
                )
            end
        else
            if isnothing(g_wrapped!)
                Optim.optimize(
                    f_wrapped,
                    x_start,
                    Optim.LBFGS(m = options.memory_size),
                    opt_options;
                    autodiff = Optim.ADTypes.AutoFiniteDiff(),
                )
            else
                Optim.optimize(
                    f_wrapped,
                    g_wrapped!,
                    x_start,
                    Optim.LBFGS(m = options.memory_size),
                    opt_options;
                    inplace = true,
                )
            end
        end
    catch err
        if err isa _LBFGSBNonFinite
            msg = err.what === :gradient ?
                "Error: gradient contains non-finite values" :
                "Error: objective function returned non-finite value"
            return (
                x_opt = copy(x_start),
                f_opt = f0,
                n_iter = 0,
                fail = 52,
                fn_evals = f_calls[],
                gr_evals = g_calls[],
                message = msg,
            )
        end
        rethrow()
    end

    converged = Optim.converged(result)
    fail = converged ? 0 : (result.stopped_by.iterations ? 1 : 52)
    msg = if converged
        "Converged"
    elseif fail == 1
        "maximum iterations reached"
    else
        string("abnormal termination: ", Optim.termination_code(result))
    end

    x_opt = Vector{Float64}(Optim.minimizer(result))
    _project_bounds!(x_opt, lb, ub)

    gr_calls_result = try
        Optim.g_calls(result)
    catch
        g_calls[]
    end

    return (
        x_opt = x_opt,
        f_opt = Float64(Optim.minimum(result)),
        n_iter = Optim.iterations(result),
        fail = fail,
        fn_evals = max(f_calls[], Optim.f_calls(result)),
        gr_evals = max(g_calls[], gr_calls_result),
        message = msg,
    )
end
