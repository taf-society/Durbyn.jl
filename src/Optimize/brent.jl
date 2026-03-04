"""
    BrentOptions(; tol=1.5e-8, trace=false, maxit=1000)

Options for 1D Brent minimization.
"""
struct BrentOptions
    tol::Float64
    trace::Bool
    maxit::Int
end

BrentOptions(; tol = 1.5e-8, trace = false, maxit = 1000) =
    BrentOptions(tol, trace, maxit)

"""
    brent(f, lower, upper; options=BrentOptions())

Public Brent wrapper that delegates to `Optim.Brent`.
Returns `(x_opt, f_opt, n_iter, fail, fn_evals)`.
"""
function brent(f, lower::Float64, upper::Float64; options::BrentOptions = BrentOptions())
    if lower >= upper
        throw(ArgumentError("'xmin' not less than 'xmax'"))
    end

    tol = options.tol
    trace = options.trace
    maxit = options.maxit
    fn_eval_count = Ref(0)

    f_wrapped = function (x)
        fx = Float64(f(x))
        fn_eval_count[] += 1
        if !isfinite(fx)
            if fx == -Inf
                @warn "Brent: non-finite f(x) clamped to floatmax (-Inf)"
                return -floatmax(Float64)
            else
                @warn "Brent: non-finite f(x) clamped to floatmax ($(isnan(fx) ? "NaN" : "Inf"))"
                return floatmax(Float64)
            end
        end
        return fx
    end

    result = Optim.optimize(
        f_wrapped,
        lower,
        upper;
        method = Optim.Brent(),
        rel_tol = tol,
        abs_tol = tol,
        iterations = maxit,
        show_trace = trace,
    )

    return (
        x_opt = Float64(Optim.minimizer(result)),
        f_opt = Float64(Optim.minimum(result)),
        n_iter = Optim.iterations(result),
        fail = Optim.converged(result) ? 0 : 1,
        fn_evals = max(fn_eval_count[], Optim.f_calls(result)),
    )
end
