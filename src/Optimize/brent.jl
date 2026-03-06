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

Public Brent minimizer implemented directly from Brent's bounded 1D
golden-section/parabolic interpolation equations.
Returns `(x_opt, f_opt, n_iter, fail, fn_evals)`.
"""
function brent(f, lower::Float64, upper::Float64; options::BrentOptions = BrentOptions())
    if lower >= upper
        throw(ArgumentError("'xmin' not less than 'xmax'"))
    end

    tol = options.tol
    trace = options.trace
    maxit = options.maxit
    tol > 0.0 || throw(ArgumentError("'tol' must be positive"))
    maxit > 0 || throw(ArgumentError("'maxit' must be positive"))
    fn_eval_count = Ref(0)

    f_wrapped = function (x::Float64)
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

    # Brent constants.
    golden_ratio_complement = 0.5 * (3.0 - sqrt(5.0))
    machine_tolerance_factor = sqrt(eps(Float64))

    # Bracket endpoints.
    lower_bound = lower
    upper_bound = upper

    # Point triplet and function values.
    x_best = lower_bound + golden_ratio_complement * (upper_bound - lower_bound)
    x_second = x_best
    x_previous = x_best

    f_best = f_wrapped(x_best)
    f_second = f_best
    f_previous = f_best

    # Step trackers (fmin.f notation: d, e).
    step_size = 0.0
    previous_step = 0.0

    @inbounds for iteration_index in 1:maxit
        midpoint = 0.5 * (lower_bound + upper_bound)
        tolerance_absolute = machine_tolerance_factor * abs(x_best) + tol / 3.0
        tolerance_bracket = 2.0 * tolerance_absolute

        if abs(x_best - midpoint) <= tolerance_bracket - 0.5 * (upper_bound - lower_bound)
            return (
                x_opt = x_best,
                f_opt = f_best,
                n_iter = iteration_index - 1,
                fail = 0,
                fn_evals = fn_eval_count[],
            )
        end

        use_golden_section = true
        parabolic_p = 0.0
        parabolic_q = 0.0

        if abs(previous_step) > tolerance_absolute
            r_coefficient = (x_best - x_second) * (f_best - f_previous)
            q_coefficient = (x_best - x_previous) * (f_best - f_second)
            parabolic_p = (x_best - x_previous) * q_coefficient - (x_best - x_second) * r_coefficient
            parabolic_q = 2.0 * (q_coefficient - r_coefficient)

            if parabolic_q > 0.0
                parabolic_p = -parabolic_p
            else
                parabolic_q = -parabolic_q
            end

            old_previous_step = previous_step
            previous_step = step_size

            if abs(parabolic_p) < 0.5 * parabolic_q * abs(old_previous_step) &&
               parabolic_p > parabolic_q * (lower_bound - x_best) &&
               parabolic_p < parabolic_q * (upper_bound - x_best)
                step_size = parabolic_p / parabolic_q
                trial_point = x_best + step_size
                if (trial_point - lower_bound) < tolerance_bracket || (upper_bound - trial_point) < tolerance_bracket
                    step_size = copysign(tolerance_absolute, midpoint - x_best)
                end
                use_golden_section = false
            end
        end

        if use_golden_section
            previous_step = x_best >= midpoint ? (lower_bound - x_best) : (upper_bound - x_best)
            step_size = golden_ratio_complement * previous_step
        end

        trial_point = if abs(step_size) >= tolerance_absolute
            x_best + step_size
        else
            x_best + copysign(tolerance_absolute, step_size)
        end

        f_trial = f_wrapped(trial_point)

        if trace
            println(
                "iter=", iteration_index,
                " lower=", lower_bound,
                " upper=", upper_bound,
                " x=", x_best,
                " u=", trial_point,
                " f_u=", f_trial,
            )
        end

        if f_trial <= f_best
            if trial_point >= x_best
                lower_bound = x_best
            else
                upper_bound = x_best
            end

            x_previous = x_second
            f_previous = f_second
            x_second = x_best
            f_second = f_best
            x_best = trial_point
            f_best = f_trial
        else
            if trial_point < x_best
                lower_bound = trial_point
            else
                upper_bound = trial_point
            end

            if f_trial <= f_second || x_second == x_best
                x_previous = x_second
                f_previous = f_second
                x_second = trial_point
                f_second = f_trial
            elseif f_trial <= f_previous || x_previous == x_best || x_previous == x_second
                x_previous = trial_point
                f_previous = f_trial
            end
        end
    end

    return (
        x_opt = x_best,
        f_opt = f_best,
        n_iter = maxit,
        fail = 1,
        fn_evals = fn_eval_count[],
    )
end
