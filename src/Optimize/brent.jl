"""
    BrentOptions(; tol=1.5e-8, trace=false, maxit=1000)

Options for Brent's method for 1D minimization.

# Keyword Arguments

- `tol::Float64=1.5e-8`: Desired interval uncertainty. The algorithm stops when
  the interval is smaller than this tolerance.
- `trace::Bool=false`: If `true`, prints progress during optimization.
- `maxit::Int=1000`: Maximum number of iterations.

# Example

```julia
options = BrentOptions(tol=1e-6, trace=true, maxit=500)
result = brent(f, lower, upper; options=options)
```
"""
struct BrentOptions
    tol::Float64
    trace::Bool
    maxit::Int
end

BrentOptions(; tol=1.5e-8, trace=false, maxit=1000) =
    BrentOptions(tol, trace, maxit)

# Golden section ratio: (3 - √5) / 2 ≈ 0.381966
# Brent (1973), Chapter 5
const GOLDEN_SECTION = 0.5 * (3.0 - sqrt(5.0))

# Machine epsilon for convergence checks
# Brent (1973), Chapter 5
const SQRT_MACH_EPS = sqrt(eps(Float64))

"""
    brent(f, lower, upper; options=BrentOptions())

Find the minimum of a univariate function on a bounded interval using Brent's method.

Brent's method combines golden section search with successive parabolic interpolation
for derivative-free 1D optimization. Convergence is never much slower than a Fibonacci
search. For functions with a continuous positive second derivative at the minimum,
convergence is superlinear with order approximately 1.324.

# Arguments

- `f::Function`: Objective function taking a single `Float64` argument.
- `lower::Float64`: Left endpoint of the search interval.
- `upper::Float64`: Right endpoint of the search interval.

# Keyword Arguments

- `options::BrentOptions`: Solver settings (tolerance, tracing, max iterations).

# Returns

Named tuple `(x_opt, f_opt, n_iter, fail, fn_evals)`.

# References

- Brent, R. P. (1973). *Algorithms for Minimization Without Derivatives*, Chapter 5.
  Prentice-Hall.
- Lagarias, J. C. et al. (1998). Convergence properties of the Nelder-Mead simplex
  method in low dimensions. *SIAM J. Optim.*, 9(1), 112–147.

# Examples

```julia
f(x) = (x - 2.0)^2 + 1.0
result = brent(f, 0.0, 5.0)

f(x) = cos(x) + x/10
result = brent(f, -10.0, 10.0; options=BrentOptions(tol=1e-10, trace=true))
```
"""
function brent(f::Function, lower::Float64, upper::Float64; options::BrentOptions=BrentOptions())
    if lower >= upper
        throw(ArgumentError("'xmin' not less than 'xmax'"))
    end

    tol = options.tol
    trace = options.trace
    maxit = options.maxit

    lo = lower
    hi = upper
    x_min = lo + GOLDEN_SECTION * (hi - lo)
    x_second = x_min   # second-best point
    x_prev = x_min     # previous second-best point
    prev_step = 0.0     # distance moved two steps ago
    step_size = 0.0     # current step size

    f_min = f(x_min)
    f_second = f_min
    f_prev = f_min
    fn_eval_count = 1

    if !isfinite(f_min)
        if f_min == -Inf
            @warn "Brent: non-finite f(x) clamped to floatmax (-Inf)"
            f_min = -floatmax(Float64)
        else
            @warn "Brent: non-finite f(x) clamped to floatmax ($(isnan(f_min) ? "NaN" : "Inf"))"
            f_min = floatmax(Float64)
        end
        f_second = f_min
        f_prev = f_min
    end

    if trace
        println("Brent's fmin: Initial f(x) = ", f_min, " at x = ", x_min)
    end

    iter = 0
    fail = 1

    while iter < maxit
        iter += 1

        midpoint = 0.5 * (lo + hi)
        tol_abs = SQRT_MACH_EPS * abs(x_min) + tol / 3.0
        tol_bracket = 2.0 * tol_abs

        # Convergence check: is x_min close enough to the midpoint?
        if abs(x_min - midpoint) <= (tol_bracket - 0.5 * (hi - lo))
            fail = 0
            break
        end

        # Decide between parabolic interpolation and golden section
        if abs(prev_step) <= tol_abs
            # Golden section step
            if x_min >= midpoint
                prev_step = lo - x_min
            else
                prev_step = hi - x_min
            end
            step_size = GOLDEN_SECTION * prev_step
        else
            # Attempt parabolic interpolation
            r = (x_min - x_second) * (f_min - f_prev)
            q = (x_min - x_prev) * (f_min - f_second)
            p = (x_min - x_prev) * q - (x_min - x_second) * r
            q = 2.0 * (q - r)
            if q > 0.0
                p = -p
            end
            q = abs(q)
            r = prev_step
            prev_step = step_size

            parabola_ok = (abs(p) < abs(0.5 * q * r)) &&
                          (p > q * (lo - x_min)) &&
                          (p < q * (hi - x_min))

            if parabola_ok
                step_size = p / q
                u = x_min + step_size

                # Ensure trial point is not too close to bounds
                if (u - lo) < tol_bracket || (hi - u) < tol_bracket
                    step_size = copysign(tol_abs, midpoint - x_min)
                end
            else
                # Fall back to golden section
                if x_min >= midpoint
                    prev_step = lo - x_min
                else
                    prev_step = hi - x_min
                end
                step_size = GOLDEN_SECTION * prev_step
            end
        end

        # Compute trial point, ensuring minimum step size
        if abs(step_size) >= tol_abs
            u = x_min + step_size
        else
            u = x_min + copysign(tol_abs, step_size)
        end

        fu = f(u)
        fn_eval_count += 1

        if trace && (iter % 10 == 0 || !isfinite(fu))
            println("Iter $iter: x = $u, f(x) = $fu, interval = [$(lo), $(hi)]")
        end

        if !isfinite(fu)
            if fu == -Inf
                @warn "Brent: non-finite f(x) clamped to floatmax (-Inf)"
                fu = -floatmax(Float64)
            else
                @warn "Brent: non-finite f(x) clamped to floatmax ($(isnan(fu) ? "NaN" : "Inf"))"
                fu = floatmax(Float64)
            end
        end

        # Update bracket and best/second-best/previous points
        if fu <= f_min
            if u >= x_min
                lo = x_min
            else
                hi = x_min
            end
            x_prev = x_second
            f_prev = f_second
            x_second = x_min
            f_second = f_min
            x_min = u
            f_min = fu
        else
            if u < x_min
                lo = u
            else
                hi = u
            end

            if fu <= f_second || x_second == x_min
                x_prev = x_second
                f_prev = f_second
                x_second = u
                f_second = fu
            elseif fu <= f_prev || x_prev == x_min || x_prev == x_second
                x_prev = u
                f_prev = fu
            end
        end
    end

    if trace
        println("Final value: f(x) = $f_min at x = $x_min")
        if fail == 0
            println("Converged in $iter iterations")
        else
            println("Stopped after $iter iterations (maximum reached)")
        end
    end

    return (
        x_opt = x_min,
        f_opt = f_min,
        n_iter = iter,
        fail = fail,
        fn_evals = fn_eval_count
    )
end
