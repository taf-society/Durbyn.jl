"""
    FminOptions(; tol=1.5e-8, trace=false, maxit=1000)

Options for the Brent (fmin) optimization algorithm for 1D minimization.

# Keyword Arguments

- `tol::Float64=1.5e-8`:
    Desired length of the interval of uncertainty of the final result.
    The algorithm stops when the interval is smaller than this tolerance.

- `trace::Bool=false`:
    If `true`, progress and diagnostics are printed during optimization.

- `maxit::Int=1000`:
    Maximum number of iterations.

# Example

```julia
options = FminOptions(tol=1e-6, trace=true, maxit=500)
```

You can then pass this `options` object to the optimizer:

```julia
result = fmin(f, lower, upper; options=options)
```
"""
struct FminOptions
    tol::Float64
    trace::Bool
    maxit::Int
end

FminOptions(; tol=1.5e-8, trace=false, maxit=1000) =
    FminOptions(tol, trace, maxit)

"""
    fmin(f, lower, upper; options=FminOptions())

Finds an approximation to the point where `f` attains a minimum on the interval `(lower, upper)`.

# Description

This is an implementation of Brent's method, which combines golden section search and
successive parabolic interpolation for 1D optimization without derivatives.
Convergence is never much slower than that for a Fibonacci search. If `f` has a
continuous second derivative which is positive at the minimum (which is not at
`lower` or `upper`), then convergence is superlinear, and usually of the order of
about 1.324.

The function `f` is never evaluated at two points closer together than
`eps*abs(x_min) + (tol/3)`, where `eps` is approximately the square root of the
relative machine precision. If `f` is a unimodal function and the computed values
of `f` are always unimodal when separated by at least `eps*abs(x) + (tol/3)`,
then `fmin` approximates the abcissa of the global minimum of `f` on the interval
`[lower, upper]` with an error less than `3*eps*abs(x_min) + tol`. If `f` is not
unimodal, then `fmin` may approximate a local, but perhaps non-global, minimum to
the same accuracy.

# Arguments

- `f::Function`: Objective function to minimize. Takes a single Float64 argument and returns a Float64.
- `lower::Float64`: Left endpoint of the initial interval.
- `upper::Float64`: Right endpoint of the initial interval.

# Keyword Arguments

- `options::FminOptions`: Optional. Struct with solver settings (tolerance, tracing, max iterations).

# Returns

A named tuple with the following fields:
- `x_opt`: Optimal parameter value found (abcissa of the minimum).
- `f_opt`: Function value at the optimum.
- `n_iter`: Number of iterations performed.
- `fail`: Status flag (`0` if converged, `1` if maximum iterations reached).
- `fn_evals`: Number of function evaluations.

# Examples

## Simple Quadratic Function

```julia
f(x) = (x - 2.0)^2 + 1.0
result = fmin(f, 0.0, 5.0)
println("Minimum at x = ", result.x_opt)  # Should be near 2.0
println("Function value = ", result.f_opt)  # Should be near 1.0
```

## Rosenbrock-like 1D Function

```julia
f(x) = 100 * (x^2 - 1)^2 + (x - 1)^2
result = fmin(f, -2.0, 2.0, options=FminOptions(tol=1e-10, trace=true))
println("Minimum at x = ", result.x_opt)  # Should be near 1.0
```

## Cosine Function

```julia
f(x) = cos(x) + x/10
result = fmin(f, -10.0, 10.0)
println("Minimum at x = ", result.x_opt)
println("Function value = ", result.f_opt)
```

## With Custom Options

```julia
f(x) = x^4 - 14*x^3 + 60*x^2 - 70*x
options = FminOptions(tol=1e-8, trace=true, maxit=100)
result = fmin(f, 0.0, 2.0, options=options)
println("Minimum at x = ", result.x_opt)
println("Converged: ", result.fail == 0)
println("Function evaluations: ", result.fn_evals)
```

# Source

A Julia translation of Fortran code https://netlib.org/fmm/fmin.f (author(s) unstated)
based on the Algol 60 procedure localmin given in the reference below.

# References

Brent, R. (1973) *Algorithms for Minimization without Derivatives*.
Englewood Cliffs, N.J.: Prentice-Hall.

# See Also

- [`FminOptions`](@ref): Options struct for configuring the optimizer.
- [`nmmin`](@ref): Nelder-Mead simplex algorithm for multidimensional optimization.
- [`bfgsmin`](@ref): BFGS quasi-Newton algorithm for multidimensional optimization with gradients.
"""
function fmin(f::Function, lower::Float64, upper::Float64; options::FminOptions=FminOptions())
    if lower >= upper
        error("'xmin' not less than 'xmax'")
    end

    tol = options.tol
    trace = options.trace
    maxit = options.maxit

    c = 0.5 * (3.0 - sqrt(5.0))

    eps = 1.0
    tol1 = 1.0 + eps
    while tol1 > 1.0
        eps = eps / 2.0
        tol1 = 1.0 + eps
    end
    eps = sqrt(eps)

    a = lower
    b = upper
    v = a + c * (b - a)
    w = v
    x = v
    e = 0.0
    d = 0.0

    fx = f(x)
    fv = fx
    fw = fx
    funcount = 1

    if !isfinite(fx)
        if fx == -Inf
            @warn "-Inf replaced by maximally negative value"
            fx = -floatmax(Float64)
        else
            @warn "$(isnan(fx) ? "NaN" : "Inf") replaced by maximum positive value"
            fx = floatmax(Float64)
        end
        fv = fx
        fw = fx
    end

    if trace
        println("Brent's fmin: Initial f(x) = ", fx, " at x = ", x)
    end

    iter = 0
    fail = 1

    while iter < maxit
        iter += 1

        xm = 0.5 * (a + b)
        tol1 = eps * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1

        if abs(x - xm) <= (tol2 - 0.5 * (b - a))
            fail = 0
            break
        end

        if abs(e) <= tol1
            if x >= xm
                e = a - x
            else
                e = b - x
            end
            d = c * e
        else
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0
                p = -p
            end
            q = abs(q)
            r = e
            e = d

            parabola_ok = (abs(p) < abs(0.5 * q * r)) &&
                          (p > q * (a - x)) &&
                          (p < q * (b - x))

            if parabola_ok
                d = p / q
                u = x + d

                if (u - a) < tol2 || (b - u) < tol2
                    d = copysign(tol1, xm - x)
                end
            else
                if x >= xm
                    e = a - x
                else
                    e = b - x
                end
                d = c * e
            end
        end

        if abs(d) >= tol1
            u = x + d
        else
            u = x + copysign(tol1, d)
        end

        fu = f(u)
        funcount += 1

        if trace && (iter % 10 == 0 || !isfinite(fu))
            println("Iter $iter: x = $u, f(x) = $fu, interval = [$(a), $(b)]")
        end

        if !isfinite(fu)
            if fu == -Inf
                @warn "-Inf replaced by maximally negative value"
                fu = -floatmax(Float64)
            else
                @warn "$(isnan(fu) ? "NaN" : "Inf") replaced by maximum positive value"
                fu = floatmax(Float64)
            end
        end

        if fu <= fx
            if u >= x
                a = x
            else
                b = x
            end
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else
            if u < x
                a = u
            else
                b = u
            end

            if fu <= fw || w == x
                v = w
                fv = fw
                w = u
                fw = fu
            elseif fu <= fv || v == x || v == w
                v = u
                fv = fu
            end
        end
    end

    if trace
        println("Final value: f(x) = $fx at x = $x")
        if fail == 0
            println("Converged in $iter iterations")
        else
            println("Stopped after $iter iterations (maximum reached)")
        end
    end

    return (
        x_opt = x,
        f_opt = fx,
        n_iter = iter,
        fail = fail,
        fn_evals = funcount
    )
end