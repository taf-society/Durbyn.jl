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

- Brent, R. P. (1973). *Algorithms for Minimization Without Derivatives*.
  Prentice-Hall.
- Nash, J. C. (1990). *Compact Numerical Methods for Computers*, 2nd ed. Adam Hilger.

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