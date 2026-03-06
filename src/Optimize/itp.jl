"""
    ITPOptions(; tol=1.5e-8, k1=nothing, k2=2.0, n0=1, maxit=1000, trace=false)

Options for ITP (Interpolate-Truncate-Project) bracketed root finding.

# Parameters
- `tol`: Target half-width tolerance `eps` for stopping (`b - a <= 2*tol`).
- `k1`: Truncation size parameter. If `nothing`, uses `0.2 / (b0 - a0)`.
- `k2`: Truncation exponent in `[1, 1 + phi]`.
- `n0`: Projection slack parameter (`>= 0`).
- `maxit`: Maximum ITP iterations.
- `trace`: Print per-iteration diagnostics.
"""
struct ITPOptions
    tol::Float64
    k1::Union{Nothing,Float64}
    k2::Float64
    n0::Int
    maxit::Int
    trace::Bool
end

ITPOptions(; tol = 1.5e-8, k1 = nothing, k2 = 2.0, n0 = 1, maxit = 1000, trace = false) =
    ITPOptions(Float64(tol), isnothing(k1) ? nothing : Float64(k1), Float64(k2), Int(n0), Int(maxit), Bool(trace))

@inline function _safe_root_eval(f::Function, x::Float64)
    y = Float64(f(x))
    isfinite(y) || throw(ArgumentError("itp requires finite function values on the bracket"))
    return y
end

"""
    itp(f, a, b; options=ITPOptions())

Find a root of a continuous scalar function `f` in bracket `[a, b]` using ITP.

Requires a sign change (`f(a) * f(b) < 0`), unless one endpoint is already a root.

Returns a named tuple:
`(x_root, f_root, n_iter, fail, fn_evals)`, where:
- `fail == 0`: converged (`|f(x_root)| == 0` or bracket width `<= 2*tol`)
- `fail == 1`: maximum iterations reached before tolerance

# References
- Oliveira, I. F. D. & Takahashi, R. H. C. (2021).
  *An Enhancement of the Bisection Method Average Performance Preserving Minmax Optimality*.
  ACM TOMS, 47(1).
"""
function itp(f::Function, a::Float64, b::Float64; options::ITPOptions = ITPOptions())
    a < b || throw(ArgumentError("'a' must be strictly less than 'b'"))

    tol = options.tol
    tol > 0.0 || throw(ArgumentError("'tol' must be positive"))
    options.n0 >= 0 || throw(ArgumentError("'n0' must be non-negative"))
    options.maxit > 0 || throw(ArgumentError("'maxit' must be positive"))

    phi = (1.0 + sqrt(5.0)) / 2.0
    (1.0 <= options.k2 <= 1.0 + phi) || throw(ArgumentError("'k2' must satisfy 1 <= k2 <= 1 + phi"))

    f_a = _safe_root_eval(f, a)
    f_b = _safe_root_eval(f, b)
    fn_evals = 2

    if f_a == 0.0
        return (x_root = a, f_root = 0.0, n_iter = 0, fail = 0, fn_evals = fn_evals)
    end
    if f_b == 0.0
        return (x_root = b, f_root = 0.0, n_iter = 0, fail = 0, fn_evals = fn_evals)
    end

    f_a * f_b < 0.0 || throw(ArgumentError("itp requires a sign change on [a, b]"))

    # Normalize sign convention internally to y(a) < 0 < y(b) without
    # changing endpoint order.
    sign_scale = f_a < 0.0 ? 1.0 : -1.0
    y_a = sign_scale * f_a
    y_b = sign_scale * f_b

    initial_width = b - a
    default_k1 = 0.2 / initial_width
    k1 = isnothing(options.k1) ? default_k1 : options.k1
    k1 > 0.0 || throw(ArgumentError("'k1' must be positive"))

    n_half = max(0, ceil(Int, log2(initial_width / (2.0 * tol))))
    n_bound = n_half + options.n0
    max_iterations = min(options.maxit, n_bound + 1)

    iteration_count = 0
    while (b - a) > 2.0 * tol && iteration_count < max_iterations
        x_mid = 0.5 * (a + b)
        denominator = y_a - y_b
        x_false = denominator == 0.0 ? x_mid : (b * y_a - a * y_b) / denominator

        sigma = sign(x_mid - x_false)
        if sigma == 0.0
            sigma = 1.0
        end

        delta = k1 * (b - a)^options.k2
        x_trunc = if delta <= abs(x_mid - x_false)
            x_false + sigma * delta
        else
            x_mid
        end

        radius = ldexp(tol, n_bound - iteration_count) - 0.5 * (b - a)
        x_itp = if abs(x_trunc - x_mid) <= radius
            x_trunc
        else
            x_mid - sigma * radius
        end
        # Ensure strictly interior query to preserve bracket invariant.
        x_itp = clamp(x_itp, nextfloat(a), prevfloat(b))

        f_itp = _safe_root_eval(f, x_itp)
        fn_evals += 1

        if options.trace
            println("iter=", iteration_count + 1,
                    " a=", a,
                    " b=", b,
                    " x=", x_itp,
                    " f=", f_itp)
        end

        y_itp = sign_scale * f_itp

        if f_itp == 0.0
            return (x_root = x_itp, f_root = 0.0, n_iter = iteration_count + 1, fail = 0, fn_evals = fn_evals)
        elseif y_itp > 0.0
            b = x_itp
            y_b = y_itp
        else
            a = x_itp
            y_a = y_itp
        end

        iteration_count += 1
    end

    x_root = 0.5 * (a + b)
    f_root = _safe_root_eval(f, x_root)
    fn_evals += 1
    converged = (b - a) <= 2.0 * tol
    return (
        x_root = x_root,
        f_root = f_root,
        n_iter = iteration_count,
        fail = converged ? 0 : 1,
        fn_evals = fn_evals,
    )
end
