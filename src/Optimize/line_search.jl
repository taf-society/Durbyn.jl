"""
    Line Search Algorithms

Implements strong Wolfe line search (Nocedal & Wright (2006), Algorithms 3.5 and 3.6)
with cubic interpolation and bound-aware feasibility constraints.

Shared by both BFGS and L-BFGS-B solvers.

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed.,
  Algorithms 3.5, 3.6, Eq. 3.59. Springer.
"""

# Allocation-free dot product and norm for hot-path use.
# LinearAlgebra.dot/norm dispatch through BLAS on small vectors and allocate
# wrapper objects; these hand-written versions avoid that overhead entirely.

@inline function _ls_dot(a::Vector{Float64}, b::Vector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end

@inline function _ls_norm(v::Vector{Float64})
    s = 0.0
    @inbounds @simd for i in eachindex(v)
        s += v[i] * v[i]
    end
    sqrt(s)
end

@inline function _ls_axpy!(dst::Vector{Float64}, x::Vector{Float64}, alpha::Float64, d::Vector{Float64})
    @inbounds @simd for i in eachindex(dst, x, d)
        dst[i] = x[i] + alpha * d[i]
    end
end


"""
    _max_feasible_step(x, d, lb, ub)

Compute the maximum step length α such that `x + α*d` remains feasible
(within bounds `lb` and `ub`).

Nocedal & Wright (2006), Section 16.7.
"""
@inline function _max_feasible_step(x::Vector{Float64}, d::Vector{Float64},
    lb::Vector{Float64}, ub::Vector{Float64})
    alpha_max = Inf
    @inbounds for i in eachindex(x)
        di = d[i]
        if di < 0.0 && isfinite(lb[i])
            if x[i] <= lb[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (lb[i] - x[i]) / di)
        elseif di > 0.0 && isfinite(ub[i])
            if x[i] >= ub[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (ub[i] - x[i]) / di)
        end
    end
    return alpha_max
end


"""
    _cubic_interpolation_min(a1, f1, dphi1, a2, f2, dphi2)

Find the minimizer of the cubic interpolant through two points with
function values and derivatives.

Nocedal & Wright (2006), Eq. (3.59).
"""
function _cubic_interpolation_min(a1, f1, dphi1, a2, f2, dphi2)
    d1 = dphi1 + dphi2 - 3.0 * (f2 - f1) / (a2 - a1)
    disc = d1 * d1 - dphi1 * dphi2
    disc < 0.0 && return NaN
    d2 = copysign(sqrt(disc), a2 - a1)
    return a2 - (a2 - a1) * (dphi2 + d2 - d1) / (dphi2 - dphi1 + 2.0 * d2)
end


# Inline evaluation: compute xtrial = x + alpha*d, then f(xtrial) and g(xtrial).
# Returns (f_val, gtrial) and increments eval counters via Ref.
@inline function _ls_eval!(xtrial, gtrial, x, alpha, d, f, g, fevals, gevals)
    _ls_axpy!(xtrial, x, alpha, d)
    ft = f(xtrial)
    fevals[] += 1
    gt = g(xtrial)
    gevals[] += 1
    if gt !== gtrial
        copyto!(gtrial, gt)
    end
    return ft, gtrial
end


"""
    _strong_wolfe_line_search!(x, fx, grad, d, lb, ub, f, g; kwargs...)

Perform a line search satisfying the strong Wolfe conditions using
bracketing and zoom.

Nocedal & Wright (2006), Algorithm 3.5.

# Arguments
- `x::Vector{Float64}`: Current point
- `fx::Float64`: Function value at x
- `grad::Vector{Float64}`: Gradient at x
- `d::Vector{Float64}`: Search direction
- `lb, ub::Vector{Float64}`: Bound vectors for feasibility
- `f, g::Function`: Objective and gradient functions

# Keyword Arguments
- `c1::Float64=1e-3`: Sufficient decrease parameter
- `c2::Float64=0.9`: Curvature condition parameter
- `iter::Int=1`: Iteration number (affects initial step size)
- `xtol::Float64=0.1`: Tolerance for bracketing width
- `boxed::Bool=false`: Whether problem is box-constrained
- `max_evals::Int=40`: Maximum function/gradient evaluations
- `max_step::Float64=Inf`: Maximum allowable step size

# Returns
Tuple `(success, alpha, f_new, g_new, x_new, fevals, gevals)`
"""
function _strong_wolfe_line_search!(x::Vector{Float64}, fx::Float64, grad::Vector{Float64},
    d::Vector{Float64}, lb::Union{Nothing,Vector{Float64}}, ub::Union{Nothing,Vector{Float64}},
    f::Function, g::Function;
    c1::Float64=1e-3, c2::Float64=0.9, iter::Int=1,
    xtol::Float64=0.1, boxed::Bool=false,
    max_evals::Int=40, max_step::Float64=Inf,
    xtrial_buf::Union{Nothing,Vector{Float64}}=nothing,
    gtrial_buf::Union{Nothing,Vector{Float64}}=nothing,
    gbest_buf::Union{Nothing,Vector{Float64}}=nothing,
    gsave_buf::Union{Nothing,Vector{Float64}}=nothing)

    phi0prime = _ls_dot(grad, d)
    if !(isfinite(phi0prime)) || phi0prime >= 0.0
        return false, 0.0, fx, grad, x, 0, 0
    end

    alpha_max = if boxed
        (isnothing(lb) || isnothing(ub)) &&
            throw(ArgumentError("boxed line search requires finite bound vectors"))
        _max_feasible_step(x, d, lb, ub)
    else
        Inf
    end
    if isfinite(max_step) && max_step < alpha_max
        alpha_max = max_step
    end
    if !(alpha_max > 0.0)
        return false, 0.0, fx, grad, x, 0, 0
    end

    Dnorm = _ls_norm(d)
    alpha_init = (iter == 1 && !boxed) ? (1.0 / max(Dnorm, eps())) : 1.0
    alpha = isfinite(alpha_max) ? min(alpha_init, alpha_max) : alpha_init

    xtrial = isnothing(xtrial_buf) ? similar(x) : xtrial_buf
    gtrial = isnothing(gtrial_buf) ? similar(grad) : gtrial_buf

    fevals = Ref(0)
    gevals = Ref(0)

    f_prev = fx
    alpha_prev = 0.0
    f_best = fx
    alpha_best = 0.0
    g_best = isnothing(gbest_buf) ? similar(grad) : gbest_buf
    g_saved = isnothing(gsave_buf) ? similar(grad) : gsave_buf
    copyto!(g_best, grad)

    ft, gt = _ls_eval!(xtrial, gtrial, x, alpha, d, f, g, fevals, gevals)
    phialpha = _ls_dot(gt, d)
    if ft <= fx + c1 * alpha * phi0prime
        f_best = ft
        alpha_best = alpha
        copyto!(g_best, gt)
    end

    for k in 1:max_evals
        if (ft > fx + c1 * alpha * phi0prime) || (k > 1 && ft >= f_prev)
            ok, alphaz, fz, gz, ez_fe, ez_ge = _wolfe_zoom!(x, fx, grad, d, lb, ub, f, g,
                alpha_prev, alpha, f_prev, ft,
                phi0prime;
                c1=c1, c2=c2, xtol=xtol, max_evals=max_evals - (fevals[] + gevals[]),
                xtrial_buf=xtrial, gtrial_buf=gtrial, gsave_buf=g_saved)
            fevals[] += ez_fe
            gevals[] += ez_ge
            if ok
                _ls_axpy!(xtrial, x, alphaz, d)
                return true, alphaz, fz, gz, xtrial, fevals[], gevals[]
            else
                if alpha_best > 0.0
                    _ls_axpy!(xtrial, x, alpha_best, d)
                    return true, alpha_best, f_best, g_best, xtrial, fevals[], gevals[]
                end
                return false, 0.0, fx, grad, x, fevals[], gevals[]
            end
        end
        if abs(phialpha) <= c2 * abs(phi0prime)
            _ls_axpy!(xtrial, x, alpha, d)
            return true, alpha, ft, gt, xtrial, fevals[], gevals[]
        end
        if phialpha >= 0.0
            ok, alphaz, fz, gz, ez_fe, ez_ge = _wolfe_zoom!(x, fx, grad, d, lb, ub, f, g,
                alpha, alpha_prev, ft, f_prev,
                phi0prime;
                c1=c1, c2=c2, xtol=xtol, max_evals=max_evals - (fevals[] + gevals[]),
                xtrial_buf=xtrial, gtrial_buf=gtrial, gsave_buf=g_saved)
            fevals[] += ez_fe
            gevals[] += ez_ge
            if ok
                _ls_axpy!(xtrial, x, alphaz, d)
                return true, alphaz, fz, gz, xtrial, fevals[], gevals[]
            else
                if alpha_best > 0.0
                    _ls_axpy!(xtrial, x, alpha_best, d)
                    return true, alpha_best, f_best, g_best, xtrial, fevals[], gevals[]
                end
                return false, 0.0, fx, grad, x, fevals[], gevals[]
            end
        end

        alpha_prev, f_prev = alpha, ft
        alpha = min(2.0 * alpha, alpha_max)
        if alpha == alpha_prev
            if alpha_best > 0.0
                _ls_axpy!(xtrial, x, alpha_best, d)
                return true, alpha_best, f_best, g_best, xtrial, fevals[], gevals[]
            elseif ft < fx
                _ls_axpy!(xtrial, x, alpha, d)
                return true, alpha, ft, gt, xtrial, fevals[], gevals[]
            else
                return false, 0.0, fx, grad, x, fevals[], gevals[]
            end
        end
        ft, gt = _ls_eval!(xtrial, gtrial, x, alpha, d, f, g, fevals, gevals)
        phialpha = _ls_dot(gt, d)
        if ft <= fx + c1 * alpha * phi0prime
            f_best = ft
            alpha_best = alpha
            copyto!(g_best, gt)
        end
    end

    if alpha_best > 0.0
        _ls_axpy!(xtrial, x, alpha_best, d)
        return true, alpha_best, f_best, g_best, xtrial, fevals[], gevals[]
    end
    return false, 0.0, fx, grad, x, fevals[], gevals[]
end


"""
    _wolfe_zoom!(x, fx, gx, d, lb, ub, f, g, alpha_lo, alpha_hi, f_lo, f_hi, phi0prime; kwargs...)

Zoom phase of the strong Wolfe line search. Given a bracket [alpha_lo, alpha_hi]
known to contain an acceptable step length, narrow the bracket until the strong
Wolfe conditions are satisfied.

Nocedal & Wright (2006), Algorithm 3.6.
"""
function _wolfe_zoom!(x, fx, gx, d, lb, ub, f, g,
    alpha_lo, alpha_hi, f_lo, f_hi, phi0prime;
    c1=1e-3, c2=0.9, xtol=0.1, max_evals=30,
    xtrial_buf::Union{Nothing,Vector{Float64}}=nothing,
    gtrial_buf::Union{Nothing,Vector{Float64}}=nothing,
    gsave_buf::Union{Nothing,Vector{Float64}}=nothing)
    fevals = 0
    gevals = 0
    xtrial = isnothing(xtrial_buf) ? similar(x) : xtrial_buf
    gtrial = isnothing(gtrial_buf) ? similar(gx) : gtrial_buf
    g_lo_saved = isnothing(gsave_buf) ? similar(gx) : gsave_buf

    _ls_axpy!(xtrial, x, alpha_lo, d)
    f_lo_eval = f_lo
    g_lo = g(xtrial)
    gevals += 1
    if g_lo !== gtrial
        copyto!(gtrial, g_lo)
    end
    phi_lo = _ls_dot(gtrial, d)
    copyto!(g_lo_saved, gtrial)

    phi_hi = NaN

    for _ in 1:max_evals
        bracket_max = max(alpha_lo, alpha_hi)
        if bracket_max > 0 && abs(alpha_hi - alpha_lo) <= xtol * bracket_max
            return true, alpha_lo, f_lo_eval, g_lo_saved, fevals, gevals
        end
        alpha_j = _cubic_interpolation_min(alpha_lo, f_lo_eval, phi_lo, alpha_hi, f_hi, phi_hi)
        a_min = min(alpha_lo, alpha_hi)
        a_max = max(alpha_lo, alpha_hi)
        if !isfinite(alpha_j) || alpha_j <= a_min || alpha_j >= a_max
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
        end

        # Inline eval_at
        _ls_axpy!(xtrial, x, alpha_j, d)
        fj = f(xtrial)
        fevals += 1
        gj_raw = g(xtrial)
        gevals += 1
        if gj_raw !== gtrial
            copyto!(gtrial, gj_raw)
        end
        gj = gtrial

        if (fj > fx + c1 * alpha_j * phi0prime) || (fj >= f_lo_eval)
            alpha_hi, f_hi = alpha_j, fj
            phi_hi = _ls_dot(gj, d)
        else
            phij = _ls_dot(gj, d)
            if abs(phij) <= c2 * abs(phi0prime)
                return true, alpha_j, fj, gj, fevals, gevals
            end
            if phij * (alpha_hi - alpha_lo) >= 0
                alpha_hi, f_hi = alpha_lo, f_lo_eval
                phi_hi = phi_lo
            end
            alpha_lo, f_lo_eval = alpha_j, fj
            phi_lo = phij
            copyto!(g_lo_saved, gj)
        end
        if abs(alpha_hi - alpha_lo) <= 1e-16
            return true, alpha_lo, f_lo_eval, g_lo_saved, fevals, gevals
        end
    end
    return false, 0.0, fx, gx, fevals, gevals
end
