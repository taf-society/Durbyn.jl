using LinearAlgebra: dot, norm

struct LBFGSBOptions
    m::Int
    factr::Float64
    pgtol::Float64
    maxit::Int
    iprint::Int
end

LBFGSBOptions(; m::Int=10, factr::Float64=1e7, pgtol::Float64=1e-5, maxit::Int=1000, iprint::Int=0) =
    LBFGSBOptions(m, factr, pgtol, maxit, iprint)


function _nbd_from_bounds(l::Vector{Float64}, u::Vector{Float64})
    n = length(l)
    nbd = Vector{Int32}(undef, n)
    @inbounds for i in 1:n
        if isfinite(l[i]) && isfinite(u[i])
            nbd[i] = 2
        elseif isfinite(l[i])
            nbd[i] = 1
        elseif isfinite(u[i])
            nbd[i] = 3
        else
            nbd[i] = 0
        end
    end
    return nbd
end

@inline function _project!(y::AbstractVector{Float64}, l::Vector{Float64}, u::Vector{Float64})
    @inbounds for i in eachindex(y)
        yi = y[i]
        if yi < l[i]
            y[i] = l[i]
        elseif yi > u[i]
            y[i] = u[i]
        end
    end
    return y
end


function _proj_grad!(pg::Vector{Float64}, x::Vector{Float64}, g::Vector{Float64},
    l::Vector{Float64}, u::Vector{Float64}, nbd::Vector{Int32})
    @inbounds for i in eachindex(x)
        gi = g[i]
        bi = nbd[i]
        if bi != 0
            if gi < 0.0
                if bi >= 2
                    gi = max(x[i] - u[i], gi)
                end
            else
                if bi <= 2
                    gi = min(x[i] - l[i], gi)
                end
            end
        end
        pg[i] = gi
    end

    m = 0.0
    @inbounds for i in eachindex(pg)
        v = abs(pg[i])
        if v > m
            m = v
        end
    end
    return m
end

function _lbfgs_direction!(d::Vector{Float64}, g::Vector{Float64},
    S::Vector{Vector{Float64}}, Y::Vector{Vector{Float64}},
    rho::Vector{Float64}, gamma0::Float64)
    copyto!(d, g)
    k = length(rho)
    if k == 0
        return
    end
    alpha = Vector{Float64}(undef, k)

    for i in k:-1:1
        alpha[i] = rho[i] * dot(S[i], d)
        LinearAlgebra.BLAS.axpy!(-alpha[i], Y[i], d)
    end

    @inbounds for i in eachindex(d)
        d[i] *= gamma0
    end

    for i in 1:k
        beta = rho[i] * dot(Y[i], d)
        LinearAlgebra.BLAS.axpy!(alpha[i] - beta, S[i], d)
    end
end

@inline function _feasible_step_cap(x::Vector{Float64}, d::Vector{Float64},
    l::Vector{Float64}, u::Vector{Float64})
    alpha_max = Inf
    @inbounds for i in eachindex(x)
        di = d[i]
        if di < 0.0 && isfinite(l[i])
            if x[i] <= l[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (l[i] - x[i]) / di)
        elseif di > 0.0 && isfinite(u[i])
            if x[i] >= u[i]
                return 0.0
            end
            alpha_max = min(alpha_max, (u[i] - x[i]) / di)
        end
    end
    return alpha_max
end


function _line_search_wolfe!(x::Vector{Float64}, fx::Float64, gx::Vector{Float64},
    d::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64},
    f::Function, g::Function, n::Int;
    c1::Float64=1e-4, c2::Float64=0.9, iter::Int=1,
    max_evals::Int=40)

    phi0prime = dot(gx, d)
    if !(isfinite(phi0prime)) || phi0prime >= 0.0

        return false, 0.0, fx, gx, x, 0, 0
    end

    alpha_max = _feasible_step_cap(x, d, l, u)
    if !(alpha_max > 0.0) || !isfinite(alpha_max)
        return false, 0.0, fx, gx, x, 0, 0
    end

    Dnorm = sqrt(dot(d, d))
    alpha = min(iter == 1 ? (1.0 / max(Dnorm, eps())) : 1.0, alpha_max)

    xtrial = similar(x)
    gtrial = similar(gx)

    fevals = 0
    gevals = 0

    f_prev = fx
    alpha_prev = 0.0
    f_best = fx
    alpha_best = 0.0
    g_best = copy(gx)

    eval_at = function (alphat)
        @inbounds @. xtrial = x + alphat * d
        ft = f(n, xtrial, nothing)
        fevals += 1
        gt = g(n, xtrial, nothing)
        gevals += 1
        return ft, gt
    end

    ft, gt = eval_at(alpha)
    phialpha = dot(gt, d)
    if ft <= fx + c1 * alpha * phi0prime
        f_best = ft
        alpha_best = alpha
        g_best .= gt
    end

    for k in 1:max_evals
        if (ft > fx + c1 * alpha * phi0prime) || (k > 1 && ft >= f_prev)

            ok, alphaz, fz, gz, ez_fe, ez_ge = _zoom!(x, fx, gx, d, l, u, f, g, n,
                alpha_prev, alpha, f_prev, ft,
                phi0prime; c1=c1, c2=c2, max_evals=max_evals - (fevals + gevals))
            fevals += ez_fe
            gevals += ez_ge
            if ok
                @inbounds @. xtrial = x + alphaz * d
                return true, alphaz, fz, gz, xtrial, fevals, gevals
            else
                if alpha_best > 0.0
                    @inbounds @. xtrial = x + alpha_best * d
                    return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
                end
                return false, 0.0, fx, gx, x, fevals, gevals
            end
        end
        if abs(phialpha) <= c2 * abs(phi0prime)
            @inbounds @. xtrial = x + alpha * d
            return true, alpha, ft, gt, xtrial, fevals, gevals
        end
        if phialpha >= 0.0

            ok, alphaz, fz, gz, ez_fe, ez_ge = _zoom!(x, fx, gx, d, l, u, f, g, n,
                alpha, alpha_prev, ft, f_prev,
                phi0prime; c1=c1, c2=c2, max_evals=max_evals - (fevals + gevals))
            fevals += ez_fe
            gevals += ez_ge
            if ok
                @inbounds @. xtrial = x + alphaz * d
                return true, alphaz, fz, gz, xtrial, fevals, gevals
            else
                if alpha_best > 0.0
                    @inbounds @. xtrial = x + alpha_best * d
                    return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
                end
                return false, 0.0, fx, gx, x, fevals, gevals
            end
        end

        alpha_prev, f_prev = alpha, ft
        alpha = min(2.0 * alpha, alpha_max)
        if alpha == alpha_prev
            if alpha_best > 0.0
                @inbounds @. xtrial = x + alpha_best * d
                return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
            elseif ft < fx
                @inbounds @. xtrial = x + alpha * d
                return true, alpha, ft, gt, xtrial, fevals, gevals
            else
                return false, 0.0, fx, gx, x, fevals, gevals
            end
        end
        ft, gt = eval_at(alpha)
        phialpha = dot(gt, d)
        if ft <= fx + c1 * alpha * phi0prime
            f_best = ft
            alpha_best = alpha
            g_best .= gt
        end
    end

    if alpha_best > 0.0
        @inbounds @. xtrial = x + alpha_best * d
        return true, alpha_best, f_best, g_best, xtrial, fevals, gevals
    end
    return false, 0.0, fx, gx, x, fevals, gevals
end

function _zoom!(x, fx, gx, d, l, u, f, g, n,
    alpha_lo, alpha_hi, f_lo, f_hi, phi0prime; c1=1e-4, c2=0.9, max_evals=30)
    fevals = 0
    gevals = 0
    xtrial = similar(x)
    gtrial = similar(gx)

    @inbounds @. xtrial = x + alpha_lo * d
    f_lo_eval = f_lo
    g_lo = g(n, xtrial, nothing)
    gevals += 1
    phi_lo = dot(g_lo, d)

    eval_at = function (alphat)
        @inbounds @. xtrial = x + alphat * d
        ft = f(n, xtrial, nothing)
        fevals += 1
        gt = g(n, xtrial, nothing)
        gevals += 1
        return ft, gt
    end

    for _ in 1:max_evals
        alpha_j = 0.5 * (alpha_lo + alpha_hi)
        fj, gj = eval_at(alpha_j)
        if (fj > fx + c1 * alpha_j * phi0prime) || (fj >= f_lo_eval)
            alpha_hi, f_hi = alpha_j, fj
        else
            phij = dot(gj, d)
            if abs(phij) <= c2 * abs(phi0prime)
                return true, alpha_j, fj, gj, fevals, gevals
            end
            if phij * (alpha_hi - alpha_lo) >= 0
                alpha_hi, f_hi = alpha_lo, f_lo_eval
            end
            alpha_lo, f_lo_eval = alpha_j, fj
        end
        if abs(alpha_hi - alpha_lo) <= 1e-16
            return false, alpha_j, fj, gj, fevals, gevals
        end
    end
    return false, 0.0, fx, gx, fevals, gevals
end

"""
    lbfgsbmin(f, g, x0; mask=trues(length(x0)), l=nothing, u=nothing, options=LBFGSBOptions())

Limited-memory BFGS with box constraints (L-BFGS-B–like).

- `f(n, x, ex)` → scalar objective
- `g(n, x, ex)` → gradient vector (size n)
- `mask` freezes variables by setting [l=u=x0] internally
- `l`, `u` optional bounds (`nothing` = unbounded)
- `options`: `m` (history), `factr` (f_tol = factr * eps()), `pgtol` (projected grad ∞-norm),
             `maxit` (iteration cap), `iprint` (0 silent, >0 prints)

Returns named tuple: `x_opt, f_opt, n_iter, fail, fn_evals, gr_evals`.
"""
function lbfgsbmin(f::Function, g::Function, x0::Vector{Float64};
    mask=trues(length(x0)),
    l::Union{Nothing,Vector{Float64}}=nothing,
    u::Union{Nothing,Vector{Float64}}=nothing,
    options::LBFGSBOptions=LBFGSBOptions())

    n = length(x0)
    if length(mask) != n
        error("mask length must equal x0 length")
    end

    l2 = l === nothing ? fill(-Inf, n) : copy(l)
    u2 = u === nothing ? fill(+Inf, n) : copy(u)
    @inbounds for i in 1:n
        if !mask[i]
            l2[i] = x0[i]
            u2[i] = x0[i]
        end
    end
    nbd = _nbd_from_bounds(l2, u2)

    # Initialize
    x = copy(x0)
    _project!(x, l2, u2)
    fx = f(n, x, nothing)
    gx = g(n, x, nothing)
    fn_evals = 1
    gr_evals = 1

    pg = similar(gx)
    pg_norm_inf = _proj_grad!(pg, x, gx, l2, u2, nbd)

    f_tol = options.factr * eps(Float64)
    f_prev = fx
    fail = 1
    iter = 0

    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()
    rho = Float64[]

    if options.iprint > 0
        println("iter  f(x)                ||pg||inf          alpha")
    end

    while iter < options.maxit
        iter += 1

        if pg_norm_inf <= options.pgtol
            fail = 0
            break
        end

        d = similar(gx)

        gamma0 = 1.0
        if !isempty(rho)
            yk = Y[end]
            sk = S[end]
            ys = dot(yk, sk)
            yy = dot(yk, yk)
            if yy > 0
                gamma0 = ys / yy
            end
        end
        _lbfgs_direction!(d, gx, S, Y, rho, gamma0)
        @. d = -d

        @inbounds for i in 1:n
            if (x[i] <= l2[i] && d[i] < 0.0) || (x[i] >= u2[i] && d[i] > 0.0)
                d[i] = 0.0
            end
        end

        if dot(gx, d) >= 0.0
            @inbounds for i in 1:n
                di = -gx[i]
                if (x[i] <= l2[i] && di < 0.0) || (x[i] >= u2[i] && di > 0.0)
                    di = 0.0
                end
                d[i] = di
            end
        end

        if maximum(abs, d) < 1e-16
            fail = 0
            break
        end

        ok, alpha, fnew, gnew, xnew, fe, ge = _line_search_wolfe!(x, fx, gx, d, l2, u2, f, g, n; iter=iter)
        fn_evals += fe
        gr_evals += ge

        if options.iprint > 0
            println("iter=", iter, " f=", fnew, " pg_inf=", pg_norm_inf, " alpha=", (ok ? alpha : 0.0))
        end

        if !ok
            fail = 1
            break
        end

        s = xnew .- x
        y = gnew .- gx
        ys = dot(y, s)

        if ys > 1e-10 * norm(s) * norm(y)
            push!(S, s)
            push!(Y, y)
            push!(rho, 1.0 / ys)
            if length(S) > options.m
                popfirst!(S)
                popfirst!(Y)
                popfirst!(rho)
            end
        end

        x .= xnew
        fx = fnew
        gx .= gnew

        pg_norm_inf = _proj_grad!(pg, x, gx, l2, u2, nbd)

        rel_decr = abs(f_prev - fx) / max(1.0, abs(f_prev), abs(fx))
        f_prev = fx

        if (pg_norm_inf <= options.pgtol) || (rel_decr <= f_tol)
            fail = 0
            break
        end
    end

    return (x_opt=x, f_opt=fx, n_iter=iter,
        fail=fail, fn_evals=fn_evals, gr_evals=gr_evals)
end
