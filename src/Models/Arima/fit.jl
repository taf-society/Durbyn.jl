function compute_css_residuals(
    y::AbstractArray,
    arma::Vector{Int},
    phi::AbstractArray,
    theta::AbstractArray,
    ncond::Int,
)
    n = length(y)
    p = length(phi)
    q = length(theta)

    w = copy(y)

    for _ = 1:arma[6]
        for l = n:-1:2
            w[l] -= w[l-1]
        end
    end

    ns = arma[5]
    for _ = 1:arma[7]
        for l = n:-1:(ns+1)
            w[l] -= w[l-ns]
        end
    end

    resid = Vector{Float64}(undef, n)
    for i = 1:ncond
        resid[i] = 0.0
    end

    ssq = 0.0
    nu = 0

    for l = (ncond+1):n
        tmp = w[l]
        for j = 1:p
            if (l - j) < 1
                continue
            end
            tmp -= phi[j] * w[l-j]
        end

        jmax = min(l - ncond, q)
        for j = 1:jmax
            if (l - j) < 1
                continue
            end
            tmp -= theta[j] * resid[l-j]
        end

        resid[l] = tmp

        if !isnan(tmp)
            nu += 1
            ssq += tmp^2
        end
    end

    return (sigma2 = ssq / nu, residuals = resid)
end

function process_xreg(xreg::Union{NamedMatrix,Nothing}, n::Int)
    if isnothing(xreg)
        xreg_mat = Matrix{Float64}(undef, n, 0)
        ncxreg = 0
        nmxreg = String[]
    else
        if size(xreg.data, 1) != n
            throw(ArgumentError("Lengths of x and xreg do not match!"))
        end
        xreg_mat = xreg.data
        if !(eltype(xreg_mat) <: Float64)
            xreg_mat = Float64.(xreg_mat)
        end
        ncxreg = size(xreg_mat, 2)
        nmxreg = xreg.colnames
    end
    return xreg_mat, ncxreg, nmxreg
end

function regress_and_update!(
    x::AbstractArray,
    xreg::Matrix,
    narma::Int,
    ncxreg::Int,
    order_d::Int,
    seasonal_d::Int,
    m::Int,
    Delta::AbstractArray,
)

    init0 = zeros(narma)
    parscale = ones(narma)

    dx = copy(x)
    dxreg = copy(xreg)
    if order_d > 0
        dx = diff(dx; lag = 1, differences = order_d)
        dxreg = diff(dxreg; lag = 1, differences = order_d)
        dx, dxreg = dropmissing(dx, dxreg)
    end
    if m > 1 && seasonal_d > 0
        dx = diff(dx; lag = m, differences = seasonal_d)
        dxreg = diff(dxreg; lag = m, differences = seasonal_d)
        dx, dxreg = dropmissing(dx, dxreg)
    end

    if length(dx) > size(dxreg, 2)
        try
            fit = Stats.ols(dx, dxreg)
            fit_rank = rank(dxreg)
        catch e
            @warn "Fitting OLS to difference data failed: $e"
            fit = nothing
            fit_rank = 0
        end
    else
        @debug "Not enough observations to fit OLS" length_dx=length(dx) predictors=size(dxreg, 2)
        fit = nothing
        fit_rank = 0
    end

    if fit_rank == 0
        x_clean, xreg_clean = dropmissing(x, xreg)
        fit = Stats.ols(x_clean, xreg_clean)
    end

    has_na = isnan.(x) .| [any(isnan, row) for row in eachrow(xreg)]
    n_used = sum(.!has_na) - length(Delta)
    model_coefs = Stats.coefficients(fit)
    init0 = append!(init0, model_coefs)
    ses = fit.se
    parscale = append!(parscale, 10 * ses)

    return init0, parscale, n_used
end

function prep_coefs(arma::Vector{Int}, coef::AbstractArray, cn::Vector{String}, ncxreg::Int)
    names = String[]
    if arma[1] > 0
        append!(names, ["ar$(i)" for i in 1:arma[1]])
    end
    if arma[2] > 0
        append!(names, ["ma$(i)" for i in 1:arma[2]])
    end
    if arma[3] > 0
        append!(names, ["sar$(i)" for i in 1:arma[3]])
    end
    if arma[4] > 0
        append!(names, ["sma$(i)" for i in 1:arma[4]])
    end
    if ncxreg > 0
        append!(names, cn)
    end
    mat = reshape(coef, 1, :)
    return NamedMatrix(mat, names)
end

"""
    fit!(model::SARIMA)

Fit a SARIMA model in-place using the method specified in `model.method`.

Supported methods: `:css`, `:ml`, `:css_ml` (default).

After fitting, `model.results` contains a `SARIMAResults` and `model.system`
contains the final state-space representation.

Returns the model itself (for chaining).
"""
function fit!(model::SARIMA{Fl}) where {Fl}
    order = model.order
    arma = arma_vector(order)
    narma = sum(arma[1:4])
    Delta = build_delta(order)
    method = model.method
    transform_pars = model.transform_pars
    optim_method = model.optim_method
    optim_control = model.optim_control
    kappa = model.kappa
    SSinit = model.SSinit
    SS_G = SSinit === :gardner1980

    x = model.y
    n = length(x)
    y_save = copy(x)

    nd = order.d + order.D
    n_used = length(dropmissing(x)) - length(Delta)

    xreg_original = model.xreg_original

    xreg = model.xreg
    if model.include_mean && (nd == 0)
        if isnothing(xreg)
            xreg = NamedMatrix(zeros(n, 0), String[])
        end
        xreg = add_drift_term(xreg, ones(n), "intercept")
    end

    xreg_mat, ncxreg, nmxreg = process_xreg(xreg, n)

    if method === :css_ml
        has_missing = xi -> (ismissing(xi) || isnan(xi))
        anyna = any(has_missing, x)
        if ncxreg > 0
            anyna |= any(has_missing, xreg_mat)
        end
        if anyna
            method = :ml
        end
    end

    n_cond_input = model.fixed
    if method in (:css, :css_ml)
        ncond = order.d + order.D * order.s
        ncond1 = order.p + order.P * order.s
        ncond += ncond1
    else
        ncond = 0
    end

    fixed = if isnothing(model.fixed)
        fill(NaN, narma + ncxreg)
    else
        f = Float64.(model.fixed)
        if length(f) != narma + ncxreg
            throw(ArgumentError("Wrong length for 'fixed'"))
        end
        f
    end
    mask = isnan.(fixed)
    no_optim = !any(mask)

    if no_optim
        transform_pars = false
    end

    if transform_pars
        ind = arma[1] + arma[2] .+ (1:arma[3])
        if any(.!mask[1:arma[1]]) || any(.!mask[ind])
            @warn "Some AR parameters were fixed: Setting transform_pars = false"
            transform_pars = false
        end
    end

    orig_xreg_flag = true
    S_svd = nothing
    if ncxreg > 0
        orig_xreg_flag = (ncxreg == 1) || any(.!mask[(narma+1):(narma+ncxreg)])
        if !orig_xreg_flag
            rows_good = [all(isfinite, row) for row in eachrow(xreg_mat)]
            S_svd = svd(xreg_mat[rows_good, :])
            xreg_mat = xreg_mat * S_svd.V
        end
    end

    if ncxreg > 0
        init0, parscale, n_used =
            regress_and_update!(x, xreg_mat, narma, ncxreg, order.d, order.D, order.s, Delta)
    else
        init0 = zeros(narma)
        parscale = ones(narma)
    end

    if n_used <= 0
        throw(ArgumentError("Too few non-missing observations"))
    end

    init = model.init
    if !isnothing(init)
        init = copy(init)
        if length(init) != length(init0)
            throw(ArgumentError("'init' is of the wrong length"))
        end
        ind = map(xi -> isnan(xi) || ismissing(xi), init)
        if any(ind)
            init[ind] .= init0[ind]
        end
        if method === :ml
            p_ar = arma[1]
            P_ar = arma[3]
            if p_ar > 0 && !ar_check(init[1:p_ar])
                error("non-stationary AR part")
            end
            if P_ar > 0
                sa_start = arma[1] + arma[2] + 1
                sa_stop = arma[1] + arma[2] + P_ar
                if !ar_check(init[sa_start:sa_stop])
                    error("non-stationary seasonal AR part")
                end
            end
        end
    else
        init = copy(init0)
    end

    coef = copy(Float64.(fixed))
    kalman_ws = Ref{Union{KalmanWorkspace,Nothing}}(nothing)

    _ssl(y_in, mod) = compute_arima_likelihood(y_in, mod, 0, true)

    function _armafn(p, trans)
        par = copy(coef)
        par[mask] = p
        trarma = transform_arima_parameters(par, arma, trans)
        xxi = copy(x)

        Z = try
            update_arima(mod, trarma[1], trarma[2]; ss_g=SS_G)
        catch e
            @warn "Updating arima failed $e"
            return typemax(Float64)
        end

        if ncxreg > 0
            xxi = xxi .- xreg_mat * par[narma+1:narma+ncxreg]
        end
        resss = compute_arima_likelihood(xxi, Z, 0, false; workspace=kalman_ws[])

        nu = resss[3]
        nu <= 0 && return typemax(Float64)
        s2 = resss[1] / nu
        (s2 < 0 || isnan(s2) || s2 == Inf) && return typemax(Float64)
        result = 0.5 * (log(s2) + resss[2] / nu)
        return isnan(result) || result == Inf ? typemax(Float64) : result
    end

    function _armaCSS(p)
        par = copy(fixed)
        par[mask] .= p
        trarma = transform_arima_parameters(par, arma, false)
        x_in = copy(x)

        if ncxreg > 0
            x_in = x_in .- xreg_mat * par[narma+1:narma+ncxreg]
        end

        ross = compute_css_residuals(x_in, arma, trarma[1], trarma[2], ncond)
        sigma2 = ross[:sigma2]
        (sigma2 < 0 || isnan(sigma2) || sigma2 == Inf) && return typemax(Float64)
        result = 0.5 * log(sigma2)
        return isnan(result) || result == Inf ? typemax(Float64) : result
    end

    if method === :css
        if no_optim
            res = (converged=true, minimizer=zeros(0), minimum=_armaCSS(zeros(0)))
        else
            opt = optimize(p -> _armaCSS(p), init[mask], optim_method;
                param_scale = parscale[mask],
                step_sizes = get(optim_control, "ndeps", fill(1e-2, sum(mask))),
                max_iterations = get(optim_control, "maxit", 500))
            res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
        end

        if !res.converged
            @warn "CSS optimization convergence issue"
        end

        coef[mask] .= res.minimizer
        trarma = transform_arima_parameters(coef, arma, false)
        mod = initialize_arima_state(trarma[1], trarma[2], Delta; kappa=Float64(kappa), SSinit=SSinit)

        if ncxreg > 0
            x = x - xreg_mat * coef[narma+1:narma+ncxreg]
        end
        _ssl(x, mod)
        val = compute_css_residuals(x, arma, trarma[1], trarma[2], ncond)
        sigma2 = val[:sigma2]

        if no_optim
            var = zeros(0)
        else
            hessian = numerical_hessian(p -> _armaCSS(p), res.minimizer)
            var = inv(hessian * n_used)
        end

    else
        if method in (:css_ml, :ml)
            if method === :ml
                ncond = order.d + order.D * order.s
                ncond1 = order.p + order.P * order.s
                ncond += ncond1
            end

            if no_optim
                res = (converged=true, minimizer=zeros(sum(mask)), minimum=_armaCSS(zeros(0)))
            else
                opt = optimize(p -> _armaCSS(p), init[mask], optim_method;
                    param_scale = parscale[mask],
                    step_sizes = get(optim_control, "ndeps", fill(1e-2, sum(mask))),
                    max_iterations = get(optim_control, "maxit", 500))
                res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
            end

            if res.converged
                init[mask] .= res.minimizer
            end

            if arma[1] > 0 && !ar_check(init[1:arma[1]])
                error("Non-stationary AR part from CSS")
            end
            if arma[3] > 0 && !ar_check(init[(sum(arma[1:2])+1):(sum(arma[1:2])+arma[3])])
                error("Non-stationary seasonal AR part from CSS")
            end

            ncond = 0
        end

        if transform_pars
            init = inverse_arima_parameter_transform(init, arma)
            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                init[ind] .= ma_invert(init[ind])
            end
            if arma[4] > 0
                ind = (sum(arma[1:3])+1):(sum(arma[1:3])+arma[4])
                init[ind] .= ma_invert(init[ind])
            end
        end

        trarma = transform_arima_parameters(init, arma, transform_pars)
        mod = initialize_arima_state(trarma[1], trarma[2], Delta; kappa=Float64(kappa), SSinit=SSinit)

        rd = length(mod.a)
        d_len = length(mod.Delta)
        kalman_ws[] = KalmanWorkspace(rd, n, d_len, false)

        if no_optim
            res = (converged=true, minimizer=zeros(0), minimum=_armafn(zeros(0), transform_pars))
        else
            opt = optimize(p -> _armafn(p, transform_pars), init[mask], optim_method;
                param_scale = parscale[mask],
                step_sizes = get(optim_control, "ndeps", nothing),
                max_iterations = get(optim_control, "maxit", nothing))
            res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
        end

        if !res.converged
            @warn "Possible convergence problem"
        end

        coef[mask] .= res.minimizer

        if transform_pars
            if arma[2] > 0
                ind = (arma[1]+1):(arma[1]+arma[2])
                if all(mask[ind])
                    coef[ind] .= ma_invert(coef[ind])
                end
            end
            if arma[4] > 0
                ind = (sum(arma[1:3])+1):(sum(arma[1:3])+arma[4])
                if all(mask[ind])
                    coef[ind] .= ma_invert(coef[ind])
                end
            end

            if any(coef[mask] .!= res.minimizer)
                opt = optimize(p -> _armafn(p, true), coef[mask], optim_method;
                    param_scale = parscale[mask],
                    max_iterations = 0)
                res = (converged=opt.converged, minimizer=opt.minimizer, minimum=opt.minimum)
                hessian = numerical_hessian(p -> _armafn(p, true), res.minimizer)
                coef[mask] .= res.minimizer
            else
                hessian = numerical_hessian(p -> _armafn(p, true), res.minimizer)
            end

            A = compute_arima_transform_gradient(coef, arma)
            A = A[mask, mask]
            var = A' * ((hessian * n_used) \ A)
            coef = undo_arima_parameter_transform(coef, arma)
        else
            if no_optim
                var = zeros(0)
            else
                hessian = numerical_hessian(p -> _armafn(p, true), res.minimizer)
                var = inv(hessian * n_used)
            end
        end

        trarma = transform_arima_parameters(coef, arma, false)
        mod = initialize_arima_state(trarma[1], trarma[2], Delta; kappa=Float64(kappa), SSinit=SSinit)

        val = if ncxreg > 0
            _ssl(x - xreg_mat * coef[narma+1:narma+ncxreg], mod)
        else
            _ssl(x, mod)
        end
        sigma2 = val[1][1] / n_used
    end

    value = 2 * n_used * res.minimum + n_used + n_used * log(2 * π)
    aic = method !== :css ? value + 2 * sum(mask) + 2 : nothing
    loglik = -0.5 * value

    if ncxreg > 0 && !orig_xreg_flag
        ind = narma .+ (1:ncxreg)
        coef[ind] = S_svd.V * coef[ind]
        A = Matrix{Float64}(I, narma + ncxreg, narma + ncxreg)
        A[ind, ind] = S_svd.V
        A = A[mask, mask]
        var = A * var * transpose(A)
    end

    arima_coef = prep_coefs(arma, coef, nmxreg, ncxreg)
    resid = val[:residuals]
    fitted_vals = y_save .- resid

    if size(var) == (0,)
        var = reshape(var, 0, 0)
    end

    o = order
    s = model.order
    fit_method = if ncxreg > 0
        "Regression with ARIMA($(o.p),$(o.d),$(o.q))($(s.P),$(s.D),$(s.Q))[$(s.s)] errors"
    else
        "ARIMA($(o.p),$(o.d),$(o.q))($(s.P),$(s.D),$(s.Q))[$(s.s)]"
    end

    model.system = SARIMASystem{Fl}(
        Fl.(mod.phi), Fl.(mod.theta), Fl.(mod.Delta),
        Fl.(mod.Z), Fl.(mod.T), Fl.(mod.V),
        Fl(mod.h), Fl.(mod.a), Fl.(mod.P), Fl.(mod.Pn),
    )

    model.hyperparameters = HyperParameters{Fl}(
        Fl.(coef), Fl.(coef), mask, arima_coef.colnames, Fl.(fixed),
    )

    model.results = SARIMAResults{Fl}(
        arima_coef,
        sigma2,
        var,
        loglik,
        aic,
        nothing,
        nothing,
        nothing,
        resid,
        fitted_vals,
        res.converged,
        ncond,
        n_used,
    )

    model.method = method
    model.n_cond = ncond
    model.xreg = xreg_original
    model.transform_pars = transform_pars

    return model
end

"""
    arima(x, m; kwargs...) -> ArimaFit

Fit an ARIMA model. This is a wrapper that constructs a [`SARIMA`](@ref) model,
calls [`fit!`](@ref), and returns an [`ArimaFit`](@ref) for backward compatibility.
"""
function arima(
    x::AbstractArray,
    m::Int;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg::Union{Nothing, NamedMatrix} = nothing,
    include_mean::Bool = true,
    transform_pars::Bool = true,
    fixed::Union{Nothing, AbstractArray} = nothing,
    init::Union{Nothing, AbstractArray} = nothing,
    method::Symbol = :css_ml,
    n_cond::Union{Nothing, AbstractArray} = nothing,
    SSinit::Symbol = :gardner1980,
    optim_method::Symbol = :bfgs,
    optim_control::Dict = Dict(),
    kappa::Real = 1e6,
)
    x_clean = [ismissing(xi) ? NaN : Float64(xi) for xi in x]
    model = SARIMA(
        x_clean, m;
        order=order, seasonal=seasonal, xreg=xreg,
        include_mean=include_mean, transform_pars=transform_pars,
        fixed=fixed, init=init, method=method,
        SSinit=SSinit, optim_method=optim_method,
        optim_control=optim_control, kappa=kappa,
    )

    fit!(model)

    o = model.order
    r = model.results
    has_xreg_coefs = !isnothing(r) && length(r.coef.colnames) > sum(arma_vector(o)[1:4])

    fit_method = if has_xreg_coefs
        "Regression with ARIMA($(o.p),$(o.d),$(o.q))($(o.P),$(o.D),$(o.Q))[$(o.s)] errors"
    else
        "ARIMA($(o.p),$(o.d),$(o.q))($(o.P),$(o.D),$(o.Q))[$(o.s)]"
    end

    return ArimaFit(model, fit_method)
end
