using Polynomials

using Durbyn
using Durbyn.Arima

"""
    AutoArimaFit

A struct containing the results of an ARIMA model fit, extended to hold BIC, AICc, and the selected IC.

# Fields

- `coef::ArimaCoef`     — parameter vector
- `sigma2::Float64`     — innovation variance
- `var_coef::Matrix{Float64}` — coefficient covariance
- `mask::Vector{Bool}`  — estimated vs. fixed flags
- `loglik::Float64`     — maximized log‑likelihood
- `aic::Union{Float64,Nothing}` — Akaike IC
- `bic::Float64`        — Bayesian IC
- `aicc::Float64`       — corrected AIC
- `ic::Float64`         — chosen information criterion value
- `arma::Vector{Int}`   — model orders `[p,q,P,Q,s,d,D]`
- `residuals::Vector{Float64}` — model residuals
- `convergence_code::Bool` — optimizer success flag
- `n_cond::Int`         — observations conditioned out
- `nobs::Int`           — observations used
- `model::NamedTuple`   — metadata (names, call, series)
- `xreg::Any`           — regression matrix
"""
struct AutoArimaFit
    coef::ArimaCoef
    sigma2::Float64
    var_coef::Matrix{Float64}
    mask::Vector{Bool}
    loglik::Float64
    aic::Union{Float64,Nothing}
    bic::Union{Float64,Nothing}
    aicc::Union{Float64,Nothing}
    ic::String
    arma::Vector{Int}
    residuals::Vector{Float64}
    convergence_code::Bool
    n_cond::Int
    nobs::Int
    model::NamedTuple
    xreg::Any
end

"""
Construct an AutoArimaFit from an existing ArimaFit, seeding BIC/AICc/IC fields with NaN until recomputed.
"""
AutoArimaFit(fit::ArimaFit) = AutoArimaFit(
    fit.coef, fit.sigma2, fit.var_coef, fit.mask,
    fit.loglik, fit.aic,
    NaN, NaN, NaN,
    fit.arma, fit.residuals, fit.convergence_code,
    fit.n_cond, fit.nobs, fit.model, fit.xreg)

"""
    update_fit(fit::AutoArimaFit; 
               coef=fit.coef, sigma2=fit.sigma2, var_coef=fit.var_coef,
               mask=fit.mask, loglik=fit.loglik, aic=fit.aic,
               bic=fit.bic, aicc=fit.aicc, ic=fit.ic,
               arma=fit.arma, residuals=fit.residuals,
               convergence_code=fit.convergence_code,
               n_cond=fit.n_cond, nobs=fit.nobs,
               model=fit.model, xreg=fit.xreg)

Helper that propagates all 3 IC fields.
"""
function update_fit(fit::AutoArimaFit;
                    coef=fit.coef,
                    sigma2=fit.sigma2,
                    var_coef=fit.var_coef,
                    mask=fit.mask,
                    loglik=fit.loglik,
                    aic=fit.aic,
                    bic=fit.bic,
                    aicc=fit.aicc,
                    ic=fit.ic,
                    arma=fit.arma,
                    residuals=fit.residuals,
                    convergence_code=fit.convergence_code,
                    n_cond=fit.n_cond,
                    nobs=fit.nobs,
                    model=fit.model,
                    xreg=fit.xreg)
    return AutoArimaFit(
        coef, sigma2, var_coef, mask, loglik,
        aic, bic, aicc, ic,
        arma, residuals, convergence_code,
        n_cond, nobs, model, xreg)
end

"""
Usage in `myarima`: immediately wrap the raw ArimaFit before any update_fit calls.

```julia
# after calling the base `arima(...)`:
if fit_raw isa ArimaFit
    # convert to AutoArimaFit with empty ICs
    fit = AutoArimaFit(fit_raw)
else
    return Dict(:ic=>Inf)
end

# ... later, when computing ICs:
if !isnan(aic_val)
    bic_val  = aic_val + npar*(log(nstar)-2)
    aicc_val = aic_val + 2*npar*(npar+1)/(nstar-npar-1)
    ic_val   = ic==:bic  ? bic_val : ic==:aicc ? aicc_val : aic_val
    fit = update_fit(fit;
                     aic=aic_val,
                     bic=bic_val,
                     aicc=aicc_val,
                     ic=ic_val)
else
    fit = update_fit(fit; ic=Inf)
end
```




function update_fit(fit::ArimaFit;
                    coef = fit.coef,
                    sigma2 = fit.sigma2,
                    var_coef = fit.var_coef,
                    mask = fit.mask,
                    loglik = fit.loglik,
                    aic = fit.aic,
                    arma = fit.arma,
                    residuals = fit.residuals,
                    convergence_code = fit.convergence_code,
                    n_cond = fit.n_cond,
                    nobs = fit.nobs,
                    model = fit.model,
                    xreg = fit.xreg)
    return ArimaFit(coef, sigma2, var_coef, mask, loglik, aic,
                    arma, residuals, convergence_code, n_cond, nobs,
                    model, xreg)
end

function checkarima(fit::ArimaFit)
    try
        return any(isnan.(sqrt.(diag(fit.var_coef))))
    catch
        return true
    end
end

function trace_arima(order::PDQ, seasonal::PDQ, m::Int, constant::Bool, diffs::Int, use_season::Bool, ic_val)
    seasonal_str = use_season ? " (" * string(seasonal.p, ",", seasonal.d, ",", seasonal.q) * ")[" * string(m) * "]" : ""
    mean_str = diffs == 1 && constant ? " with drift" :
               diffs == 0 && constant ? " with non-zero mean" :
               diffs == 0 && !constant ? " with zero mean" : ""
    #@info "ARIMA(\$(order.p),\$(order.d),\$(order.q))\$seasonal_str\$mean_str: \$ic_val"
    println("ARIMA(", order.p, ",", order.d, ",", order.q, ")", seasonal_str, mean_str, ": ", ic_val)
end

function myarima(x::AbstractArray,
                 m::Int,
                 order::PDQ = PDQ(0, 0, 0),
                 seasonal::PDQ = PDQ(0, 0, 0);
                 constant::Bool = true,
                 ic::Symbol = :aic,
                 trace::Bool = false,
                 approximation::Bool = false,
                 offset::Real = 0,
                 xreg::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
                 transform_pars::Bool = true,
                 fixed = nothing,
                 init = nothing,
                 method::Union{String, Nothing} = nothing,
                 n_cond = nothing,
                 SSinit::String = "Gardner1980",
                 options::NelderMeadOptions = NelderMeadOptions(),
                 kappa::Float64 = 1e6)

    missing = ismissing.(x)
    firstnonmiss = findfirst(!, missing)
    lastnonmiss  = findlast(!, missing)
    n = sum(.!missing[firstnonmiss:lastnonmiss])
    diffs = order.d + seasonal.d
    use_season = (seasonal.p + seasonal.d + seasonal.q > 0) && m > 0

    if isnothing(method)
        method = approximation ? "CSS" : "CSS-ML"
    end

    println("The method: ", method)

    if diffs == 1 && constant
        drift = collect(1:length(x))
        xreg = isnothing(xreg) ? reshape(drift, :, 1) : hcat(drift, xreg)
    end

    fit = try
        if diffs == 1 && constant
            if use_season
                arima(x, m; order=order, seasonal=seasonal, xreg=xreg, method=method, include_mean=nothing)
            else
                arima(x, m; order=order, xreg=xreg, method=method, include_mean=nothing)
            end
        else
            if use_season
                #arima(x, m; order=order, seasonal=seasonal, xreg=xreg, include_mean=constant, method=method)
                arima(x, m, order = order, seasonal = seasonal, xreg = xreg, include_mean = constant, transform_pars = transform_pars, fixed = fixed, init = init, method = method, n_cond = n_cond, SSinit = SSinit, options = options, kappa = kappa,)
            else
                #arima(x, m; order=order, xreg=xreg, include_mean=constant, method=method)
                arima(x, m, order = order, seasonal = PDQ(0,0,0), xreg = xreg, include_mean = constant, transform_pars = transform_pars, fixed = fixed, init = init, method = method, n_cond = n_cond, SSinit = SSinit, options = options, kappa = kappa,)
           
            
            end
        end
    catch err
        if occursin("unused argument", String(err))
            rethrow(err)
        else
            return Dict(:ic => Inf)
        end
    end

    nxreg = isnothing(xreg) ? 0 : size(Matrix(xreg), 2)

    if fit isa ArimaFit
        nstar = n - order.d - seasonal.d * m

        if diffs == 1 && constant
            fit = update_fit(fit; xreg=xreg)
        end

        npar = count(fit.mask) + 1

        if method == "CSS"
            new_aic = offset + nstar * log(fit.sigma2) + 2 * npar
            fit = update_fit(fit; aic=new_aic)
        end

        aic_val = fit.aic === nothing ? NaN : fit.aic
        if !isnan(aic_val)
            bic_val  = aic_val + npar * (log(nstar) - 2)
            aicc_val = aic_val + 2 * npar * (npar + 1) / (nstar - npar - 1)
            ic_val = ic == :bic ? bic_val : ic == :aicc ? aicc_val : aic_val

            fit = update_fit(fit; aic  = aic_val, bic  = bic_val, aicc = aicc_val, ic   = ic_val)
        else
            fit = update_fit(fit; ic = Inf)
        end

        resid = fit.residuals
        new_sigma2 = sum(skipmissing(resid .^ 2)) / (nstar - npar + 1)
        fit = update_fit(fit; sigma2=new_sigma2)

        minroot = 2.0
        if order.p + seasonal.p > 0
            phi = get(fit.model, :phi, [])
            nonzero = findlast(abs.(phi) .> 1e-8)
            testvec = isnothing(nonzero) ? [] : phi[1:nonzero]
            if !isempty(testvec)
                try
                    proots = polyroots(Polynomial([1; -testvec]))
                    minroot = min(minroot, minimum(abs.(proots)))
                catch
                    ic_val = Inf
                end
            end
        end

        if order.q + seasonal.q > 0 && ic_val < Inf
            theta = get(fit.model, :theta, [])
            nonzero = findlast(abs.(theta) .> 1e-8)
            testvec = isnothing(nonzero) ? [] : theta[1:nonzero]
            if !isempty(testvec)
                try
                    proots = polyroots(Polynomial([1; testvec]))
                    minroot = min(minroot, minimum(abs.(proots)))
                catch
                    ic_val = Inf
                end
            end
        end

        if minroot < 1.01 || checkarima(fit)
            ic_val = Inf
        end

        if trace
            trace_arima(order, seasonal, m, constant, diffs, use_season, ic_val)
        end

        return fit
    else
        if trace
            trace_arima(order, seasonal, m, constant, diffs, use_season, Inf)
        end
        return Dict(:ic => Inf)
    end
end

function search_arima(x::AbstractArray, m::Int;
    d::Int = 0, D::Int = 0,
    max_p::Int = 5, max_q::Int = 5,
    max_P::Int = 2, max_Q::Int = 2,
    max_order::Int = 5, stationary::Bool = false,
    ic::Symbol = :aic, trace::Bool = false,
    approximation::Bool = false,
    xreg::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
    offset::Real = 0,
    allowdrift::Bool = true, allowmean::Bool = true,
    parallel::Bool = false, num_cores::Int = 2)

    # Adjust flags
    allowdrift &= (d + D) == 1
    allowmean &= (d + D) == 0
    maxK = (allowdrift || allowmean) ? 1 : 0

    best_ic = Inf
    bestfit = nothing
    constant = false

    if !parallel
        for i in 0:max_p, j in 0:max_q, I in 0:max_P, J in 0:max_Q
            if i + j + I + J <= max_order
                for K in 0:maxK
                    fit = myarima(x, m,
                        PDQ(i, d, j),
                        PDQ(I, D, J);
                        constant = (K == 1),
                        ic = ic,
                        trace = trace,
                        approximation = approximation,
                        offset = offset,
                        xreg = xreg
                    )
                    if fit.ic < best_ic
                        best_ic = fit.ic
                        bestfit = fit
                        constant = (K == 1)
                    end
                end
            end
        end
    else
        to_check = WhichModels(max_p, max_q, max_P, max_Q, maxK)

        function par_all_arima(desc)
            i, j, I, J, K = UndoWhichModels(desc)
            if i + j + I + J <= max_order
                fit = myarima(x, m,
                    PDQ(i, d, j),
                    PDQ(I, D, J);
                    constant = (K == 1),
                    ic = ic,
                    trace = trace,
                    approximation = approximation,
                    offset = offset,
                    xreg = xreg
                )
                return (fit, K == 1)
            else
                return nothing
            end
        end

        results = ThreadsX.map(par_all_arima, to_check)
        results = filter(!isnothing, results)

        for (fit, is_const) in results
            if fit.ic < best_ic
                bestfit = fit
                best_ic = fit.ic
                constant = is_const
            end
        end
    end

    if bestfit === nothing
        error("No ARIMA model could be estimated.")
    end

    # Optional re-fit without approximation
    if approximation
        if trace
            println("\n\n Now re-fitting the best model(s) without approximations...\n")
        end
        arma = bestfit.arma
        refit = myarima(x, m,
            PDQ(arma[1], arma[6], arma[2]),
            PDQ(arma[3], arma[7], arma[4]);
            constant = constant,
            ic = ic,
            trace = false,
            approximation = false,
            offset = offset,
            xreg = xreg
        )
        if refit.ic != Inf
            bestfit = refit
        else
            error("Refitting failed. Try again without approximation.")
        end
    end

    return bestfit
end


using Durbyn.Arima
using Durbyn

ap = air_passengers();

search_arima(ap, 12)

myarima(ap, 12, PDQ(0, 1, 1),  PDQ(0, 1, 1))