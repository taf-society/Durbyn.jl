"""
    OCSB

Result object for the OCSB (Osborn-Chui-Smith-Birchenhall) seasonal unit root test.

# Fields
- `type::Symbol`: Deterministic component used in the test.
    - For OCSB this is fixed to `:seasonal` (test targets seasonal unit roots).
- `lag::Int`: AR lag order `p` used in the prewhitening and the final regression (selected per `lag_method`).
- `teststat::Float64`: OCSB test statistic (t-statistic on `Z5 = λ(B)ΔX_{t-m}` in the final regression).
- `cval::Float64`: Critical value from the smoothed OCSB surface.
- `clevels::Vector{Float64}`: Nominal significance levels (e.g. `[0.10, 0.05, 0.01]`).
- `coef::Vector{Float64}`: OLS coefficients from the final OCSB regression.
- `se::Vector{Float64}`: OLS standard errors for `coef`.
- `testreg::NamedTuple`: OLS fit details from the final regression.
- `res::Vector{Float64}`: Residuals from the final OCSB regression.

# Notes
The tested regression (Osborn et al., 1988) is

ΔΔₘ Xₜ = β₁ Z₄,ₜ₋₁ + β₂ Z₅,ₜ₋ₘ + α₁ ΔΔₘ Xₜ₋₁ + ⋯ + αₚ ΔΔₘ Xₜ₋ₚ + uₜ,

where `Z₄,ₜ = λ̂(B) Δₘ Xₜ`, `Z₅,ₜ = λ̂(B) Δ Xₜ`, and `λ̂(B)` is an AR(p) operator estimated
from ΔΔₘ Xₜ over the same time window. The reported test statistic is the t-statistic of β₂.

References:
Osborn DR, Chui APL, Smith J, Birchenhall CR (1988), *Seasonality and the order of integration for consumption*, Oxford Bulletin of Economics and Statistics 50(4):361-377.
"""
struct OCSB
    type::Symbol
    lag::Int
    teststat::Float64
    cval::Float64
    clevels::Vector{Float64}
    coef::Vector{Float64}
    se::Vector{Float64}
    testreg::NamedTuple
    res::Vector{Float64}
end

import Base: show, summary

"""
    summary(x::OCSB) -> String

Concise one-liner with the key bits: test statistic, type, and lag.
Includes an approximate p-value when critical values are available.
"""
function summary(x::OCSB)
    stat = round(x.teststat; digits=4)
    tstr = x.type === :seasonal ? "seasonal" : String(x.type)
    pstr = try
        pv = round(pvalue(x); digits=4)
        ", p=$(pv)"
    catch
        ""
    end
    return "OCSB(stat=$(stat), type=$(tstr), lag=$(x.lag)$(pstr))"
end

function Base.propertynames(::OCSB; private::Bool=false)
    base = (
        :type, :lag, :teststat, :cval, :clevels, :coef, :se, :testreg, :res,
        :deterministic, :lags, :lag_order, :stats, :primary_cval, :critical_values,
        :coefficients, :standard_errors, :regression, :residuals,
    )
    return private ? base : base
end

function Base.getproperty(x::OCSB, sym::Symbol)
    if sym === :deterministic
        return getfield(x, :type)
    elseif sym === :lags
        return getfield(x, :lag)
    elseif sym === :lag_order
        return getfield(x, :lag)
    elseif sym === :stats
        return (ocsb = getfield(x, :teststat),)
    elseif sym === :primary_cval
        return [getfield(x, :cval)]
    elseif sym === :critical_values
        level_count = max(length(getfield(x, :clevels)), 1)
        return (ocsb = fill(getfield(x, :cval), level_count),)
    elseif sym === :coefficients
        return getfield(x, :coef)
    elseif sym === :standard_errors
        return getfield(x, :se)
    elseif sym === :regression
        return getfield(x, :testreg)
    elseif sym === :residuals
        return getfield(x, :res)
    end
    return getfield(x, sym)
end

"""
Pretty printer for plain-text displays.
Respects `io[:compact]` and prints a compact single-line form if requested.
"""
function show(io::IO, ::MIME"text/plain", x::OCSB)
    if get(io, :compact, false)
        print(io, "OCSB(", round(x.teststat; digits=4), ")")
        return
    end

    println(io, "OCSB Seasonal Unit Root Test")
    dtype = x.type === :seasonal ? "seasonal" : String(x.type)
    println(io, "Deterministic component (type): ", dtype)
    println(io, "AR lag order (p): ", x.lag)
    println(io, "Test statistic (t on Z₅): ", round(x.teststat; digits=4))

    cvals = fill(x.cval, length(x.clevels))
    has_cvals = !isempty(cvals) && !isempty(x.clevels)
    if has_cvals
        try
            pv = round(pvalue(x); digits=4)
            println(io, "Approx. p-value (interp.): ", pv)
        catch
        end
    end

    if has_cvals
        println(io, "\nCritical values:")
        for (α, cv) in zip(x.clevels, cvals)
            lvl = isa(α, Number) ? "$(Int(round(100α)))%" : string(α)
            println(io, "  ", lpad(lvl, 4), " : ", round(cv; digits=4))
        end
    end
end

"""
Fallback `show` for non-mime contexts.
Defers to text/plain unless `:compact` is set, in which case we print a short form.
"""
function show(io::IO, x::OCSB)
    if get(io, :compact, false)
        print(io, "OCSB(", round(x.teststat; digits=4), ")")
    else
        show(io, MIME("text/plain"), x)
    end
end

@inline _delta_delta_m(x, t, m) = x[t] - x[t-1] - x[t-m] + x[t-m-1]
@inline _delta_m(x, t, m) = x[t] - x[t-m]
@inline _delta(x, t)    = x[t] - x[t-1]

function _fit_ocsb_regression(x::AbstractVector{<:Real}, lag::Int, maxlag::Int, m::Int)
    (lag ≥ 0 && maxlag ≥ 0 && m ≥ 1) || throw(ArgumentError("lag and maxlag must be non-negative; m must be ≥ 1"))
    n = length(x)

    t0 = m + 2 + max(maxlag, lag)
    n ≥ t0 || throw(ArgumentError("not enough observations for m=$(m), lag=$(lag), maxlag=$(maxlag)"))
    T  = t0:n
    L  = length(T)

    y_ar = Vector{Float64}(undef, L)
    X_ar = (lag == 0) ? Array{Float64}(undef, L, 0) : Array{Float64}(undef, L, lag)
    @inbounds for (i, t) in enumerate(T)
        y_ar[i] = _delta_delta_m(x, t, m)
        if lag > 0
            for j in 1:lag
                X_ar[i, j] = _delta_delta_m(x, t - j, m)
            end
        end
    end
    λ = if lag == 0
        Float64[]
    else
        ols(y_ar, X_ar).coef
    end

    Z4 = Vector{Float64}(undef, L)
    Z5 = Vector{Float64}(undef, L)
    @inbounds for (i, t) in enumerate(T)
        s4 = _delta_m(x, t - 1, m)
        for j in 1:lag
            s4 -= λ[j] * _delta_m(x, t - 1 - j, m)
        end
        Z4[i] = s4
        s5 = _delta(x, t - m)
        for j in 1:lag
            s5 -= λ[j] * _delta(x, t - m - j)
        end
        Z5[i] = s5
    end

    Xlags = (lag == 0) ? Array{Float64}(undef, L, 0) : Array{Float64}(undef, L, lag)
    @inbounds for (i, t) in enumerate(T)
        if lag > 0
            for j in 1:lag
                Xlags[i, j] = _delta_delta_m(x, t - j, m)
            end
        end
    end
    X_final = (lag == 0) ? hcat(Z4, Z5) : hcat(Xlags, Z4, Z5)
    return ols(y_ar, X_final)
end

function _ocsb_critical_value(m::Int)
      log_m = log(m)
      a = -0.2850853 * (log_m - 0.7656451)
      b = (-0.05983644) * ((log_m - 0.7656451)^2)
      return -0.2937411 * exp(a + b) - 1.652202
end

_residual_sum_squares(fit)   = sum(fit.residuals .^ 2)
_n_observations(fit)  = length(fit.residuals)
_n_params(fit) = length(fit.coef)
_n_params_total(fit) = _n_params(fit) + 1

function _loglik_ml(fit)
    n = _n_observations(fit)
    r = _residual_sum_squares(fit)
    return -0.5 * n * (log(2π) + 1 + log(r/n))
end

_aic(fit)  = -2 * _loglik_ml(fit) + 2 * _n_params_total(fit)
_bic(fit)  = -2 * _loglik_ml(fit) + log(_n_observations(fit)) * _n_params_total(fit)
_aicc(fit) = begin
    n = _n_observations(fit); k = _n_params_total(fit)
    AIC = _aic(fit)
    AIC + (2k*(k+1)) / (n - k - 1)
end

"Approximate OCSB p-value surrogate using the provided critical-value threshold."
function pvalue(x::OCSB)
    if isempty(x.clevels)
        return NaN
    end
    left_tail_level = minimum(x.clevels)
    right_tail_level = maximum(x.clevels)
    return x.teststat <= x.cval ? left_tail_level : right_tail_level
end

"""
ocsb(x::AbstractVector{<:Real}, m::Int; lag_method::Symbol = :fixed,
        maxlag::Int = 0, clevels::AbstractVector{<:Real} = [0.10, 0.05, 0.01])

Osborn-Chui-Smith-Birchenhall (OCSB) test for seasonal unit roots.

Model
-----
The regression follows Osborn et al. (1988):

ΔΔₘ X_t = β₁ Z_{4,t-1} + β₂ Z_{5,t-m} + α₁ ΔΔₘ X_{t-1} + ⋯ + α_p ΔΔₘ X_{t-p},

where

- Z_{4,t} = λ̂(B) Δₘ X_t,
- Z_{5,t} = λ̂(B) Δ X_t, and
- λ̂(B) is an AR(p) lag operator with coefficients estimated from an AR(p) fit to ΔΔₘ X_t.

Here, Δ is the first-difference operator and Δₘ is the seasonal difference with period m.

Arguments
---------
- x::AbstractVector{<:Real}: Time series.
- m::Int: Seasonal period (must be ≥ 2).

Keyword Arguments
-----------------
- lag_method::Symbol = :fixed: Lag selection method. One of :fixed, :AIC, :BIC, or :AICc
  (case-insensitive; e.g. :aic and :AIC are both accepted).
- maxlag::Int = 0: If lag_method == :fixed, the AR order is set to maxlag. Otherwise, maxlag
  is the maximum number of lags considered in a selection procedure that minimizes the chosen
  information criterion over p = 1:maxlag. Lag selection runs only if maxlag > 0.
- clevels::AbstractVector{<:Real} = [0.10, 0.05, 0.01]: Nominal levels for critical values.


Details
-------
When lag_method == :fixed, the number of lagged differences in the regression is fixed to maxlag.
Otherwise, the lag order is chosen to minimize the specified information criterion over
p = 1, …, maxlag. Supported criteria are AIC, BIC, and the small-sample corrected AIC (AICc), with

AICc = AIC + (2k(k+1)) / (n - k - 1),

where k is the number of estimated parameters and n is the number of usable observations in the
regression.

Critical values for the OCSB statistic are based on simulation and smoothed across seasonal periods
to provide values for arbitrary m.

Returns
-------
An OCSB struct with fields:

- type::Symbol      — test type (e.g. :seasonal)
- lag::Int          — selected AR order
- teststat::Float64 — OCSB test statistic (t-statistic on Z_{5,t-m})
- cval::Vector{Float64}    — critical values corresponding to clevels
- clevels::Vector{Float64} — nominal critical value levels
- res::AbstractVector{<:Real} — regression residuals

Notes
-----
- The final regression must include the Z₄ and Z₅ regressors; an assertion will trigger if not.
- Symbols for lag_method are matched case-insensitively.

References
----------
Osborn DR, Chui APL, Smith J, and Birchenhall CR (1988). "Seasonality and the order of integration
for consumption." Oxford Bulletin of Economics and Statistics 50(4): 361-377.

Examples
--------
y = randn(200)
ocsb = ocsb(y; m=12, lag_method=:AIC, maxlag=12)
ocsb = ocsb(y; m=4,  lag_method=:fixed, maxlag=2)

"""
function ocsb(
    x::AbstractVector{<:Real},
    m::Int;
    lag_method::Symbol = :fixed,
    maxlag::Int = 0,
    clevels::AbstractVector{<:Real} = [0.05],
)
    m ≥ 2 || throw(ArgumentError("Data must be seasonal (m ≥ 2) to use ocsb."))

    lag_method_lower = Symbol(lowercase(string(lag_method)))
    lag_method_lower in (:fixed, :aic, :bic, :aicc) || throw(ArgumentError("lag_method must be one of: :fixed, :aic, :bic, :aicc."))

    selected_lag = maxlag
    if maxlag > 0 && lag_method_lower != :fixed
        fits = [_fit_ocsb_regression(x, p, maxlag, m) for p in 1:maxlag]
        ic_values = zeros(Float64, maxlag)
        if lag_method_lower == :aic
            for p in 1:maxlag; ic_values[p] = _aic(fits[p]); end
        elseif lag_method_lower == :bic
            for p in 1:maxlag; ic_values[p] = _bic(fits[p]); end
        else
            for p in 1:maxlag; ic_values[p] = _aicc(fits[p]); end
        end
        id = argmin(ic_values)
        selected_lag = id - 1
    end

    reg = _fit_ocsb_regression(x, selected_lag, selected_lag, m)

    tstats = reg.coef ./ reg.se
    length(tstats) ≥ 2 || throw(ArgumentError("Final regression must include Z4 and Z5 regressors."))
    teststat = tstats[end]
    residuals = reg.residuals

    cval = _ocsb_critical_value(m)

    return OCSB(
        :seasonal,
        selected_lag,
        teststat,
        cval,
        Float64.(clevels),
        Vector{Float64}(reg.coef),
        Vector{Float64}(reg.se),
        (β = reg.coef, se = reg.se, residuals = reg.residuals, σ2 = reg.sigma2, df_residual = reg.df_residual),
        residuals,
    )
end
