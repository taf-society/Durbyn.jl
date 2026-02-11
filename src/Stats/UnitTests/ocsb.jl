"""
    OCSB

Result object for the OCSB (Osborn-Chui-Smith-Birchenhall) seasonal unit root test.

# Fields
- `type::Symbol`: Deterministic component used in the test.
    - For OCSB this is fixed to `:seasonal` (test targets seasonal unit roots).
- `lag::Int`: AR lag order `p` used in the prewhitening and the final regression (selected per `lag_method`).
- `teststat::Float64`: OCSB test statistic (t-statistic on `Z5 = λ(B)ΔX_{t-m}` in the final regression).
- `cval::Vector{Float64}`: Critical values corresponding to `clevels` (from your critical-value provider).
- `clevels::Vector{Float64}`: Nominal significance levels (e.g. `[0.10, 0.05, 0.01]`).
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

    has_cvals = !isempty(x.cval) && !isempty(x.clevels) && length(x.cval) == length(x.clevels)
    if has_cvals
        try
            pv = round(pvalue(x); digits=4)
            println(io, "Approx. p-value (interp.): ", pv)
        catch
        end
    end

    if has_cvals
        println(io, "\nCritical values:")
        for (α, cv) in zip(x.clevels, x.cval)
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

@inline z_at(x, t, m) = x[t] - x[t-1] - x[t-m] + x[t-m-1]
@inline w_at(x, t, m) = x[t] - x[t-m]
@inline v_at(x, t)    = x[t] - x[t-1]

function fit_ocsb(x::AbstractVector{<:Real}, lag::Int, maxlag::Int, m::Int)
    @assert lag ≥ 0 && maxlag ≥ 0 && m ≥ 1
    n = length(x)

    t0 = m + 2 + max(maxlag, lag)
    @assert n ≥ t0 "not enough observations for m=$(m), lag=$(lag), maxlag=$(maxlag)"
    T  = t0:n
    L  = length(T)

    y_ar = Vector{Float64}(undef, L)
    X_ar = (lag == 0) ? Array{Float64}(undef, L, 0) : Array{Float64}(undef, L, lag)
    @inbounds for (i, t) in enumerate(T)
        y_ar[i] = z_at(x, t, m)
        if lag > 0
            for j in 1:lag
                X_ar[i, j] = z_at(x, t - j, m)
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
        s4 = w_at(x, t - 1, m)
        for j in 1:lag
            s4 -= λ[j] * w_at(x, t - 1 - j, m)
        end
        Z4[i] = s4
        s5 = v_at(x, t - m)
        for j in 1:lag
            s5 -= λ[j] * v_at(x, t - m - j)
        end
        Z5[i] = s5
    end

    Xlags = (lag == 0) ? Array{Float64}(undef, L, 0) : Array{Float64}(undef, L, lag)
    @inbounds for (i, t) in enumerate(T)
        if lag > 0
            for j in 1:lag
                Xlags[i, j] = z_at(x, t - j, m)
            end
        end
    end
    X_final = (lag == 0) ? hcat(Z4, Z5) : hcat(Xlags, Z4, Z5)
    return ols(y_ar, X_final)
end

function calculate_ocsb_critical_value(m::Int)
      log_m = log(m)
      a = -0.2850853 * (log_m - 0.7656451)
      b = (-0.05983644) * ((log_m - 0.7656451)^2)
      return -0.2937411 * exp(a + b) - 1.652202
end

rss(fit)   = sum(fit.residuals .^ 2)
nobs(fit)  = length(fit.residuals)
p_params(fit) = length(fit.coef)
k_params(fit) = p_params(fit) + 1

function loglik_ml_lm(fit)
    n = nobs(fit)
    r = rss(fit)
    return -0.5 * n * (log(2π) + 1 + log(r/n))
end

aic_lm(fit)  = -2 * loglik_ml_lm(fit) + 2 * k_params(fit)
bic_lm(fit)  = -2 * loglik_ml_lm(fit) + log(nobs(fit)) * k_params(fit)
aicc_lm(fit) = begin
    n = nobs(fit); k = k_params(fit)
    AIC = aic_lm(fit)
    AIC + (2k*(k+1)) / (n - k - 1)
end

"""
ocsb(x::AbstractVector{<:Real}, m::Int; lag_method::Symbol = :fixed,
        maxlag::Int = 0, clevels::AbstractVector{<:Real} = [0.10, 0.05, 0.01])

ocsb(; x::AbstractVector{<:Real}, m::Int, lag_method::String = "fixed",
        maxlag::Int = 0, clevels::AbstractVector{<:Real} = [0.10, 0.05, 0.01],)

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
  information criterion over p = 1:maxlag. R-style selection runs only if maxlag > 0.
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
Osborn DR, Chui APL, Smith J, and Birchenhall CR (1988). “Seasonality and the order of integration
for consumption.” Oxford Bulletin of Economics and Statistics 50(4): 361-377.

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
    clevels::AbstractVector{<:Real} = [0.10, 0.05, 0.01],
)
    @assert m ≥ 2 "Data must be seasonal (m ≥ 2) to use ocsb."

    lmeth = Symbol(lowercase(string(lag_method)))
    @assert lmeth in (:fixed, :aic, :bic, :aicc) "lag_method must be one of: :fixed, :AIC, :BIC, :AICc."

    chosen_lag = maxlag
    if maxlag > 0 && lmeth != :fixed
        fits = [fit_ocsb(x, p, maxlag, m) for p in 1:maxlag]
        icvals = zeros(Float64, maxlag)
        if lmeth == :aic
            for p in 1:maxlag; icvals[p] = aic_lm(fits[p]); end
        elseif lmeth == :bic
            for p in 1:maxlag; icvals[p] = bic_lm(fits[p]); end
        else
            for p in 1:maxlag; icvals[p] = aicc_lm(fits[p]); end
        end
        id = argmin(icvals) 
        chosen_lag = id - 1
    end

    reg = fit_ocsb(x, chosen_lag, chosen_lag, m)

    tstats = reg.coef ./ reg.se
    @assert length(tstats) ≥ 2 "Final regression must include Z4 and Z5 regressors."
    teststat = tstats[end]
    residuals = reg.residuals

    cval = calculate_ocsb_critical_value(m)

    return OCSB(
        :seasonal,
        chosen_lag,
        teststat,
        cval,
        Float64.(clevels),
        residuals,
    )
end

function ocsb(;
    x::AbstractVector{<:Real},
    m::Int,
    lag_method::String = "fixed",
    maxlag::Int = 0,
    clevels::AbstractVector{<:Real} = [0.10, 0.05, 0.01],
)

    lag_method = match_arg(lag_method, ["fixed", "aic", "bic", "aicc"])
    lag_method = Symbol(lag_method)
    return ocsb(x, m, lag_method = lag_method, maxlag = maxlag, clevels = clevels)
end
