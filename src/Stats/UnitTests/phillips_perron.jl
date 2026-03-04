"""
    struct PhillipsPerron

Container for the Phillips-Perron (PP) unit root test results.

# Fields
- `type::Symbol`: Test type. One of `:Z_alpha` or `:Z_tau`.
- `model::Symbol`: Deterministic specification used in the test regression.
    - `:constant` → regression includes an intercept.
    - `:trend`    → regression includes an intercept and linear trend.
- `lag::Int`: Truncation lag (`max_lag`) used in the Bartlett long-run variance estimator.
- `teststat::Float64`: Value of the PP test statistic of the requested `type`.
- `cval::Vector{Float64}`: Finite-sample critical values for the requested `model` when `type == :Z_tau`.
  Order corresponds to `clevels`. For `:Z_alpha`, entries are `NaN`.
- `clevels::Vector{Float64}`: Significance levels associated with `cval` (by default `[0.01, 0.05, 0.10]`).
- `testreg::NamedTuple`: Results from the auxiliary OLS regression used by the test, with fields
  `β` (coefficients), `se` (standard errors), `residuals`, `σ2` (residual variance), and `df_residual`.
- `auxstat::Vector{Float64}`: Additional Z-statistics for the deterministic components of the test regression:
    - For `model == :constant`: `[Z_tau_mu]` (intercept).
    - For `model == :trend`: `[Z_tau_mu, Z_tau_beta]` (intercept and trend).
- `res::Vector{Float64}`: Residuals from the auxiliary regression.
"""
struct PhillipsPerron
    type::Symbol
    model::Symbol
    lag::Int
    teststat::Float64
    cval::Vector{Float64}
    clevels::Vector{Float64}
    testreg::NamedTuple
    auxstat::Vector{Float64}
    res::Vector{Float64}
end

function Base.propertynames(::PhillipsPerron; private::Bool=false)
    base = (
        :type, :model, :lag, :teststat, :cval, :clevels, :testreg, :auxstat, :res,
        :deterministic, :lags, :lag_order, :stats, :primary_cval, :critical_values,
        :coefficients, :standard_errors, :regression, :residuals,
    )
    return private ? base : base
end

function Base.getproperty(x::PhillipsPerron, sym::Symbol)
    if sym === :deterministic
        return getfield(x, :model)
    elseif sym === :lags
        return getfield(x, :lag)
    elseif sym === :lag_order
        return getfield(x, :lag)
    elseif sym === :stats
        if getfield(x, :type) === :Z_tau
            return (z_tau = getfield(x, :teststat), z_tau_aux = getfield(x, :auxstat))
        else
            return (z_alpha = getfield(x, :teststat), z_tau_aux = getfield(x, :auxstat))
        end
    elseif sym === :primary_cval
        return getfield(x, :cval)
    elseif sym === :critical_values
        if getfield(x, :type) === :Z_tau
            return (z_tau = getfield(x, :cval),)
        else
            return (z_alpha = getfield(x, :cval),)
        end
    elseif sym === :coefficients
        return getfield(x, :testreg).β
    elseif sym === :standard_errors
        return getfield(x, :testreg).se
    elseif sym === :regression
        return getfield(x, :testreg)
    elseif sym === :residuals
        return getfield(x, :res)
    end
    return getfield(x, sym)
end

"""
    phillips_perron(x; type=:Z_alpha, model=:constant, lags=:short, use_lag=nothing) -> PhillipsPerron

    Phillips and Perron (1988) unit root test. Computes either the `:Z_alpha` or `:Z_tau`
statistic using a Bartlett window to correct for serial correlation and heteroskedasticity.
Also reports Z-statistics for the deterministic terms in the test regression.

This is a Julia reimplementation of `ur.pp` from **urca**, following the same formulas
(including finite-sample critical values for `:Z_tau`).

# Arguments
- `x`: Univariate series (any iterable). Missing values are dropped before computation.

# Keyword Arguments
- `type::Symbol = :Z_tau`: Test type. One of `:Z_tau` or `:Z_alpha`.
- `model::Symbol = :constant`: Deterministic specification in the test regression.
    - `:constant` → `y_t = α + ρ y_{t-1} + u_t`
    - `:trend`    → `y_t = α + ρ y_{t-1} + β·t + u_t` with centered trend `t - n/2`
- `lags::Symbol = :short`: Rule for the Bartlett truncation lag `max_lag` if `use_lag` is not given.
    - `:short` → `max_lag = ⌊ 4 (n/100)^0.25 ⌋`
    - `:long`  → `max_lag = ⌊12 (n/100)^0.25 ⌋`
- `use_lag::Union{Nothing,Int} = nothing`: Manually specify `max_lag`. If negative, a warning is issued
  and the `:short` rule is used.

# Details
Let `y_t = x_{t}` for `t=2,…,n` and `y_{t-1} = x_{t-1}`. The function runs the OLS regression implied by `model`,
computes the residual variance `s`, and forms a Bartlett long-run variance estimate
`σ̂² = s + (2/n) ∑_{ℓ=1}^{max_lag} (1 - ℓ/(max_lag+1)) * ∑_{t=ℓ+1}^{n} e_t e_{t-ℓ}`.
From this it constructs the PP statistics `Z_tau` or `Z_alpha` as in Phillips-Perron (1988).
Finite-sample critical values (MacKinnon-style approximations used in **urca**) are returned for `:Z_tau`
and `model ∈ {:constant, :trend}`; for `:Z_alpha` the critical values are not provided (`NaN`).

# Returns
A `PhillipsPerron` struct with:
- `teststat`: PP statistic of the requested `type`;
- `cval`, `clevels`: critical values and their levels (if available);
- `auxstat`: Z-statistics for deterministic terms (intercept; and trend if included);
- `lag`: Bartlett truncation lag used;
- `testreg`: OLS regression internals; and
- `res`: residuals of the test regression.

# References
- Phillips, P. C. B. and Perron, P. (1988). *Testing for a unit root in time series regression*. **Biometrika**, 75(2), 335-346.
- MacKinnon, J. G. (1991). *Critical Values for Cointegration Tests*, in Engle & Granger (eds.), *Long-Run Economic Relationships*. Oxford.

# See also
`PhillipsPerron`, `pvalue(::PhillipsPerron)` (if defined), and helper routines for OLS and Bartlett LRV.

# Examples
```julia
using Random, Statistics

# Stationary AR(1)
Random.seed!(42)
n = 300
ε = randn(n)
x = similar(ε)
x[1] = ε[1]
for t in 2:n
    x[t] = 0.6x[t-1] + ε[t]
end

pp = phillips_perron(x; type=:Z_tau, model=:constant, lags=:short)
pp.teststat, pp.cval, pp.clevels

# Trend case
pp_tr = phillips_perron(x; type=:Z_tau, model=:trend, use_lag=4)
pp_tr.auxstat  # Z-tau-mu and Z-tau-beta

# Random walk (should fail to reject unit root more often)
rw = cumsum(randn(n))
pp_rw = phillips_perron(rw; type=:Z_tau, model=:constant)
```
"""
function phillips_perron(
    x;
    type::Symbol = :Z_alpha,
    model::Symbol = :constant,
    lags::Symbol = :short,
    use_lag::Union{Nothing,Int} = nothing,
)
    xv = _skipmissing_to_vec(x)
    n0 = length(xv)
    n0 >= 2 || throw(ArgumentError("series too short for PP test"))

    y = @view xv[2:end]
    y1 = @view xv[1:end-1]
    n = length(y)

    lags in (:short, :long) || throw(ArgumentError("lags must be :short or :long"))
    model in (:constant, :trend) || throw(ArgumentError("model must be :constant or :trend"))
    type in (:Z_alpha, :Z_tau) || throw(ArgumentError("type must be :Z_alpha or :Z_tau"))

    max_lag = if !isnothing(use_lag)
        L = Int(use_lag)
        if L < 0
            @warn "use_lag must be a nonnegative integer; using lags=:short default."
            L = trunc(Int, 4 * (n / 100)^0.25)
        end
        L
    elseif lags == :short
        trunc(Int, 4 * (n / 100)^0.25)
    else
        trunc(Int, 12 * (n / 100)^0.25)
    end

    if model == :trend

        cval = [
            -3.9638 - 8.353 / n - 47.44 / (n^2),
            -3.4126 - 4.039 / n - 17.83 / (n^2),
            -3.1279 - 2.418 / n - 7.58 / (n^2),
        ]
        clevels = [0.01, 0.05, 0.10]

        trend = collect(1:n) .- n / 2
        X = hcat(ones(Float64, n), Vector{Float64}(y1), Vector{Float64}(trend))
        fit = _ols(y, X)
        β = fit.β
        se = fit.se
        res = fit.residuals

        intercept_tstat = β[1] / se[1]
        trend_tstat = β[3] / se[3]
        s = sum(res .^ 2) / n

        ybar_var = sum((y .- mean(y)) .^ 2) / n^2
        sum_y_sq_scaled = sum(y .^ 2) / n^2
        sum_ty_scaled = (sum((1:n) .* y)) / n^(5 / 2)
        sum_y_scaled = sum(y) / n^(3 / 2)

        add = _bartlett_lrv(res, n, max_lag)
        long_run_variance = s + add
        λ = 0.5 * (long_run_variance - s)
        λ′ = λ / long_run_variance

        M =
            (1 - n^(-2)) * sum_y_sq_scaled - 12 * sum_ty_scaled^2 + 12 * (1 + 1 / n) * sum_ty_scaled * sum_y_scaled -
            (4 + 6 / n + 2 / n^2) * sum_y_scaled^2

        z_intercept =
            sqrt(s / long_run_variance) * intercept_tstat - λ′ * sqrt(long_run_variance) * sum_y_scaled / (sqrt(M) * sqrt(M + sum_y_scaled^2))
        z_trend =
            sqrt(s / long_run_variance) * trend_tstat -
            λ′ * sqrt(long_run_variance) * (0.5 * sum_y_scaled - sum_ty_scaled) / (sqrt(M / 12) * sqrt(ybar_var))

        aux = [round(z_intercept, digits = 4), round(z_trend, digits = 4)]

        if type == :Z_tau
            tstat = (β[2] - 1) / se[2]
            teststat = sqrt(s / long_run_variance) * tstat - λ′ * sqrt(long_run_variance) / sqrt(M)
        else
            α = β[2]
            teststat = n * (α - 1) - λ / M
            cval = fill(NaN, 3)
        end

        return PhillipsPerron(
            type,
            model,
            max_lag,
            teststat,
            cval,
            clevels,
            (β = β, se = se, residuals = res, σ2 = fit.σ2, df_residual = fit.df_residual),
            aux,
            res
        )

    else
        cval = [
            -3.4335 - 5.999 / n - 29.25 / (n^2),
            -2.8621 - 2.738 / n - 8.36 / (n^2),
            -2.5671 - 1.438 / n - 4.48 / (n^2),
        ]
        clevels = [0.01, 0.05, 0.10]

        X = hcat(ones(Float64, n), Vector{Float64}(y1))
        fit = _ols(y, X)
        β = fit.β
        se = fit.se
        res = fit.residuals

        intercept_tstat = β[1] / se[1]
        s = sum(res .^ 2) / n

        ybar_var = sum((y .- mean(y)) .^ 2) / n^2
        sum_y_sq_scaled = sum(y .^ 2) / n^2
        sum_y_scaled = sum(y) / n^(3 / 2)

        add = _bartlett_lrv(res, n, max_lag)
        long_run_variance = s + add
        λ = 0.5 * (long_run_variance - s)
        λ′ = λ / long_run_variance

        z_intercept =
            sqrt(s / long_run_variance) * intercept_tstat + λ′ * sqrt(long_run_variance) * sum_y_scaled / (sqrt(sum_y_sq_scaled) * sqrt(ybar_var))
        aux = [round(z_intercept, digits = 4)]

        if type == :Z_tau
            tstat = (β[2] - 1) / se[2]
            teststat = sqrt(s / long_run_variance) * tstat - λ′ * sqrt(long_run_variance) / sqrt(ybar_var)
        else
            α = β[2]
            teststat = n * (α - 1) - λ / ybar_var
            cval = fill(NaN, 3)
        end

        return PhillipsPerron(
            type,
            model,
            max_lag,
            teststat,
            cval,
            clevels,
            (β = β, se = se, residuals = res, σ2 = fit.σ2, df_residual = fit.df_residual),
            aux,
            res)
    end
end


"Approximate p-value for PP by linear interpolation between critical values (when present)."
function pvalue(u::PhillipsPerron)
    if any(isnan, u.cval)
        return NaN
    else
        return _pvalue_from_cvals(u.teststat, u.cval, u.clevels)
    end
end


summary(x::PhillipsPerron) =
    "PhillipsPerron(teststat=$(round(x.teststat, digits=4)), type=$(x.type), model=$(x.model), lag=$(x.lag))"

function show(io::IO, ::MIME"text/plain", x::PhillipsPerron)
    println(io, "Phillips-Perron Unit Root Test")

    println(io, "Deterministic component (model): ",
        x.model == :trend ? "trend (intercept + trend)" : "constant (intercept only)")

    println(io, "Test type: ", x.type == :Z_tau ? "Z-tau" : "Z-alpha")

    println(io, "Lag truncation (lmax): ", x.lag)

    println(io, "\nThe value of the test statistic is: ", round(x.teststat, digits=4))

    if !isempty(x.auxstat)
        println(io, "\nAuxiliary statistics:")
        if x.model == :trend
            labels = ["Z-tau-mu", "Z-tau-beta"]
        else
            labels = ["Z-tau-mu"]
        end
        for (lab, val) in zip(labels, x.auxstat)
            println(io, "  ", rpad(lab, 10), ": ", round(val, digits=4))
        end
    end

    if !isempty(x.cval) && !isempty(x.clevels) && length(x.cval) == length(x.clevels)
        println(io, "\nCritical values:")
        for (α, cv) in zip(x.clevels, x.cval)
            level = isa(α, Number) ? Int(round(100α)) : α
            println(io, "  ", lpad("$(level)%", 4), " : ", round(cv, digits=4))
        end
    end
end

function show(io::IO, x::PhillipsPerron)

    if get(io, :compact, false)
        print(io, "PhillipsPerron($(round(x.teststat, digits=4)))")
    else
        show(io, MIME("text/plain"), x)
    end
end
