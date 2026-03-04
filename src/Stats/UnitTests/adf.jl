"""
    ADF

Container for Augmented Dickey-Fuller (ADF) unit root test results.

# Fields
- `model::Symbol`: Deterministic component in the regression (`:none`, `:drift`, `:trend`).
- `cval::Matrix{Float64}`: Critical values; rows align with `testnames`, columns with `clevels`.
- `clevels::Vector{Float64}`: Significance levels corresponding to `cval` columns.
- `lags::Int`: Selected augmentation lag order `k`.
- `tau::Float64`: ADF tau statistic (t-ratio on the lagged level term).
- `phi1::Union{Float64,Missing}`: Joint F-statistic for `:drift` (`H0: beta1 = mu = 0`).
- `phi2::Union{Float64,Missing}`: Joint F-statistic for `:trend` (`H0: beta1 = mu = delta = 0`).
- `phi3::Union{Float64,Missing}`: Joint F-statistic for `:trend` (`H0: beta1 = delta = 0`).
- `beta::Vector{Float64}`: OLS coefficient estimates from the selected full regression.
- `se::Vector{Float64}`: OLS standard errors for `beta`.
- `testreg::NamedTuple`: OLS regression internals for the selected full regression.
- `res::Vector{Float64}`: Residuals from the selected full regression.
- `testnames::Vector{Symbol}`: Names of rows in `cval` (e.g. `[:tau2, :phi1]`).

# Notes
`ADF` also supports compatibility aliases:
- `result.lag` (alias of `result.lags`)
- `result.teststat` returning a named tuple of reported statistics.
"""
struct ADF
    model::Symbol
    cval::Matrix{Float64}
    clevels::Vector{Float64}
    lags::Int
    tau::Float64
    phi1::Union{Float64,Missing}
    phi2::Union{Float64,Missing}
    phi3::Union{Float64,Missing}
    beta::Vector{Float64}
    se::Vector{Float64}
    testreg::NamedTuple
    res::Vector{Float64}
    testnames::Vector{Symbol}
end


function _normalize_symbol(sym::Symbol)
    return Symbol(lowercase(String(sym)))
end

function _resolve_adf_model(model::Union{Nothing,Symbol}, type::Union{Nothing,Symbol})
    if !isnothing(model) && !isnothing(type)
        _normalize_symbol(model) == _normalize_symbol(type) ||
            throw(ArgumentError("model and type must match when both are provided"))
    end
    raw = isnothing(model) ? (isnothing(type) ? :none : type) : model
    out = _normalize_symbol(raw)
    return _check_arg(out, (:none, :drift, :trend), "model")
end

function _resolve_adf_criterion(
    criterion::Union{Nothing,Symbol},
    selectlags::Union{Nothing,Symbol},
    lags::Union{Nothing,Int},
    k_max::Union{Nothing,Int},
)
    if !isnothing(criterion) && !isnothing(selectlags)
        _normalize_symbol(criterion) == _normalize_symbol(selectlags) ||
            throw(ArgumentError("criterion and selectlags must match when both are provided"))
    end

    if isnothing(criterion) && isnothing(selectlags)
        return (isnothing(k_max) || !isnothing(lags)) ? :fixed : :aic
    end

    raw = isnothing(criterion) ? selectlags : criterion
    out = _normalize_symbol(raw)
    return _check_arg(out, (:fixed, :aic, :bic), "criterion")
end

function _adf_critical_values(model::Symbol, n_obs::Int)
    row = n_obs < 25 ? 1 :
          n_obs < 50 ? 2 :
          n_obs < 100 ? 3 :
          n_obs < 250 ? 4 :
          n_obs < 500 ? 5 : 6

    if model == :none
        cval_tau1 = [
            -2.66 -1.95 -1.60;
            -2.62 -1.95 -1.61;
            -2.60 -1.95 -1.61;
            -2.58 -1.95 -1.62;
            -2.58 -1.95 -1.62;
            -2.58 -1.95 -1.62
        ]
        return reshape(cval_tau1[row, :], 1, 3), [:tau1]
    elseif model == :drift
        cval_tau2 = [
            -3.75 -3.00 -2.63;
            -3.58 -2.93 -2.60;
            -3.51 -2.89 -2.58;
            -3.46 -2.88 -2.57;
            -3.44 -2.87 -2.57;
            -3.43 -2.86 -2.57
        ]
        cval_phi1 = [
            7.88 5.18 4.12;
            7.06 4.86 3.94;
            6.70 4.71 3.86;
            6.52 4.63 3.81;
            6.47 4.61 3.79;
            6.43 4.59 3.78
        ]
        C = vcat(reshape(cval_tau2[row, :], 1, 3), reshape(cval_phi1[row, :], 1, 3))
        return C, [:tau2, :phi1]
    else
        cval_tau3 = [
            -4.38 -3.60 -3.24;
            -4.15 -3.50 -3.18;
            -4.04 -3.45 -3.15;
            -3.99 -3.43 -3.13;
            -3.98 -3.42 -3.13;
            -3.96 -3.41 -3.12
        ]
        cval_phi2 = [
            8.21 5.68 4.67;
            7.02 5.13 4.31;
            6.50 4.88 4.16;
            6.22 4.75 4.07;
            6.15 4.71 4.05;
            6.09 4.68 4.03
        ]
        cval_phi3 = [
            10.61 7.24 5.91;
            9.31 6.73 5.61;
            8.73 6.49 5.47;
            8.43 6.49 5.47;
            8.34 6.30 5.36;
            8.27 6.25 5.34
        ]
        C = vcat(
            reshape(cval_tau3[row, :], 1, 3),
            reshape(cval_phi2[row, :], 1, 3),
            reshape(cval_phi3[row, :], 1, 3),
        )
        return C, [:tau3, :phi2, :phi3]
    end
end

function _validate_adf_regression_size(X::AbstractMatrix{<:Real})
    n, p = size(X)
    n > p || throw(ArgumentError("not enough observations for selected lag/model (n_obs=$n, n_params=$p)"))
end

function _run_adf_regression(yv::Vector{Float64}, model::Symbol, lag::Int)
    lag >= 0 || throw(ArgumentError("lags must be a nonnegative integer"))

    L = lag + 1
    z = _skipmissing_to_vec(diff(yv))
    n = length(z)
    n >= L || throw(ArgumentError("not enough observations for selected lag"))

    x = time_delay_embed(z, L)
    n_obs = size(x, 1)

    delta_y = Vector{Float64}(@view x[:, 1])
    y_lagged = Vector{Float64}(@view yv[L:n])
    trend = Float64.(collect(L:(L + n_obs - 1)))
    z_diff_lag = lag > 0 ? Matrix{Float64}(@view x[:, 2:L]) : Matrix{Float64}(undef, n_obs, 0)

    if model == :none
        X = hcat(y_lagged, z_diff_lag)
        _validate_adf_regression_size(X)
        fit = _ols(delta_y, X)

        tau = fit.β[1] / fit.se[1]
        return (
            tau = tau,
            phi1 = missing,
            phi2 = missing,
            phi3 = missing,
            beta = fit.β,
            se = fit.se,
            testreg = fit,
            res = fit.residuals,
            n_obs = n_obs,
            p = length(fit.β),
            testnames = [:tau1],
        )
    elseif model == :drift
        X = hcat(ones(Float64, n_obs), y_lagged, z_diff_lag)
        _validate_adf_regression_size(X)
        fit = _ols(delta_y, X)
        tau = fit.β[2] / fit.se[2]

        fitR = _ols(delta_y, z_diff_lag)
        RSSr = sum(fitR.residuals .^ 2)
        RSSf = sum(fit.residuals .^ 2)
        phi1 = _f_test_restricted_vs_full(RSSr, fitR.df_residual, RSSf, fit.df_residual)

        return (
            tau = tau,
            phi1 = phi1,
            phi2 = missing,
            phi3 = missing,
            beta = fit.β,
            se = fit.se,
            testreg = fit,
            res = fit.residuals,
            n_obs = n_obs,
            p = length(fit.β),
            testnames = [:tau2, :phi1],
        )
    else
        X = hcat(ones(Float64, n_obs), y_lagged, trend, z_diff_lag)
        _validate_adf_regression_size(X)
        fit = _ols(delta_y, X)
        tau = fit.β[2] / fit.se[2]

        fitR2 = _ols(delta_y, z_diff_lag)
        phi2 = _f_test_restricted_vs_full(
            sum(fitR2.residuals .^ 2),
            fitR2.df_residual,
            sum(fit.residuals .^ 2),
            fit.df_residual,
        )

        fitR3 = _ols(delta_y, hcat(ones(Float64, n_obs), z_diff_lag))
        phi3 = _f_test_restricted_vs_full(
            sum(fitR3.residuals .^ 2),
            fitR3.df_residual,
            sum(fit.residuals .^ 2),
            fit.df_residual,
        )

        return (
            tau = tau,
            phi1 = missing,
            phi2 = phi2,
            phi3 = phi3,
            beta = fit.β,
            se = fit.se,
            testreg = fit,
            res = fit.residuals,
            n_obs = n_obs,
            p = length(fit.β),
            testnames = [:tau3, :phi2, :phi3],
        )
    end
end

function _select_adf_lag(yv::Vector{Float64}, model::Symbol, k_max::Int, criterion::Symbol)
    k_max >= 0 || throw(ArgumentError("k_max must be a nonnegative integer"))
    criterion in (:aic, :bic) || throw(ArgumentError("criterion must be :aic or :bic for automatic lag selection"))

    best_k = nothing
    best_ic = Inf

    for k in 0:k_max
        out = try
            _run_adf_regression(yv, model, k)
        catch
            nothing
        end

        isnothing(out) && continue

        RSS = sum(out.res .^ 2)
        n_obs = length(out.res)
        kpen = criterion == :aic ? 2.0 : log(n_obs)
        ic = _information_criterion(RSS, n_obs, out.p, kpen)
        if ic < best_ic
            best_ic = ic
            best_k = k
        end
    end

    isnothing(best_k) && throw(ArgumentError("unable to fit ADF regression for any lag in 0:$k_max"))
    return Int(best_k)
end

function _adf_teststat(x::ADF)
    if x.model == :none
        return (tau1 = x.tau,)
    elseif x.model == :drift
        return (tau2 = x.tau, phi1 = Float64(coalesce(x.phi1, NaN)))
    else
        return (
            tau3 = x.tau,
            phi2 = Float64(coalesce(x.phi2, NaN)),
            phi3 = Float64(coalesce(x.phi3, NaN)),
        )
    end
end

function Base.propertynames(::ADF; private::Bool=false)
    base = (
        :model, :cval, :clevels, :lags, :tau, :phi1, :phi2, :phi3, :beta, :se,
        :testreg, :res, :testnames, :lag, :teststat,
    )
    return private ? base : base
end

function Base.getproperty(x::ADF, sym::Symbol)
    if sym === :lag
        return getfield(x, :lags)
    elseif sym === :teststat
        return _adf_teststat(x)
    end
    return getfield(x, sym)
end

"""
    adf(y; model=:none, lags=1, k_max=nothing, criterion=nothing, type=nothing, selectlags=nothing)

Augmented Dickey-Fuller unit root test.

The tested regression is:

- `model=:none`:
  `Delta y_t = beta1 * y_{t-1} + sum(gamma_i * Delta y_{t-i}) + eps_t`
- `model=:drift`:
  `Delta y_t = mu + beta1 * y_{t-1} + sum(gamma_i * Delta y_{t-i}) + eps_t`
- `model=:trend`:
  `Delta y_t = mu + beta1 * y_{t-1} + delta * t + sum(gamma_i * Delta y_{t-i}) + eps_t`

`tau` is the t-statistic on `beta1`. For `:drift` and `:trend`, phi statistics are
computed as restricted-vs-full F-tests using `_f_test_restricted_vs_full`.

Keyword compatibility:
- `model` and `type` are aliases
- `criterion` and `selectlags` are aliases
- if neither criterion alias is supplied:
  - `k_max` provided without `lags` => defaults to `:aic` lag selection over `0:k_max`
  - otherwise defaults to `:fixed`
"""
function adf(
    y;
    model::Union{Nothing,Symbol}=nothing,
    type::Union{Nothing,Symbol}=nothing,
    lags::Union{Nothing,Integer}=nothing,
    k_max::Union{Nothing,Integer}=nothing,
    criterion::Union{Nothing,Symbol}=nothing,
    selectlags::Union{Nothing,Symbol}=nothing,
)
    yv = _skipmissing_to_vec(y)
    length(yv) >= 3 || throw(ArgumentError("series too short for ADF test"))

    lag_kw = isnothing(lags) ? nothing : Int(lags)
    kmax_kw = isnothing(k_max) ? nothing : Int(k_max)

    if !isnothing(lag_kw) && lag_kw < 0
        throw(ArgumentError("lags must be a nonnegative integer"))
    end
    if !isnothing(kmax_kw) && kmax_kw < 0
        throw(ArgumentError("k_max must be a nonnegative integer"))
    end

    model_sym = _resolve_adf_model(model, type)
    crit = _resolve_adf_criterion(criterion, selectlags, lag_kw, kmax_kw)

    selected_lag = if crit == :fixed
        isnothing(lag_kw) ? (isnothing(kmax_kw) ? 1 : kmax_kw) : lag_kw
    else
        default_kmax = max(0, trunc(Int, (length(yv) - 1)^(1 / 3)))
        lag_cap = isnothing(kmax_kw) ? (isnothing(lag_kw) ? default_kmax : lag_kw) : kmax_kw
        _select_adf_lag(yv, model_sym, lag_cap, crit)
    end

    out = _run_adf_regression(yv, model_sym, selected_lag)
    cval, testnames = _adf_critical_values(model_sym, out.n_obs)
    clevels = [0.01, 0.05, 0.10]

    return ADF(
        model_sym,
        cval,
        clevels,
        selected_lag,
        out.tau,
        out.phi1,
        out.phi2,
        out.phi3,
        out.beta,
        out.se,
        out.testreg,
        out.res,
        testnames,
    )
end

function adf(; y, type::String="none", lags::Integer=1, selectlags::String="fixed")
    type_sym = _normalize_symbol(Symbol(type))
    selectlags_sym = _normalize_symbol(Symbol(selectlags))
    return adf(y; type=type_sym, lags=lags, selectlags=selectlags_sym)
end

"Approximate p-value for ADF tau statistic by interpolation over tau critical values."
function pvalue(u::ADF)
    return _pvalue_from_cvals(u.tau, vec(u.cval[1, :]), u.clevels)
end

function summary(x::ADF)
    parts = String[
        "model=$(repr(x.model))",
        "lags=$(x.lags)",
        "tau=$(round(x.tau, digits=4))",
    ]
    !ismissing(x.phi1) && push!(parts, "phi1=$(round(x.phi1, digits=4))")
    !ismissing(x.phi2) && push!(parts, "phi2=$(round(x.phi2, digits=4))")
    !ismissing(x.phi3) && push!(parts, "phi3=$(round(x.phi3, digits=4))")
    return "ADF(" * join(parts, ", ") * ")"
end

function _adf_model_desc(model::Symbol)
    if model == :none
        return "none (no constant)"
    elseif model == :drift
        return "drift (intercept)"
    elseif model == :trend
        return "trend (intercept + linear trend)"
    else
        return string(model)
    end
end

function show(io::IO, ::MIME"text/plain", x::ADF)
    if get(io, :compact, false)
        print(io, "ADF(", round(x.tau, digits=4), ")")
        return
    end

    println(io, "ADF Unit Root Test")
    println(io, "Model: ", _adf_model_desc(x.model))
    println(io, "Selected lags: ", x.lags)
    println(io, "Tau statistic: ", round(x.tau, digits=4))
    !ismissing(x.phi1) && println(io, "Phi1 statistic: ", round(x.phi1, digits=4))
    !ismissing(x.phi2) && println(io, "Phi2 statistic: ", round(x.phi2, digits=4))
    !ismissing(x.phi3) && println(io, "Phi3 statistic: ", round(x.phi3, digits=4))

    tau_p = try
        round(pvalue(x), digits=4)
    catch
        NaN
    end
    println(io, "Approx. p-value (tau): ", tau_p)

    if size(x.cval, 2) == length(x.clevels)
        println(io, "\nCritical values:")
        levels = ["$(Int(round(100 * a)))%" for a in x.clevels]
        for (i, name) in enumerate(x.testnames)
            rowvals = [round(v, digits=4) for v in x.cval[i, :]]
            println(io, "  ", name, ": ", join(["$(levels[j])=$(rowvals[j])" for j in eachindex(levels)], ", "))
        end
    end

    println(io, "\nRegression coefficients:")
    for i in eachindex(x.beta)
        println(io, "  beta[$i] = ", round(x.beta[i], digits=6), " (SE = ", round(x.se[i], digits=6), ")")
    end
end

function show(io::IO, x::ADF)
    if get(io, :compact, false)
        print(io, "ADF(", round(x.tau, digits=4), ")")
    else
        show(io, MIME("text/plain"), x)
    end
end
