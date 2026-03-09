"""
    KPSS

Result object for the KPSS (Kwiatkowski-Phillips-Schmidt-Shin) stationarity test.

# Fields
- `type::Symbol`: Deterministic component used in the test.
    - `:mu`   → constant only.
    - `:tau`  → constant and linear trend.
- `lag::Int`: The bandwidth / number of lags used for the long-run variance correction.
- `teststat::Float64`: KPSS test statistic.
- `cval::Vector{Float64}`: Critical values corresponding to `clevels`.
- `clevels::Vector{Float64}`: Nominal significance levels (e.g., `[0.10, 0.05, 0.01]`).
- `coef::Vector{Float64}`: Regression coefficients (non-empty for `:tau`).
- `se::Vector{Float64}`: Standard errors of `coef`.
- `testreg::NamedTuple`: OLS fit details for the deterministic regression (empty for `:mu`).
- `res::Vector{Float64}`: Residuals from the deterministic regression used by the test.
"""
struct KPSS
    type::Symbol
    lag::Int
    teststat::Float64
    cval::Vector{Float64}
    clevels::Vector{Float64}
    coef::Vector{Float64}
    se::Vector{Float64}
    testreg::NamedTuple
    res::Vector{Float64}
end

function _kpss(y, type::Symbol=:mu, lags::Symbol=:short, use_lag::Union{Nothing,Int}=nothing)
    yv = _skipmissing_to_vec(y)
    n = length(yv)
    type in (:mu, :tau) || throw(ArgumentError("type must be :mu or :tau"))
    lags in (:short, :long, :nil) || throw(ArgumentError("lags must be :short, :long, or :nil"))

    max_lag =
        if !isnothing(use_lag)
            L = Int(use_lag)
            if L < 0
                @warn "use_lag must be a nonnegative integer; using lags=:short default."
                L = trunc(Int, 4 * (n / 100)^0.25)
            end
            L
        elseif lags == :short
            trunc(Int, 4 * (n / 100)^0.25)
        elseif lags == :long
            trunc(Int, 12 * (n / 100)^0.25)
        else
            0
        end

    regression_coef = Float64[]
    regression_se = Float64[]
    regression_fit = NamedTuple()
    if type == :mu
        cval = [0.347, 0.463, 0.574, 0.739]
        clevels = [0.10, 0.05, 0.025, 0.01]
        res = yv .- mean(yv)
    else
        cval = [0.119, 0.146, 0.176, 0.216]
        clevels = [0.10, 0.05, 0.025, 0.01]
        trend = collect(1:n)
        X = hcat(ones(Float64, n), Float64.(trend))
        fit = _ols(yv, X)
        regression_coef = fit.β
        regression_se = fit.se
        regression_fit = fit
        res = fit.residuals
    end

    cumulative_sum = cumsum(res)
    numerator = sum(cumulative_sum .^ 2) / n^2
    residual_variance = sum(res .^ 2) / n
    if max_lag == 0
        denominator = residual_variance
    else
        add = _bartlett_lrv(res, n, max_lag)
        denominator = residual_variance + add
    end

    teststat = numerator / denominator
    return KPSS(type, max_lag, teststat, cval, clevels, regression_coef, regression_se, regression_fit, res)
end

"""
    kpss(y; type::Symbol = :mu, lags::Symbol = :short, use_lag::Union{Nothing,Int} = nothing)

Perform the KPSS unit root test, where the **null hypothesis is stationarity**.

The deterministic component is controlled by `type`:
- `:mu`  — constant only,
- `:tau` — constant and linear trend.

The long-run variance correction uses a lag/bandwidth choice controlled by `lags`:
- `:short` — sets the maximum lag to ``⌊(4(n/100))^{1/4}⌋``,
- `:long`  — sets the maximum lag to ``⌊(12(n/100))^{1/4}⌋``,
- `:nil`   — no error correction (lag = 0).

If `use_lag` is provided, it **overrides** the choice implied by `lags`.

# Arguments
- `y::AbstractVector{<:Real}`: Series to test (will be converted to `Float64`).

# Keyword Arguments
- `type::Symbol = :mu`: Deterministic part (`:mu` or `:tau`).
- `lags::Symbol = :short`: Lag rule (`:short`, `:long`, or `:nil`).
- `use_lag::Union{Nothing,Int} = nothing`: User-specified number of lags (≥ 0).

# Returns
A `KPSS` struct containing the test statistic, critical values, and related
quantities.

!!! note
    The KPSS test checks stationarity as the **null**. Small p-values (large
    test statistics relative to critical values) suggest **rejecting** stationarity.

# References
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992).
  *Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root.*
  **Journal of Econometrics**, 54, 159–178.
- Schwert, G. W. (1989). *Tests for unit roots: A Monte Carlo investigation.*
  **Journal of Business & Economic Statistics**, 7(2), 147–159.

# Examples
```julia
using StatsBase # (only if you need helpers elsewhere)

y = randn(200) .+ 0.5 # stationary with mean shift
out = kpss(y; type = :mu, lags = :short)
# @show out.teststat, out.cval, out.clevels
```

```julia
# Linear trend case and user-specified lag:
y = cumsum(randn(300)) .+ 0.01 .* (1:300)  # trending, typically non-stationary
out = kpss(y; type = :tau, use_lag = 5)
```

"""
function kpss(y; type::Symbol=:mu, lags::Symbol=:short, use_lag::Union{Nothing,Int}=nothing)
    _kpss(y, type, lags, use_lag)
end

function Base.propertynames(::KPSS; private::Bool=false)
    base = (
        :type, :lag, :teststat, :cval, :clevels, :coef, :se, :testreg, :res,
        :deterministic, :lags, :lag_order, :stats, :primary_cval, :critical_values,
        :coefficients, :standard_errors, :regression, :residuals,
    )
    return private ? base : base
end

function Base.getproperty(x::KPSS, sym::Symbol)
    if sym === :deterministic
        return getfield(x, :type)
    elseif sym === :lags
        return getfield(x, :lag)
    elseif sym === :lag_order
        return getfield(x, :lag)
    elseif sym === :stats
        return (eta = getfield(x, :teststat),)
    elseif sym === :primary_cval
        return getfield(x, :cval)
    elseif sym === :critical_values
        return (eta = getfield(x, :cval),)
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


"Approximate p-value for KPSS by linear interpolation between critical values."
pvalue(u::KPSS) = _pvalue_from_cvals(u.teststat, u.cval, u.clevels)

summary(x::KPSS) = "KPSS(teststat=$(round(x.teststat, digits=4)), type=$(x.type), lag=$(x.lag))"

function show(io::IO, ::MIME"text/plain", x::KPSS)
    println(io, "KPSS Unit Root / Cointegration Test")
    println(io, "Deterministic component (type): ",
    x.type == :tau ? "trend (tau)" : "level (mu)")
    println(io, "Lag truncation (bandwidth): ", x.lag)

    println(io, "\nThe value of the test statistic is: ", round(x.teststat, digits=4))

    if !isempty(x.cval) && !isempty(x.clevels) && length(x.cval) == length(x.clevels)
        println(io, "\nCritical values:")
        for (α, cv) in zip(x.clevels, x.cval)
            level = isa(α, Number) ? Int(round(100α)) : α
            println(io, "  ", lpad("$(level)%", 4), " : ", round(cv, digits=4))
        end
    end
end

function show(io::IO, x::KPSS)
    # honor compact printing
    if get(io, :compact, false)
        print(io, "KPSS($(round(x.teststat, digits=4)))")
    else
        show(io, MIME("text/plain"), x)
    end
end
