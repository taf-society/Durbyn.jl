ordinal(n::Integer) =
    n == 1 ? "first" : n == 2 ? "second" : n == 3 ? "third" : string(n, "th")

function approx_pval(
    cvals::AbstractVector{<:Real},
    clevels::AbstractVector{<:Real},
    stat::Real,
)

    p = sortperm(cvals)
    x = collect(cvals)[p]
    y = collect(clevels)[p]
    n = length(x)
    n == 0 && return missing
    isnan(stat) && return missing
    if n == 1
        return y[1]
    end
    if stat ≤ x[1]
        return y[1] + (stat - x[1]) * (y[2] - y[1]) / (x[2] - x[1])
    elseif stat ≥ x[end]
        return y[end-1] + (stat - x[end-1]) * (y[end] - y[end-1]) / (x[end] - x[end-1])
    else
        i = searchsortedfirst(x, stat)
        i = max(i, 2)
        x0, x1 = x[i-1], x[i]
        y0, y1 = y[i-1], y[i]
        return y0 + (stat - x0) * (y1 - y0) / (x1 - x0)
    end
end

_firststat(ts) = isa(ts, Number) ? float(ts) : float(Array(ts)[1])

"""
    ndiffs(x; alpha=0.05, test=:kpss, type=:level, max_d=2; kwargs...) -> Int

Number of differences required for a stationary series.

`ndiffs` estimates the minimum number of **non-seasonal first differences** needed
to render a univariate time series `x` (approximately) stationary. The decision is
based on a unit-root test selected via `test`.

### Details

- If `test == :kpss` (Kwiatkowski-Phillips-Schmidt-Shin), the null hypothesis is
  **stationarity** of `x` against a unit-root alternative. `ndiffs` returns the
  smallest number of differences such that the KPSS test **does not reject**
  stationarity at level `alpha`.

- If `test == :adf` (Augmented Dickey-Fuller) or `test == :pp` (Phillips-Perron),
  the null hypothesis is a **unit root** in `x` against a stationary alternative.
  `ndiffs` returns the smallest number of differences such that the test
  **rejects** the unit root at level `alpha`.

`type` controls the deterministic component included in the test regression:

- `:level` → intercept only
- `:trend` → intercept + linear trend

Internally, the following mappings are used when calling the tests:

- KPSS: `type = :level` → `"mu"`, `type = :trend` → `"tau"`
- ADF:  `type = :level` → `:drift`, `type = :trend` → `:trend`
- PP:   `type = :level` → `model = "constant"`, `type = :trend` → `model = "trend"`

Critical values and their nominal significance levels are linearly interpolated
to obtain a p-value (matching R’s `approx(..., rule=2)` behavior).

If no `use_lag` is supplied to KPSS, the lag is set by default to
`trunc(3 * sqrt(length(x)) / 13)`.

`alpha` is clamped to the range `[0.01, 0.10]` with a warning if it falls outside.

### Arguments
- `x::AbstractVector`: Univariate time series (missing values are skipped; non-finite values are dropped).

### Keyword Arguments
- `alpha::Float64 = 0.05`: Test level (clamped to `[0.01, 0.10]`).
- `test::Symbol = :kpss`: Unit-root test to use. One of `:kpss`, `:adf`, `:pp`.
- `type::Symbol = :level`: Deterministic component. One of `:level`, `:trend`.
- `max_d::Int = 2`: Maximum number of differences to attempt.
- `kwargs...`: Passed through to the underlying test:
  - KPSS: e.g. `use_lag::Union{Nothing,Int}`, `lags::String|Symbol`
  - ADF: e.g. `lags::Int`, `selectlags::Symbol|String`
  - PP:  e.g. `lags::String`, `use_lag::Union{Nothing,Int}`

### Returns
- `Int`: The estimated number of first differences required for stationarity,
  up to `max_d`.

### Notes
- If the series becomes constant after differencing, `ndiffs` returns the current `d`.
- If a test call errors at some differencing order, a warning is emitted and the
  current `d` is returned (mirroring the R behavior).
- P-values are derived by interpolating the test statistic against the reported
  critical values at `clevels` (e.g., 0.01, 0.05, 0.10).

### See also
`kpss`, `adf`, `phillips_perron`

### References
- Dickey, D.A., and Fuller, W.A. (1979). *Distribution of the Estimators for Autoregressive
  Time Series with a Unit Root*. **JASA**, 74, 427-431.
- Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., and Shin, Y. (1992). *Testing the Null
  Hypothesis of Stationarity against the Alternative of a Unit Root*. **Journal of Econometrics**, 54, 159-178.
- Phillips, P.C.B., and Perron, P. (1988). *Testing for a Unit Root in Time Series Regression*. **Biometrika**, 72(2), 335-346.
- Said, S.E., and Dickey, D.A. (1984). *Testing for Unit Roots in ARMA Models of Unknown Order*. **Biometrika**, 71, 599-607.
- Osborn, D.R. (1990). *A survey of seasonality in UK macroeconomic variables*. **International Journal of Forecasting**, 6, 327-336.

### Examples
```julia
d1 = ndiffs(WWWusage)                         # KPSS by default
d2 = ndiffs(diff(log.(AirPassengers), 12);    # supply transformed series
           test = :adf, type = :trend, alpha = 0.05)

# With explicit KPSS lag selection:
d3 = ndiffs(x; test = :kpss, type = :level, use_lag = 8)

"""
function ndiffs(
    x::AbstractVector;
    alpha::Float64 = 0.05,
    test::Symbol = :kpss,
    type::Symbol = :level,
    max_d::Int = 2,
    kwargs...,
)::Int

    # normalize args
    test = Symbol(lowercase(String(test)))
    type = Symbol(lowercase(String(type)))
    test ∈ (:kpss, :adf, :pp) || throw(ArgumentError("test must be :kpss, :adf, or :pp"))
    type ∈ (:level, :trend) || throw(ArgumentError("type must be :level or :trend"))

    # alpha clamp (R behavior)
    if alpha < 0.01
        @warn "Specified alpha value is less than the minimum, setting alpha = 0.01"
        alpha = 0.01
    elseif alpha > 0.10
        @warn "Specified alpha value is larger than the maximum, setting alpha = 0.10"
        alpha = 0.10
    end

    # clean x
    xv = collect(skipmissing(x))
    xv = filter(isfinite, xv)
    d = 0

    if is_constant(xv)
        return d
    end

    # inner runner (captures d for warnings, like the R version)
    function run_tests(xvec)::Union{Bool,Missing}
        try
            if test === :kpss
                ktype = (type === :level ? "mu" : "tau")
                # use_lag default if not supplied: trunc(3 * sqrt(n) / 13)
                n = length(xvec)
                default_use = trunc(Int, 3 * sqrt(n) / 13)
                new_kwargs =
                    (:use_lag in keys(kwargs)) ? kwargs :
                    (; kwargs..., use_lag = default_use)
                t = kpss(; y = xvec, type = ktype, new_kwargs...)
                p = approx_pval(t.cval, t.clevels, t.teststat)
                return p === missing ? missing : (p < alpha)   # reject stationarity → need differencing
            elseif test === :adf
                atype = (type === :level ? :drift : :trend)
                t = adf(; y = xvec, type = atype, kwargs...)       # ADF null: unit root
                # use first row of cval and the first test statistic, matching R's urca_pval
                cvals = vec(t.cval[1, :])
                p = approx_pval(cvals, t.clevels, _firststat(t.teststat))
                return p === missing ? missing : (p > alpha)   # fail to reject unit root → need differencing
            else # :pp
                # PP null: unit root. R used type="Z-tau" and model constant/trend.
                model = (type === :level ? "constant" : "trend")
                # Prefer the string API to avoid Symbol name mismatches.
                t = phillips_perron(; x = xvec, type = "Z_tau", model = model, kwargs...)
                p = approx_pval(t.cval, t.clevels, t.teststat)
                return p === missing ? missing : (p > alpha)   # fail to reject unit root → need differencing
            end
        catch e
            # mirror R's warning style (informative but continue)
            @warn "The chosen unit root test encountered an error when testing for the $(ordinal(d+1)) difference.\n$(typeof(e)):\n$(sprint(showerror, e))\n$(d) differences will be used. Consider using a different unit root test."
            return false
        end
    end

    # first pass
    dodiff = run_tests(xv)
    if dodiff === missing
        return d
    end

    # iterate differences
    while dodiff === true && d < max_d
        d += 1
        xv = diff(xv)
        if is_constant(xv)
            return d
        end
        dodiff = run_tests(xv)
        if dodiff === missing
            return d - 1
        end
    end

    return d
end
