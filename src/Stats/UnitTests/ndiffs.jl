
"""
    ndiffs(x; alpha=0.05, test=:kpss, deterministic=:level, max_d=2; kwargs...) -> Int

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

- KPSS: `deterministic = :level` → `:mu`, `deterministic = :trend` → `:tau`
- ADF:  `deterministic = :level` → `:drift`, `deterministic = :trend` → `:trend`
- PP:   `deterministic = :level` → `model = :constant`, `deterministic = :trend` → `model = :trend`

Critical values and their nominal significance levels are linearly interpolated
to obtain a p-value with boundary clamping (matching R's `approx(..., rule=2)` behavior).

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
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed), OTexts.
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
```
"""
function ndiffs(x::AbstractVector;
    alpha::Real=0.05,
    test::Symbol=:kpss,
    deterministic::Symbol=:level,
    max_d::Integer=2,
    kwargs...)::Int

    cleaned_series = [Float64(v) for v in x if !ismissing(v) && !(v isa AbstractFloat && isnan(v))]
    if isempty(cleaned_series)
        return 0
    end

    local clamped_alpha = clamp(alpha, 0.01, 0.10)
    if clamped_alpha != alpha
        @warn clamped_alpha < alpha ? "Specified alpha greater than max; setting alpha = 0.10" :
              "Specified alpha less than min; setting alpha = 0.01"
    end

    _is_constant_series(values) = length(unique(values)) ≤ 1
    differencing_order = 0
    if _is_constant_series(cleaned_series)
        return differencing_order
    end

    requires_differencing = _run_unit_root_test(cleaned_series; test=test, alpha=clamped_alpha,
        deterministic=deterministic,
        diff_order=differencing_order, kwargs...)
    if requires_differencing === missing
        return differencing_order
    elseif requires_differencing === false
        return differencing_order
    end

    while differencing_order < max_d
        differencing_order += 1
        cleaned_series = diff(cleaned_series)
        cleaned_series = dropmissing(cleaned_series)

        if isempty(cleaned_series) || length(cleaned_series) == 1 || _is_constant_series(cleaned_series)
            return differencing_order
        end

        requires_differencing = _run_unit_root_test(cleaned_series; test=test, alpha=clamped_alpha,
            deterministic=deterministic,
            diff_order=differencing_order, kwargs...)

        if requires_differencing === missing
            return differencing_order - 1
        elseif requires_differencing === false
            return differencing_order
        end
    end

    return differencing_order
end


function _run_unit_root_test(series_values::AbstractVector;
    test::Symbol=:kpss,
    alpha::Real=0.05,
    deterministic::Symbol=:level,
    diff_order::Int=0,
    kwargs...)::Union{Bool,Missing}

    kpss_type = deterministic === :trend ? :tau : :mu
    adf_type = deterministic === :trend ? :trend : :drift
    pp_model = deterministic === :trend ? :trend : :constant

    function _kpss_with_default_lag(values; kwargs...)
        sample_size = length(values)
        default_bandwidth = trunc(Int, 3 * sqrt(sample_size) / 13)
        if !haskey(kwargs, :use_lag)
            return kpss(values; type=kpss_type, use_lag=default_bandwidth)
        else
            return kpss(values; type=kpss_type, kwargs...)
        end
    end
    _normalize_pvalue(value) = (ismissing(value) || (value isa Real && isnan(value))) ? missing : value

    try
        if test === :kpss
            kpss_result = _kpss_with_default_lag(series_values; kwargs...)
            interpolated_pvalue = _normalize_pvalue(
                only(_linear_interpolate(kpss_result.cval, kpss_result.clevels, [kpss_result.teststat])),
            )
            return interpolated_pvalue === missing ? missing : interpolated_pvalue < alpha

        elseif test === :adf
            # Said & Dickey default when user did not pin lag search controls.
            adf_kwargs = kwargs
            if !haskey(adf_kwargs, :k_max) && !haskey(adf_kwargs, :lags) && !haskey(adf_kwargs, :criterion) && !haskey(adf_kwargs, :selectlags)
                adf_kwargs = (; kwargs..., k_max=trunc(Int, (length(series_values) - 1)^(1 / 3)))
            end
            adf_result = adf(series_values; model=adf_type, adf_kwargs...)
            tau_statistic = adf_result.teststat
            tau_critical_values = adf_result.primary_cval
            interpolated_pvalue = _normalize_pvalue(
                only(_linear_interpolate(tau_critical_values, adf_result.clevels, [tau_statistic])),
            )
            return interpolated_pvalue === missing ? missing : interpolated_pvalue > alpha

        elseif test === :pp
            pp_result = phillips_perron(series_values; type=:Z_tau, model=pp_model, kwargs...)
            interpolated_pvalue = _normalize_pvalue(
                only(_linear_interpolate(pp_result.primary_cval, pp_result.clevels, [pp_result.teststat])),
            )
            return interpolated_pvalue === missing ? missing : interpolated_pvalue > alpha

        else
            throw(ArgumentError("Unhandled test: $test"))
        end

    catch e
        @warn "Unit root test error at $(diff_order==0 ? "first" : string(diff_order+1)*"th") diff; using $diff_order differences. $(nameof(typeof(e))): $(e)"
        return false
    end
end
