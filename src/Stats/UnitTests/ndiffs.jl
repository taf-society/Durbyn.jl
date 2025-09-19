
"""
    ndiffs(x; alpha=0.05, test=:kpss, type=:level, max_d=2; kwargs...) -> Int
    ndiffs(;x, alpha=0.05, test="kpss", type="level", max_d=2; kwargs...) -> Int

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
function ndiffs(x::AbstractVector;
    alpha::Real=0.05,
    test::Symbol=:kpss,
    deterministic::Symbol=:level,
    maxd::Integer=2,
    kwargs...)::Int

    xclean = collect(skipmissing(x))
    if isempty(xclean)
        return 0
    end

    local α = clamp(alpha, 0.01, 0.10)
    if α != alpha
        @warn α < alpha ? "Specified alpha greater than max; setting alpha = 0.10" :
              "Specified alpha less than min; setting alpha = 0.01"
    end

    is_constant(v) = length(unique(v)) ≤ 1
    d = 0
    if is_constant(xclean)
        return d
    end

    need_diff = run_unit_root_check(xclean; test=test, alpha=α,
        deterministic=deterministic,
        d_for_msg=d, kwargs...)
    if need_diff === missing
        return d
    elseif need_diff === false
        return d
    end

    while d < maxd
        d += 1
        xclean = diff(xclean)
        xclean = na_omit(xclean)

        if isempty(xclean) || length(xclean) == 1 || is_constant(xclean)
            return d
        end

        need_diff = run_unit_root_check(xclean; test=test, alpha=α,
            deterministic=deterministic,
            d_for_msg=d, kwargs...)

        if need_diff === missing
            return d - 1
        elseif need_diff === false
            return d
        end
    end

    return d
end

function ndiffs(;
    x::AbstractVector,
    alpha::Float64=0.05,
    test::String="kpss",
    type::String="level",
    max_d::Int=2,
    kwargs...,)::Int

    test = match_arg(test, ["kpss", "adf", "pp"])
    type = match_arg(type, ["level", "trend"])

    test = Symbol(test)
    type = Symbol(type)

    ndiffs(x, alpha=alpha, test=test, type=type, max_d=max_d, kwargs...)
end

function run_unit_root_check(xvec::AbstractVector;
    test::Symbol=:kpss,
    alpha::Real=0.05,
    deterministic::Symbol=:level,
    d_for_msg::Int=0,
    kwargs...)::Union{Bool,Missing}

    kpss_type = deterministic === :trend ? "tau" : "mu"
    adf_type = deterministic === :trend ? :trend : :drift
    pp_model = deterministic === :trend ? "trend" : "constant"

    function with_kpss_lag(y; kwargs...)
        nt = length(y)
        default_lag = trunc(Int, 3 * sqrt(nt) / 13)
        if !haskey(kwargs, :use_lag)
            return kpss(; y, type=kpss_type, use_lag=default_lag)
        else
            return kpss(; y, type=kpss_type, kwargs...)
        end
    end
    norm_p(p) = (ismissing(p) || (p isa Real && isnan(p))) ? missing : p

    try
        if test === :kpss
            t = with_kpss_lag(xvec; kwargs...)
            p = norm_p(only(approx(t.cval, t.clevels, xout=[t.teststat[1]], rule=2).y))
            return p === missing ? missing : p < alpha

        elseif test === :adf
            t = adf(; y=xvec, type=adf_type, kwargs...)
            p = norm_p(only(approx(t.cval, t.clevels, xout=[t.teststat[1]], rule=2).y))
            return p === missing ? missing : p > alpha

        elseif test === :pp
            t = phillips_perron(; x=xvec, type="Z_tau", model=pp_model, kwargs...)
            p = norm_p(only(approx(t.cval, t.clevels, xout=[t.teststat[1]], rule=2).y))
            return p === missing ? missing : p > alpha

        else
            throw(ArgumentError("Unhandled test: $test"))
        end

    catch e
        @warn "Unit root test error at $(d_for_msg==0 ? "first" : string(d_for_msg+1)*"th") diff; using $d_for_msg differences. $(nameof(typeof(e))): $(e)"
        return false
    end
end
