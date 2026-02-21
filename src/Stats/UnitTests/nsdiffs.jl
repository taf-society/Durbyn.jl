"""
    nsdiffs(x, m; alpha=0.05, test::Symbol=:seas, maxD::Int=1, kwargs...) -> Int
    nsdiffs(x, m; alpha=0.05, test::String=:seas, maxD::Int=1, kwargs...) -> Int
# Number of seasonal differences
Estimate the number of seasonal differences required to make a time series seasonally stationary.

`nsdiffs` uses seasonal unit-root / seasonality tests to determine how many seasonal
differences are needed for `x` to be made stationary (possibly in combination with
non-seasonal differencing).

## Tests
Several tests are available:

- `test = :seas` (default): a seasonal-strength heuristic is used; differencing is
  selected if `seasonal_strength(x; periods=[m], ...) > 0.64`
  (threshold chosen to minimize MASE when forecasting auto-ARIMA on M3/M4 data;
  see Wang, Smith & Hyndman, 2006).

- `test = :ocsb`: the Osborn-Chui-Smith-Birchenhall (1988) test is used
  (null hypothesis: a seasonal unit root exists). The decision rule compares the
  test statistic to the 5% critical value. Only 5% significance is supported.

> Not yet supported in this Julia port (will throw `ArgumentError` if requested):
> `test = :hegy` (Hylleberg-Engle-Granger-Yoo, 1990) and `test = :ch`
> (Canova-Hansen, 1995).

## Arguments
- `x::AbstractVector{<:Real}`: Univariate time series.
- `m::Int`: Length of the seasonal period (required).
- `alpha::Real=0.05`: Test level; clamped to `[0.01, 0.10]`.
  For `test = :ocsb`, `alpha` is forced to `0.05`.
- `test::Symbol = :seas`: Which seasonal test to use (`:seas` or `:ocsb`).
- `maxD::Int = 1`: Maximum number of seasonal differences allowed.
- `kwargs...`: Passed through to the underlying test functions
  (e.g., `lag_method`, `maxlag` for `:ocsb`; options for `seasonal_strength`).

## Behavior & edge cases
- If `m == 1`, throws `ArgumentError("Non seasonal data")`.
- If `m < 1`, or `m ≥ length(x)`, returns `0`.
- If `x` is constant, returns `0`.
- After each seasonal difference, the series is re-tested; iteration stops early if
  the test no longer suggests differencing or when `maxD` is reached.

## Returns
An `Int` giving the number of seasonal differences `D ∈ 0:maxD`.

## Examples
```julia
D = nsdiffs(x, 12)                # seasonal-strength heuristic
D = nsdiffs(x, 7, test=:ocsb)     # OCSB at 5% by design

```
References:

Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering
for time series data. Data Mining and Knowledge Discovery, 13(3), 335-364.

Osborn, D. R., Chui, A. P. L., Smith, J., & Birchenhall, C. R. (1988).
Seasonality and the order of integration for consumption. Oxford Bulletin of
Economics and Statistics, 50(4), 361-377.

Canova, F., & Hansen, B. E. (1995). Are Seasonal Patterns Constant over Time?
A Test for Seasonal Stability. Journal of Business & Economic Statistics,
13(3), 237-252. (Not currently implemented here.)

Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990).
Seasonal integration and cointegration. Journal of Econometrics, 44(1), 215-238.
(Not currently implemented here.)
"""
function nsdiffs(x::AbstractVector,
                 m::Int;
                 alpha::Real = 0.05,
                 test::Symbol = :seas,
                 maxD::Int = 1,
                 kwargs...)

    test in (:seas, :ocsb) || throw(ArgumentError("Tests :hegy and :ch are not supported in this Julia port yet."))

    α, notes = normalize_alpha(alpha, test)
    foreach(msg -> @warn(msg), notes)
    outcome = precheck(x, m)
    !isnothing(outcome) && return outcome::Int
    return compute_D(x, m, Val(test), α, maxD; kwargs...)
end

function nsdiffs(;x::AbstractVector,
    m::Int,
    alpha::Real=0.05,
    test::String="seas",
    maxD::Int=1,
    kwargs...)

    test = match_arg(test, ["ocsb", "seas"])
    test = Symbol(test)
    return nsdiffs(x, m, alpha=alpha, test=test, maxD = maxD, kwargs...)
end

function normalize_alpha(alpha::Real, test::Symbol)
    notes = String[]
    α = alpha
    if α < 0.01
        push!(notes, "Specified alpha value is less than the minimum, setting alpha = 0.01")
        α = 0.01
    elseif α > 0.10
        push!(notes, "Specified alpha value is larger than the maximum, setting alpha = 0.10")
        α = 0.10
    end
    if test === :ocsb && α != 0.05
        push!(notes, "Significance levels other than 5% are not currently supported by test=:ocsb, defaulting to alpha = 0.05.")
        α = 0.05
    end
    return α, notes
end


function precheck(x::AbstractVector{<:Real}, m::Int)
    is_constant(x) && return 0
    if m == 1
        throw(ArgumentError("Non seasonal data"))
    elseif m < 1
        @warn "Cannot handle data with frequency less than 1. Seasonality will be ignored."
        return 0
    elseif m ≥ length(x)
        return 0
    end
    return nothing
end


function compute_D(x::AbstractVector{<:Real}, m::Int, testv::Val, α::Real, maxD::Int; kwargs...)
    dodiff = run_test(x, m, testv, α; kwargs...)
    return compute_D_inner(x, m, testv, α, maxD, 0, dodiff; kwargs...)
end

function compute_D_inner(x::AbstractVector{<:Real}, m::Int, testv::Val,
                         α::Real, maxD::Int, D::Int, dodiff::Bool; kwargs...)
    if !dodiff || D ≥ maxD
        return D
    end

    y = diff(x, lag = m)
    is_constant(y) && return D + 1

    if length(y) ≥ 2m && D + 1 < maxD
        return compute_D_inner(y, m, testv, α, maxD, D + 1,
                               run_test(y, m, testv, α; kwargs...); kwargs...)
    else
        return D + 1
    end
end

function run_test(x::AbstractVector{<:Real}, m::Int, ::Val{:seas}, α::Real; kwargs...)
    try
        s = seasonal_strength(x, m, kwargs...)
        return !isempty(s) && s[1] > 0.64
    catch e
        rankword = "first"
        @warn "Seasonality heuristic failed while testing the $rankword difference. " *
              "From $(nameof(typeof(e))): $(sprint(showerror, e)). " *
              "Proceeding as if no seasonal difference is needed."
        return false
    end
end


function run_test(x::AbstractVector{<:Real}, m::Int, ::Val{:ocsb}, α::Real; kwargs...)
    try
        
        oc = ocsb(dropmissing(x), m, lag_method=:AIC, maxlag=3, kwargs...)
        return oc.teststat > oc.cval
    catch e
        @warn "OCSB test failed. From $(nameof(typeof(e))): $(sprint(showerror, e)). " *
              "Proceeding as if no seasonal difference is needed."
        return false
    end
end
