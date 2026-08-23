"""
    auto_arima(y::AbstractVector, m;
               d::Union{Nothing,Int}=nothing, D::Union{Nothing,Int}=nothing,
               max_p::Int=5, max_q::Int=5, max_P::Int=2, max_Q::Int=2, max_order::Int=5,
               max_d::Int=2, max_D::Int=1,
               start_p::Int=2, start_q::Int=2, start_P::Int=1, start_Q::Int=1,
               stationary::Bool=false, seasonal::Bool=true,
               ic::Symbol=:aicc, stepwise::Bool=true, nmodels::Int=94, trace::Bool=false,
               approximation::Union{Nothing,Bool}=nothing, method=nothing,
               truncate::Union{Nothing,Int}=nothing, xreg::Union{Nothing,NamedMatrix}=nothing,
               test::Symbol=:kpss, test_args=NamedTuple(),
               seasonal_test::Symbol=:seas, seasonal_test_args=NamedTuple(),
               allowdrift::Bool=true, allowmean::Bool=true,
               lambda::Union{Nothing,Real}=nothing, biasadj::Bool=false;
               kwargs...) -> ArimaFit

Fit the "best" ARIMA/SARIMA model to a univariate time series by minimizing an information
criterion (`:aicc` default, `:aic`, or `:bic`). By default a fast stepwise search is used
(similar in spirit to Hyndman & Khandakar, 2008); seasonal differencing may be selected via a
measure of seasonal strength (Wang, Smith & Hyndman, 2006) unless overridden.

For a single series where runtime is less of a concern, consider `stepwise=false` and
`approximation=false` to search more exhaustively.

# Arguments
- `y`: Vector of observations (ordered).
- `m`: Seasonal period (e.g. 12 for monthly with annual seasonality). Set `seasonal=false`
  for nonseasonal models.

# Key options (selection)
- `d`, `D`: Nonseasonal/seasonal differences. If `nothing`, chosen via tests (`test`, `seasonal_test`).
- `max_p`, `max_q`, `max_P`, `max_Q`, `max_order`, `max_d`, `max_D`: Bounds for the search.
- `start_p`, `start_q`, `start_P`, `start_Q`: Initial orders for stepwise search.
- `stationary`, `seasonal`: Restrict to stationary or nonseasonal families if desired.
- `allowdrift`, `allowmean`: Permit drift (when `d>0`) and mean (when `d==0`) terms.
- `ic`: `:aicc|:aic|:bic`, the information criterion to minimize.
- `stepwise`, `nmodels`, `trace`: Control the search strategy and logging.
- `approximation`, `truncate`: Use a fast CSS-style approximation during search (then refit by ML).
- `xreg`: Optional exogenous regressors (ARIMAX). Must have `size(xreg,1) == length(y)`.
- `lambda`, `biasadj`: Optional Box–Cox transform and bias adjustment on back-transform.
- `test`, `test_args`, `seasonal_test`, `seasonal_test_args`: Unit-root/seasonality testing choices.
- `kwargs...`: Passed to the underlying ARIMA fitter (e.g. optimizer options).

# Returns
An [`ArimaFit`](@ref) containing the fitted model, estimates, diagnostics, and selected IC.

# Notes
- Choose `m` appropriately (e.g., 7 for daily-with-weekly, 12 for monthly, 4 for quarterly).
- With short series or small `m`, prefer `approximation=false` to avoid IC bias during search.
- With `xreg`, you are fitting an ARIMAX model; ensure regressors align with `y` and are not collinear.

# Examples
```julia
# Seasonal monthly series (m = 12)
fit = auto_arima(y, 12)

# Nonseasonal search
fit = auto_arima(y, 1; seasonal=false)

# More exhaustive search for a single series
fit = auto_arima(y, 12; stepwise=false, approximation=false)

# With exogenous regressors
X = NamedMatrix(hcat(x1, x2), [:x1, :x2])
fit = auto_arima(y, 12; xreg=X)

# Inspect results
fit.ic, fit.arma, fit.coef, fit.sigma2
```

**References**

* Hyndman, R.J. & Khandakar, Y. (2008). *Automatic time series forecasting: the forecast package for R*. JSS, **26**(3).
* Wang, X., Smith, K.A., & Hyndman, R.J. (2006). *Characteristic-based clustering for time series data*. DMKD, **13**(3), 335-364.

"""
function auto_arima(
    y::AbstractVector,
    m;
    d::Union{Nothing,Int} = nothing,
    D::Union{Nothing,Int} = nothing,
    max_p::Int = 5,
    max_q::Int = 5,
    max_P::Int = 2,
    max_Q::Int = 2,
    max_order::Int = 5,
    max_d::Int = 2,
    max_D::Int = 1,
    start_p::Int = 2,
    start_q::Int = 2,
    start_P::Int = 1,
    start_Q::Int = 1,
    stationary::Bool = false,
    seasonal::Bool = true,
    ic::Symbol = :aicc,
    stepwise::Bool = true,
    nmodels::Int = 94,
    trace::Bool = false,
    approximation::Union{Nothing,Bool} = nothing,
    method = nothing,
    truncate::Union{Nothing,Int} = nothing,
    xreg::Union{Nothing,NamedMatrix} = nothing,
    test::Symbol = :kpss,
    test_args = NamedTuple(),
    seasonal_test::Symbol = :seas,
    seasonal_test_args = NamedTuple(),
    allowdrift::Bool = true,
    allowmean::Bool = true,
    lambda::Union{Nothing,Real} = nothing,
    biasadj::Bool = false,
    kwargs...,
)
    # ── Validate symbolic arguments ──
    _check_arg(ic, (:aicc, :aic, :bic), "ic")
    _check_arg(test, (:kpss, :adf, :pp), "test")
    _check_arg(seasonal_test, (:seas, :ocsb), "seasonal_test")

    if isnothing(approximation)
        approximation = (length(y) > 150) || (m > 12)
    end

    # ── Trim missing values ──
    x = copy(y)
    firstnm, lastnm, serieslength, x = analyze_series(x)

    if isnothing(firstnm)
        throw(ArgumentError("All data are missing"))
    end

    if !isnothing(xreg)
        xreg = select_rows(xreg, firstnm:lastnm)
    end

    # ── Handle constant data ──
    if is_constant(x)
        if all(ismissing, x)
            throw(ArgumentError("All data are missing"))
        end
        if allowmean
            return arima_rjh(x, m, order = PDQ(0, 0, 0),
                             fixed = [mean2(x, skipmissing = true)], kwargs...)
        else
            return arima_rjh(x, m, order = PDQ(0, 0, 0), include_mean = false, kwargs...)
        end
    end

    # ── Seasonal period validation ──
    if !seasonal
        m = 1
    end
    if m < 1
        if seasonal
            throw(ArgumentError("m must be ≥ 1 for seasonal models. Got m=$m"))
        else
            m = 1
        end
    end

    # ── Cap orders by series length ──
    max_p = min(max_p, fld(serieslength, 3))
    max_q = min(max_q, fld(serieslength, 3))
    max_P = min(max_P, fld(fld(serieslength, 3), m))
    max_Q = min(max_Q, fld(fld(serieslength, 3), m))

    if serieslength ≤ 3
        ic = :aic
    end

    # ── Preprocess: Box-Cox + OLS residuals for tests ──
    prep = preprocess_series(x, xreg, m; lambda = lambda)
    x_work = prep.x_work
    xreg_work = prep.xreg_work
    lambda = prep.lambda

    # ── Select differencing orders ──
    diff_result = select_differencing(prep.x_for_tests, xreg_work, m;
        d = d, D = D, max_d = max_d, max_D = max_D,
        stationary = stationary,
        test = test, test_args = test_args,
        seasonal_test = seasonal_test, seasonal_test_args = seasonal_test_args)
    d, D = diff_result.d, diff_result.D

    if m == 1
        max_P = 0
        max_Q = 0
    end

    # ── Check if differenced series is trivial ──
    dx = D > 0 ? diff(prep.x_for_tests; difference_order = D, lag_steps = m) : copy(prep.x_for_tests)
    if d > 0
        dx = diff(dx; difference_order = d, lag_steps = 1)
    end

    if length(dx) == 0
        throw(ArgumentError("Not enough data to proceed"))
    elseif is_constant(dx)
        fit = fit_constant_series(x_work, m, d, D, dx, xreg, method, kwargs)
        fit.y = y
        return fit
    end

    # ── Seasonal constraints on p/q ──
    if m > 1
        max_P > 0 && (max_p = min(max_p, m - 1))
        max_Q > 0 && (max_q = min(max_q, m - 1))
    end

    # ── Approximation offset ──
    offset = if approximation
        compute_approx_offset(x_work, d, D, m, xreg, truncate, kwargs)
    else
        0.0
    end

    allowdrift = allowdrift && (d + D) == 1
    allowmean = allowmean && (d + D) == 0
    constant = allowdrift || allowmean

    if trace && approximation
        println("\nFitting models using approximations...\n")
    end

    # ── Build config structs (like constructing SARIMA model in core) ──
    bounds = SearchBounds(max_p, max_q, max_P, max_Q, max_order)
    config = SearchConfig(
        ic, trace, approximation, offset,
        xreg, method,
        allowdrift, allowmean,
        nmodels, kwargs,
    )

    # ── Search ──
    if !stepwise
        bestfit = grid_search(x_work, m, d, D, bounds, config)
    else
        if length(x_work) < 10
            start_p = min(start_p, 1)
            start_q = min(start_q, 1)
            start_P = 0
            start_Q = 0
        end
        start_p = min(start_p, max_p)
        start_q = min(start_q, max_q)
        start_P = min(start_P, max_P)
        start_Q = min(start_Q, max_Q)

        search_out = stepwise_search(x_work, m, d, D,
            start_p, start_q, start_P, start_Q, constant,
            bounds, config)
        bestfit = search_out.bestfit

        if approximation
            bestfit = refit_stepwise_best(bestfit, search_out.results,
                x_work, m, d, D, serieslength, config)
        end
    end

    if trace
        println("Best model found!")
    end

    bestfit.lambda = lambda
    bestfit.y = y
    bestfit.fitted = fitted(bestfit)

    return bestfit
end
