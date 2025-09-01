"""
    auto_arima(y::AbstractVector, m;
               d::Union{Nothing,Int}=nothing, D::Union{Nothing,Int}=nothing,
               max_p::Int=5, max_q::Int=5, max_P::Int=2, max_Q::Int=2, max_order::Int=5,
               max_d::Int=2, max_D::Int=1,
               start_p::Int=2, start_q::Int=2, start_P::Int=1, start_Q::Int=1,
               stationary::Bool=false, seasonal::Bool=true,
               ic::String="aicc", stepwise::Bool=true, nmodels::Int=94, trace::Bool=false,
               approximation::Union{Nothing,Bool}=nothing, method=nothing,
               truncate::Union{Nothing,Int}=nothing, xreg::Union{Nothing,NamedMatrix}=nothing,
               test::String="kpss", test_args=NamedTuple(),
               seasonal_test::String="seas", seasonal_test_args=NamedTuple(),
               allowdrift::Bool=true, allowmean::Bool=true,
               lambda::Union{Nothing,Real}=nothing, biasadj::Bool=false;
               kwargs...) -> ArimaFit

Fit the “best” ARIMA/SARIMA model to a univariate time series by minimizing an information
criterion (`"aicc"` default, `"aic"`, or `"bic"`). By default a fast stepwise search is used
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
- `ic`: `"aicc"|"aic"|"bic"`, the information criterion to minimize.
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
````

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
    ic::String = "aicc",
    stepwise::Bool = true,
    nmodels::Int = 94,
    trace::Bool = false,
    approximation::Union{Nothing,Bool} = nothing,
    method = nothing,
    truncate::Union{Nothing,Int} = nothing,
    xreg::Union{Nothing,NamedMatrix} = nothing,
    test::String = "kpss",
    test_args = NamedTuple(),
    seasonal_test::String = "seas",
    seasonal_test_args = NamedTuple(),
    allowdrift::Bool = true,
    allowmean::Bool = true,
    lambda::Union{Nothing,Real} = nothing,
    biasadj::Bool = false,
    kwargs...,
)

    ic = match_arg(ic, ["aicc", "aic", "bic"])
    test = match_arg(test, ["kpss", "adf", "pp"])
    seasonal_test = match_arg(seasonal_test, ["seas", "ocsb", "hegy", "ch"])

    if isnothing(approximation)
        approximation = (length(y) > 150 | m > 12)
    end

    # Trim leading/trailing missings and count non-missing in the trimmed span
    x = copy(y)
    firstnm, serieslength, x = analyze_series(x)

    if xreg !== nothing
        indx = firstnm:size(xreg.data, 1)
        xreg = get_elements(xreg, row = indx)
    end
    # Check constant data
    if is_constant(x)
        if all(ismissing, x)
            error("All data are missing")
        end

        if allowmean
            fit = arima_rjh(
                x,
                m,
                order = PDQ(0, 0, 0),
                fixde = mean2(x, omit_na = true),
                kwargs...,
            )
        else
            fit = arima_rjh(x, m, order = PDQ(0, 0, 0), include_mean = false, kwargs...)
        end
        fit.constant = true
        return fit
    end

    if !seasonal
        m = 1
    end
    if m < 1
        @warn "m < 1 not supported; ignoring seasonality."
    end

    # Cap per series length
    max_p = min(max_p, fld(serieslength, 3))
    max_q = min(max_q, fld(serieslength, 3))
    max_P = min(max_P, fld(fld(serieslength, 3), m))
    max_Q = min(max_Q, fld(fld(serieslength, 3), m))

    # Use AIC for tiny sample
    if serieslength ≤ 3
        ic = "aic"
    end

    # Box-Cox transform
    if lambda !== nothing
        x, lambda = box_cox(x, m, lambda = lambda)
    end

    xx = copy(x)
    xregg = xreg

    if xregg !== nothing
        if is_constant_all(xregg)
            xregg = nothing
        else
            xregg = drop_constant_columns(xregg)

            if is_rank_deficient(xregg; add_intercept = false)
                error("xreg is rank deficient")
            end
            j = .!ismissing.(x) .& .!ismissing.(row_sums(xregg))
            fitt = ols(copy(x), copy(xregg.Data))
            res = residuals(fitt)
            xx[j] .= res
        end
    end

    # Choose d, D
    # stationary => no differencing
    if stationary
        d = 0
        D = 0
    end

    # seasonal differencing choice
    if m == 1
        D = 0
        max_P = 0
        max_Q = 0
    elseif D === missing && length(xx) <= 2 * m
        D = 0
    elseif D === missing
        D = nsdiffs(xx; test = seasonal_test, maxD = max_D, seasonal_test_args...)
        # Ensure xreg not null after seasonal differencing
        if D > 0 && xregg !== nothing
            diffxreg = diff(xregg; differences = D, lag = m)
            if any(is_constant(xregg))
                D -= 1
            end
        end
        # Ensure xx not all missing after seasonal differencing
        if D > 0
            dx_tmp = diff(xx; differences = D, lag = m)
            if all(ismissing, dx_tmp)
                D -= 1
            end
        end
    end

    # Apply seasonal differencing
    if D > 0
        dx = diff(xx; differences = D, lag = m)
    else
        dx = copy(xx)
    end

    # Prepare differenced xreg (seasonal part)
    diffxreg = nothing

    if xregg !== nothing
        if D > 0
            diffxreg = diff_lag(xregg; differences = D, lag = m)
        else
            diffxreg = xregg
        end
    end

    # non-seasonal differencing choice
    if d === missing
        d = ndiffs(dx; test = test, maxd = max_d, test_args...)
        # Ensure xreg not null after additional (non-seasonal) differencing
        if d > 0 && xregg !== nothing
            diffxreg = diff(diffxreg; differences = d, lag = 1)
            if any(is_constant(diffxreg))
                d -= 1
            end
        end
        # Ensure dx not all missing after additional differencing
        if d > 0
            diffdx = diff_lag(dx; differences = d, lag = 1)
            if all(isna, diffdx.data) # TODO
                d -= 1
            end
        end
    end

    # warnings about too much differencing
    if D >= 2
        @warn "Having more than one seasonal difference is not recommended. Consider using only one seasonal difference."
    elseif D + d > 2
        @warn "Having 3 or more differencing operations is not recommended. Consider reducing the total number of differences."
    end

    # apply non-seasonal differencing
    if d > 0
        dx = diff(dx; differences = d, lag = 1)
    end

    # terminal checks
    if length(dx) == 0
        error("Not enough data to proceed")
    elseif is_constant(dx)
        # constant process (after differencing)
        if xreg === nothing
            if D > 0 && d == 0
                fit = arima_rjh(
                    x,
                    m,
                    order = PDQ(0, d, 0),
                    seasonal_order = PDQ(0, D, 0),
                    include_constant = true,
                    fixed = mean2(dx) ./ m,
                    method = method,
                    kwargs...,
                )
            elseif D > 0 && d > 0
                fit = arima_rjh(
                    x,
                    m,
                    order = PDQ(0, d, 0),
                    seasonal_order = PDQ(0, D, 0),
                    method = method,
                    kwargs...,
                )
            elseif d == 2
                fit = arima_rjh(x, m, order = PDQ(0, d, 0), method = method, kwargs...)
            elseif d < 2
                fit = arima_rjh(
                    x,
                    m,
                    order = PDQ(0, d, 0),
                    include_constant = true,
                    fixed = mean2(dx),
                    method = method,
                    kwargs...,
                )
            else
                error(
                    "Data follow a simple polynomial and are not suitable for ARIMA modelling.",
                )
            end
        else
            # perfect regression case
            if D > 0
                fit = arima_rjh(
                    x,
                    m,
                    order = PDQ(0, d, 0),
                    seasonal_order = PDQ(0, D, 0),
                    xreg = xreg,
                    method = method,
                    kwargs...,
                )
            else
                fit = arima_rjh(
                    x,
                    m,
                    order = PDQ(0, d, 0),
                    xreg = xreg,
                    method = method,
                    kwargs...,
                )
            end
        end

        fit.model.x = orig_x
        return fit
    end

    # Seasonal constraints on p/q for m>1
    if m > 1
        if max_P > 0
            max_p = min(max_p, m - 1)
        end
        if max_Q > 0
            max_q = min(max_q, m - 1)
        end
    end

    # Approximation offset via CSS ARIMA(0,d,0)[(0,D,0)]
    offset = compute_approx_offset(
        approximation = approximation,
        x = x,
        d = d,
        D = D,
        m = m,
        xreg = xreg,
        truncate = truncate,
        kwargs...,
    )

    allowdrift = allowdrift && (d + D) == 1
    allowmean = allowmean && (d + D) == 0
    constant = allowdrift || allowmean

    if trace && approximation
        println("\nFitting models using approximations...\n")
    end

    if !stepwise
        bestfit = search_arima(
            x,
            m,
            d,
            D,
            max_p,
            max_q,
            max_P,
            max_Q,
            max_order,
            stationary,
            ic,
            trace,
            approximation,
            xreg,
            offset,
            allowdrift,
            allowmean,
            method = method,
            kwargs...,
        )
        bestfit.lambda = lambda
        bestfit.x = orig_x
        bestfit.fitted < -fitted(bestfit)

        return bestfit
    end

    if length(x) < 10
        start_p = min(start_p, 1)
        start_q = min(start_q, 1)
        start_P = 0
        start_Q = 0
    end

    p, start_p = min(start_p, max_p), min(start_p, max_p)
    q, start_q = min(start_p, max_q), min(start_p, max_q)
    P, start_P = min(start_p, max_P), min(start_p, max_P)
    Q, start_Q = min(start_p, max_Q), min(start_p, max_Q)
    
    result_cols = ["p", "d", "q", "P", "D", "Q", "constant", "ic"]

    result = NamedMatrix(nmodels, result_cols)

    bestfit = fit_custom_arima(
        x,
        m,
        order = PDQ(p, d, q),
        seasonal = PDQ(P, D, Q),
        constant = constant,
        ic = ic,
        trace = trace,
        approximation = approximation,
        offset = offset,
        xreg = xreg,
        method = method,
        kwargs...,
    )

    result = setrow!(result, 1, (p, d, q, P, D, Q, constant, bestfit.ic))

    fit = fit_custom_arima(
        x,
        m,
        order = PDQ(0, d, 0),
        seasonal = PDQ(0, D, 0),
        constant = constant,
        ic = ic,
        trace = trace,
        approximation = approximation,
        offset = offset,
        xreg = xreg,
        method = method,
        kwargs...,
    )

    result = setrow!(result, 2, (p, d, q, P, D, Q, constant, fit.ic))

    if fit.ic < bestfit.ic
        bestfit = fit
        p, q, P, Q = 0, 0, 0, 0
    end

    k = 2

    if max_p > 0 || max_P > 0
        fit = fit_custom_arima(
            x,
            m,
            order = PDQ(0, d, 0),
            seasonal = PDQ(0, D, 0),
            constant = constant,
            ic = ic,
            trace = trace,
            approximation = approximation,
            offset = offset,
            xreg = xreg,
            method = method,
            kwargs...,
        )
    end

    pp = max_p > 0 ? 1 : 0
    PP = m > 1 && max_P > 0 ? 1 : 0
    result = setrow!(result, k + 1, (pp, d, 0, PP, D, 0, constant, fit.ic))

    if fit.ic < bestfit.ic
        bestfit = fit
        p = max_p > 0 ? 1 : 0
        P = m > 1 && max_P > 0 ? 1 : 0
        q, Q = 0, 0
    end

    k += 1
    # Basic MA model
    if max_q > 0 || max_Q > 0
        qq = max_q > 0 ? 1 : 0
        QQ = m > 1 && max_Q > 0 ? 1 : 0

        fit = fit_custom_arima(
            x,
            m,
            order = PDQ(0, d, qq),
            seasonal = PDQ(0, D, QQ),
            constant = constant,
            ic = ic,
            trace = trace,
            approximation = approximation,
            offset = offset,
            xreg = xreg,
            method = method,
            kwargs...,
        )

        result = setrow!(result, k + 1, (0, d, qq, 0, D, QQ, constant, fit.ic))
        if fit.ic < bestfit.ic
            bestfit = fit
            p, P = 0, 0
            q = max_q > 0 ? 1 : 0
            Q = m > 1 && max_Q > 0 ? 1 : 0
        end
        k += 1
    end

    # Null model with no constant
    if constant
        fit = fit_custom_arima(
            x,
            m,
            order = PDQ(0, d, 0),
            seasonal = PDQ(0, D, 0),
            constant = false,
            ic = ic,
            trace = trace,
            approximation = approximation,
            offset = offset,
            xreg = xreg,
            method = method,
            kwargs...,
        )

        result = setrow!(result, k + 1, (0, d, 0, 0, D, 0, 0, fit.ic))

        if fit.ic < bestfit.ic
            bestfit = fit
            p, P, q, Q = 0, 0, 0, 0
        end
        k += 1
    end

    startk = 0
    while startk < k && k < nmodels
        startk = k
        newm = newmodel(
            p,
            d,
            q,
            P - 1,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )
        if P > 0 && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P - 1, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )
            result = setrow!(result, k, (p, d, q, P - 1, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                P -= 1
                continue
            end

        end

        newm = newmodel(
            p,
            d,
            q,
            P,
            D,
            Q - 1,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if Q > 0 && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P, D, Q - 1),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P, D, Q - 1, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                Q -= 1
                continue
            end
        end

        newm = newmodel(
            p,
            d,
            q,
            P + 1,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if P < max_P && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P + 1, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P + 1, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                P += 1
                continue
            end
        end

        newm = newmodel(
            p,
            d,
            q,
            P,
            D,
            Q + 1,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if Q < max_Q && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P, D, Q + 1),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P, D, Q + 1, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                Q += 1
                continue
            end
        end

        newm = newmodel(
            p,
            d,
            q,
            P - 1,
            D,
            Q - 1,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if Q > 0 && P > 0 && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P - 1, D, Q - 1),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P - 1, D, Q - 1, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                Q -= 1
                P -= 1
                continue
            end
        end

        newm = newmodel(
            p,
            d,
            q,
            P - 1,
            D,
            Q + 1,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if Q < max_Q && P > 0 && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P - 1, D, Q + 1),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P - 1, D, Q + 1, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                Q += 1
                P -= 1
                continue
            end
        end


        newm = newmodel(
            p,
            d,
            q,
            P + 1,
            D,
            Q - 1,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if Q > 0 && P < max_P && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P + 1, D, Q - 1),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P + 1, D, Q - 1, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                Q -= 1
                P += 1
                continue
            end
        end



        newm = newmodel(
            p,
            d,
            q,
            P + 1,
            D,
            Q + 1,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if Q < max_Q && P < max_P && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q),
                seasonal = PDQ(P + 1, D, Q + 1),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q, P + 1, D, Q + 1, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                Q += 1
                P += 1
                continue
            end
        end


        newm = newmodel(
            p - 1,
            d,
            q,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if p > 0 && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p - 1, d, q),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p - 1, d, q, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                p -= 1
                continue
            end
        end

        newm = newmodel(
            p,
            d,
            q - 1,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if q > 0 && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q - 1),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q - 1, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                q -= 1
                continue
            end
        end


        newm = newmodel(
            p + 1,
            d,
            q,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if p < max_p && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p + 1, d, q),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p + 1, d, q, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                p += 1
                continue
            end
        end

        newm = newmodel(
            p,
            d,
            q + 1,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if q < max_q && newm
            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p, d, q + 1),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p, d, q + 1, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                q += 1
                continue
            end
        end


        newm = newmodel(
            p - 1,
            d,
            q - 1,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if q > 0 && p > 0 && newm

            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p - 1, d, q - 1),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p - 1, d, q - 1, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                p -= 1
                q -= 1
                continue
            end
        end


        newm = newmodel(
            p - 1,
            d,
            q + 1,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if q > max_q && p > 0 && newm

            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p - 1, d, q + 1),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p - 1, d, q + 1, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                p -= 1
                q += 1
                continue
            end
        end


        newm = newmodel(
            p + 1,
            d,
            q - 1,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if q > 0 && p > max_p && newm

            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p + 1, d, q - 1),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p + 1, d, q - 1, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                p += 1
                q -= 1
                continue
            end
        end


        newm = newmodel(
            p + 1,
            d,
            q + 1,
            P,
            D,
            Q,
            constant,
            get_elements(result, row = collect(1:k), col = result_cols)
        )

        if q < max_q && p < max_p && newm

            k += 1
            if k > nmodels
                continue
            end

            fit = fit_custom_arima(
                x,
                m,
                order = PDQ(p + 1, d, q + 1),
                seasonal = PDQ(P, D, Q),
                constant = constant,
                ic = ic,
                trace = trace,
                approximation = approximation,
                offset = offset,
                xreg = xreg,
                method = method,
                kwargs...,
            )

            result = setrow!(result, k, (p + 1, d, q + 1, P, D, Q, constant, fit.ic))
            if fit.ic < bestfit.ic
                bestfit = fit
                p += 1
                q += 1
                continue
            end
        end


        if allowdrift || allowmean

            newm = newmodel(
                p,
                d,
                q,
                P,
                D,
                Q,
                constant,
                get_elements(result, row = collect(1:k), col = result_cols)
            )

            if newm

                k += 1
                if k > nmodels
                    continue
                end

                fit = fit_custom_arima(
                    x,
                    m,
                    order = PDQ(p, d, q),
                    seasonal = PDQ(P, D, Q),
                    constant = constant,
                    ic = ic,
                    trace = trace,
                    approximation = approximation,
                    offset = offset,
                    xreg = xreg,
                    method = method,
                    kwargs...,
                )

                result = setrow!(result, k, (p, d, q, P, D, Q, constant, fit.ic))
                if fit.ic < bestfit.ic
                    bestfit = fit
                    constant = !constant
                end
            end

        end

    end

    if k > nmodels
        @warn "Stepwise search was stopped early due to reaching the model number limit: $nmodels"
    end

    if approximation #&& bestfit$arma
        if trace
            println("Now re-fitting the best model(s) without approximations...")
        end
    end

    icorder = as_vector(get_elements(result, col = 8))
    nmodels = count(v -> !(ismissing(v) || isnan(v)), icorder)
    println(icorder)
    icorder = sortperm(icorder)

    for i = 1:nmodels
        mod = get_elements(result, row = i)
        println("mod = ", mod)

        fit = fit_custom_arima(
            x,
            m,
            order = PDQ(as_integer(mod[1]), d, as_integer(mod[3])),
            seasonal = PDQ(as_integer(mod[4]), D, as_integer(mod[6])),
            constant = mod[7] > 0 ? true : false,
            ic = ic,
            trace = trace,
            approximation = false,
            xreg = xreg,
            method = method,
            kwargs...,
        )

        if fit.ic < Inf
            bestfit = fit
            break
        end
    end

    if trace
        println("Best model found!")
    end

    return bestfit
end
