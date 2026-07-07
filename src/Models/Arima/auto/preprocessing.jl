# ─── Series preprocessing for automatic model selection ───────────────────────
#
# Mirrors core/covariance.jl + core/hyperparameters.jl: data preparation that
# runs once before the search loop begins. Each function takes explicit typed
# arguments and returns plain values or NamedTuples.

"""
    analyze_series(x) -> (firstnonmiss, lastnonmiss, serieslength, x_trim)

Trim leading and trailing missing values, returning indices and the trimmed series.
"""
function analyze_series(x::AbstractVector)
    miss = ismissing.(x)
    first = findfirst(!, miss)
    last = findlast(!, miss)

    if isnothing(first)
        return (firstnonmiss = nothing, lastnonmiss = nothing, serieslength = 0, x_trim = x)
    end

    serieslength = count(!, @view miss[first:last])
    x_trim = x[first:last]

    (firstnonmiss = first, lastnonmiss = last, serieslength = serieslength, x_trim = x_trim)
end

"""
    preprocess_series(x, xreg, m; lambda) -> NamedTuple

Handle Box-Cox transform and compute OLS residuals for unit-root testing when
xreg is present. Returns (x_work, xreg_work, x_for_tests, lambda).
"""
function preprocess_series(x::AbstractVector, xreg, m::Int;
                           lambda::Union{Nothing,Real})
    x_work = float.(x)

    if !isnothing(lambda)
        x_work, lambda = box_cox(x_work, m, lambda = lambda)
    end

    x_for_tests = copy(x_work)
    xreg_work = xreg

    if !isnothing(xreg_work)
        if is_constant_all(xreg_work)
            xreg_work = nothing
        else
            xreg_work = drop_constant_columns(xreg_work)
            if is_rank_deficient(xreg_work)
                throw(ArgumentError("xreg is rank deficient"))
            end
            valid = .!ismissing.(x_work) .& .!ismissing.(row_sums(xreg_work))
            xreg_with_intercept = hcat(ones(size(xreg_work.data, 1)), xreg_work.data)
            ols_fit = ols(Float64.(x_work[valid]), xreg_with_intercept[valid, :])
            res = residuals(ols_fit)
            x_for_tests[valid] .= res
        end
    end

    (x_work = x_work, xreg_work = xreg_work, x_for_tests = x_for_tests, lambda = lambda)
end

"""
    select_differencing(x_for_tests, xreg, m; d, D, max_d, max_D,
                        stationary, test, test_args, seasonal_test,
                        seasonal_test_args) -> (d, D)

Choose seasonal (D) and non-seasonal (d) differencing orders via unit-root
and seasonal-strength tests, with safety checks for constant/all-missing
post-differencing series.
"""
function select_differencing(x_for_tests::AbstractVector, xreg,
                             m::Int;
                             d::Union{Nothing,Int},
                             D::Union{Nothing,Int},
                             max_d::Int, max_D::Int,
                             stationary::Bool,
                             test::Symbol, test_args,
                             seasonal_test::Symbol, seasonal_test_args)
    if stationary
        return (d = 0, D = 0)
    end

    # ── Seasonal differencing ──
    if m == 1
        D = 0
    elseif isnothing(D) && length(x_for_tests) <= 2 * m
        D = 0
    elseif isnothing(D)
        D = nsdiffs(x_for_tests, m; test = seasonal_test, max_D = max_D, seasonal_test_args...)
        if D > 0 && !isnothing(xreg)
            diffxreg = diff(xreg; difference_order = D, lag_steps = m)
            if any(is_constant(diffxreg))
                D -= 1
            end
        end
        if D > 0
            dx_tmp = diff(x_for_tests; difference_order = D, lag_steps = m)
            if all(ismissing, dx_tmp)
                D -= 1
            end
        end
    end

    # Apply seasonal differencing for non-seasonal test
    dx = D > 0 ? diff(x_for_tests; difference_order = D, lag_steps = m) : copy(x_for_tests)
    diffxreg = nothing
    if !isnothing(xreg)
        diffxreg = D > 0 ? diff(xreg; difference_order = D, lag_steps = m) : xreg
    end

    # ── Non-seasonal differencing ──
    if isnothing(d)
        d = ndiffs(dx; test = test, max_d = max_d, test_args...)
        if d > 0 && !isnothing(xreg)
            diffxreg = diff(diffxreg; difference_order = d, lag_steps = 1)
            if any(is_constant(diffxreg))
                d -= 1
            end
        end
        if d > 0
            diffdx = diff(dx; difference_order = d, lag_steps = 1)
            if all(ismissingish, diffdx)
                d -= 1
            end
        end
    end

    if D >= 2
        @warn "Having more than one seasonal difference is not recommended. Consider using only one seasonal difference."
    elseif D + d > 2
        @warn "Having 3 or more differencing operations is not recommended. Consider reducing the total number of differences."
    end

    (d = d, D = D)
end

"""
    fit_constant_series(x, m, d, D, dx, xreg, method, arima_kwargs) -> ArimaFit

Handle the edge case where the (possibly differenced) series is constant.
Returns an ArimaFit for the appropriate trivial model.
"""
function fit_constant_series(x, m::Int, d::Int, D::Int, dx,
                             xreg::Union{Nothing,NamedMatrix},
                             method::Union{Nothing,Symbol},
                             arima_kwargs::Base.Pairs)
    if !isnothing(xreg)
        seasonal_order = D > 0 ? PDQ(0, D, 0) : PDQ(0, 0, 0)
        return arima_rjh(x, m, order = PDQ(0, d, 0), seasonal_order = seasonal_order,
                         xreg = xreg, method = method, arima_kwargs...)
    end

    if D > 0 && d == 0
        return arima_rjh(x, m, order = PDQ(0, d, 0), seasonal_order = PDQ(0, D, 0),
                         include_constant = true, fixed = [mean2(dx) / m],
                         method = method, arima_kwargs...)
    elseif D > 0 && d > 0
        return arima_rjh(x, m, order = PDQ(0, d, 0), seasonal_order = PDQ(0, D, 0),
                         method = method, arima_kwargs...)
    elseif d == 2
        return arima_rjh(x, m, order = PDQ(0, d, 0), method = method, arima_kwargs...)
    elseif d < 2
        return arima_rjh(x, m, order = PDQ(0, d, 0), include_constant = true,
                         fixed = [mean2(dx)], method = method, arima_kwargs...)
    else
        error("Data follow a simple polynomial and are not suitable for ARIMA modelling.")
    end
end

"""
    compute_approx_offset(x, d, D, m, xreg, truncate, arima_kwargs) -> Float64

Compute the CSS-based likelihood offset used during approximate search.
Fits ARIMA(0,d,0)(0,D,0) via CSS and returns offset = -2·loglik - n·log(σ²).
Returns 0.0 if fitting fails.
"""
function compute_approx_offset(
    x::AbstractVector,
    d::Int, D::Int, m::Int,
    xreg::Union{NamedMatrix,Nothing},
    truncate::Union{Int,Nothing},
    arima_kwargs::Base.Pairs,
)
    xx = x
    xreg_trunc = xreg
    n_orig = length(xx)

    if !isnothing(truncate) && n_orig > truncate
        start_idx = n_orig - truncate + 1
        xx = collect(@view xx[start_idx:end])
        if !isnothing(xreg_trunc)
            nrows = size(xreg_trunc.data, 1)
            if nrows == n_orig
                xreg_trunc = select_rows(xreg_trunc, start_idx:nrows)
            end
        end
    end

    n = length(xx)

    try
        seasonal_order = D == 0 ? PDQ(0, 0, 0) : PDQ(0, D, 0)
        fit = arima(xx, m; order = PDQ(0, d, 0), seasonal = seasonal_order,
                    xreg = xreg_trunc, arima_kwargs...)
        return -2 * fit.loglik - n * log(fit.sigma2)
    catch
        return 0.0
    end
end
