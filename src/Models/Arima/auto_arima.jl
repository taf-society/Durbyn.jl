function analyze_series(x::AbstractVector)
    miss = ismissing.(x)
    first = findfirst(!, miss)
    last = findlast(!, miss)

    if first === nothing
        return (firstnonmiss = nothing, serieslength = 0, x_trim = x)
    end

    serieslength = count(!, @view miss[first:last])
    x_trim = x[first:end]   # trim leading missings only

    (firstnonmiss = first, serieslength = serieslength, x_trim = x_trim)
end

function compute_approx_offset(;
    approximation::Bool,
    x::AbstractVector,
    d::Int,
    D::Int,
    m::Int = 1,
    xreg::Union{NamedMatrix,Nothing} = nothing,              # matrix/vector or nothing
    truncate::Union{Int,Nothing} = nothing,
    kwargs... 
    )
    # no approximation -> zero offset
    if !approximation
        #return (offset = 0.0, fit = nothing)
        return 0.0
    end


    xx = x
    Xreg = xreg
    N0 = length(xx)

    # truncate tail of x (and xreg if it aligns with x)
    if truncate !== nothing && N0 > truncate
        start_idx = N0 - truncate + 1
        xx = collect(@view xx[start_idx:end])

        if Xreg !== nothing
            # row count of xreg
            nrows = size(Xreg.data, 1)
            if nrows == N0
                Xreg = get_elements(Xreg, row = collect(start_idx:nrows))
            end
        end
    end

    serieslength = length(xx)

    # quick ARIMA fit and offset
    try
        fit = if D == 0
            arima(
                xx,
                m,
                order = PDQ(0, d, 0),
                seasonal_order = PDQ(0, 0, 0),
                xreg = Xreg,
                kwargs...,
            )
        else
            arima(
                xx,
                m;
                order = PDQ(0, d, 0),
                seasonal_order = PDQ(0, D, 0),
                xreg = Xreg,
                kwargs...,
            )
        end

        loglik = fit.loglik
        sigma2 = fit.sigma2
        offset = -2 * loglik - serieslength * log(sigma2)
        #return (offset=offset, fit=fit)
        return offset
    catch
        # mirrors try-error fallback: offset <- 0
        #return (offset=0.0, fit=nothing)
        return 0.0
    end
end


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

    result = NamedMatrix(nmodels, ["p", "d", "q", "P", "D", "Q", "constant", "ic"])

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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
            get_elements(result, row = collect(1:k)),
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
                get_elements(result, row = collect(1:k)),
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

    icorder = get_elements(result, col = 8)
    nmodels = count(v -> !(ismissing(v) || isnan(v)), icorder)
    icorder = sortperm(icorder)

    for i = 1:nmodels
        mod = get_elements(result, row = i)

        fit = fit_custom_arima(
            x,
            m,
            order = PDQ(mod[1], d, mod[3]),
            seasonal = PDQ(mod[4], D, mod[6]),
            constant = mod[7],
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
