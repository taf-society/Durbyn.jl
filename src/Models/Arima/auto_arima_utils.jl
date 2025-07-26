using Polynomials
using Durbyn.Arima


function get_missing_indices(x)
    miss = ismissing.(x) .| isnan.(x)
    notmissing = .!miss
    firstnonmiss = findfirst(notmissing)
    lastnonmiss = findlast(notmissing)
    return (miss=miss, first=firstnonmiss, last=lastnonmiss)
end

function get_n_obs(x, first, last)
    sum(!(ismissing(xi) || isnan(xi)) for xi in @view x[first:last])
end

function should_use_season(seasonal, m)
    sum(seasonal) > 0 && m > 0
end

function choose_method(method, approximation)
    method !== nothing ? method : (approximation ? "CSS" : "CSS-ML")
end


function add_drift_to_xreg(x, xreg::Union{Nothing,AbstractMatrix}=nothing)
    drift = collect(1:length(x))
    drift_mat = reshape(drift, :, 1)
    if xreg === nothing
        return drift_mat
    else
        return hcat(drift_mat, xreg)
    end
end



function fit_arima_model(x;
    order=PDQ(0, 0, 0),
    seasonal=PDQ(0, 0, 0),
    constant=true,
    method="CSS",
    xreg=nothing,
    use_season=false,
    m,
    pass_mean=false,
    kwargs...)

    try
        if use_season
            if pass_mean
                fit = arima(x, m;
                    order=order,
                    seasonal=seasonal,
                    xreg=xreg,
                    include_mean=constant,
                    method=method,
                    kwargs...)
            else
                fit = arima(x, m;
                    order=order,
                    seasonal=seasonal,
                    xreg=xreg,
                    method=method,
                    kwargs...)
            end
        else
            if pass_mean
                fit = arima(x, m;
                    order=order,
                    seasonal=PDQ(0, 0, 0),  # No seasonality
                    xreg=xreg,
                    include_mean=constant,
                    method=method,
                    kwargs...)
            else
                fit = arima(x, m;
                    order=order,
                    seasonal=PDQ(0, 0, 0),  # No seasonality
                    xreg=xreg,
                    method=method,
                    kwargs...)
            end
        end
        return fit
    catch err
        return err
    end
end

function get_nxreg(xreg)
    if xreg === nothing
        return 0
    elseif ndims(xreg) == 1
        return 1
    else
        return size(xreg, 2)
    end
end

function calculate_ic(fit::ArimaFit, nstar, npar, offset, ic, method)
    if method == "CSS"
        fit.aic = offset + nstar * log(fit.sigma2) + 2 * npar
    end
    if !isnan(fit.aic)
        fit.bic = fit.aic + npar * (log(nstar) - 2)
        fit.aicc = fit.aic + 2 * npar * (npar + 1) / (nstar - npar - 1)
        fit.ic = ic == "bic" ? fit.bic : ic == "aicc" ? fit.aicc : fit.aic
    else
        fit.aic = fit.bic = fit.aicc = fit.ic = Inf
    end
    return fit
end

function check_roots(fit, order, seasonal, m)
    minroot = 2.0
    # AR roots
    if order.p + seasonal.p > 0
        phi = fit.model.phi
        k = abs.(phi) .> 1e-8
        last_nonzero = any(k) ? findlast(k) : 0
        if last_nonzero > 0
            try
                proots = roots(Polynomial([1.0; -phi[1:last_nonzero]]))
                minroot = min(minroot, minimum(abs.(proots)))
            catch
                fit.ic = Inf
            end
        end
    end
    # MA roots
    if order.q + seasonal.q > 0 && fit.ic < Inf
        theta = fit.model.theta
        k = abs.(theta) .> 1e-8
        last_nonzero = any(k) ? findlast(k) : 0
        if last_nonzero > 0
            try
                proots = roots(Polynomial([1.0; theta[1:last_nonzero]]))
                minroot = min(minroot, minimum(abs.(proots)))
            catch
                fit.ic = Inf
            end
        end
    end
    minroot
end

function checkarima(fit::ArimaFit)
    try
        return any(isnan.(sqrt.(diag(fit.var_coef))))
    catch
        return true
    end
end

function should_reject_fit(minroot, fit)
    minroot < 1.01 || checkarima(fit)
end

function handle_fit_error(fit, order, seasonal, m, constant, trace)
    if occursin("unused argument", sprint(showerror, fit))
        error(first(split(sprint(showerror, fit), '\n')))
    end
    if trace
        println("\n ARIMA($(order.p),$(order.d),$(order.q))",
            (seasonal.p + seasonal.d + seasonal.q > 0 && m > 0) ?
            "($(seasonal.p),$(seasonal.d),$(seasonal.q))[$m]" : "",
            constant && (order.d + seasonal.d == 0) ? " with non-zero mean" : "",
            constant && (order.d + seasonal.d == 1) ? " with drift" : "",
            !constant && (order.d + seasonal.d == 0) ? " with zero mean" : "",
            " : ", Inf)
    end
    return (ic=Inf,)
end

get_pdq(x) = hasproperty(x, :p) ? (x.p, x.d, x.q) : (x[1], x[2], x[3])
get_sum(x) = hasproperty(x, :p) ? x.p + x.d + x.q : sum(x)

function arima_trace_str(order, seasonal, m, constant)
    (p, d, q) = get_pdq(order)
    (P, D, Q) = get_pdq(seasonal)
    seasonal_part = get_sum(seasonal) > 0 && m > 0 ? "($P,$D,$Q)[$m]" : ""
    if constant && (d + D == 0)
        mean_str = " with non-zero mean"
    elseif constant && (d + D == 1)
        mean_str = " with drift        "
    elseif !constant && (d + D == 0)
        mean_str = " with zero mean    "
    else
        mean_str = "                   "
    end
    # Combine all parts, pad to fixed width
    s = " ARIMA($(p),$(d),$(q))$(seasonal_part)$(mean_str) : Inf"

    return s
end

function handle_fit_error(fit, order, seasonal, m, constant, trace)
    if occursin("unused argument", sprint(showerror, fit))
        error(first(split(sprint(showerror, fit), '\n')))
    end
    if trace
        println()
        println(arima_trace_str(order, seasonal, m, constant))
    end
    return (ic=Inf,)
end

function fit_custom_arima(x, m;
        order=PDQ(0,0,0),
        seasonal=PDQ(0,0,0),
        constant=true,
        ic="aic",
        trace=false,
        approximation=false,
        offset=0,
        xreg=nothing,
        method=nothing,
        kwargs...)

    miss_info = get_missing_indices(x)
    n = get_n_obs(x, miss_info.first, miss_info.last)
    use_season = should_use_season((seasonal.p, seasonal.d, seasonal.q), m)
    diffs = order.d + seasonal.d
    method = choose_method(method, approximation)
    drift_case = (diffs == 1 && constant)

    if drift_case
        xreg = add_drift_to_xreg(x, xreg)
    end
    pass_mean = !drift_case

    fit = fit_arima_model(x;
        order=order,
        seasonal=seasonal,
        include_mean=constant,
        method=method,
        xreg=xreg,
        m=m,
        pass_mean=pass_mean,
        use_season=use_season,
        kwargs...)

    if !(fit isa Exception)
        nstar = n - order.d - seasonal.d * m
        if diffs == 1 && constant
            fit.xreg = xreg
        end

        nxreg = get_nxreg(xreg)

        npar = sum(fit.mask) + 1

        if method == "CSS"
            fit.aic = offset + nstar * log(fit.sigma2) + 2 * npar
        end

        fit = calculate_ic(fit, nstar, npar, offset, ic, method)

        fit.sigma2 = sum(fit.residuals .^ 2) / (nstar - npar + 1)

        minroot = check_roots(fit, order, seasonal, m)
        if should_reject_fit(minroot, fit)
            fit.ic = Inf
        end

        fit.xreg = xreg

        if trace
            println()
            println(arima_trace_str(order, seasonal, m, constant), " :", fit.ic)
        end

        return fit
    else
        return handle_fit_error(fit, order, seasonal, m, constant, trace)
    end
end

# Generate all model combinations
function generate_model_grid(max_p, max_q, max_P, max_Q, maxK, max_order)
    # Create all combinations
    combos = [
        (i=i, j=j, I=I, J=J, K=K)
        for i in 0:max_p, j in 0:max_q, I in 0:max_P, J in 0:max_Q, K in 0:maxK
        if (i + j + I + J) <= max_order
    ]
    return combos
end

# Helper: Fit a single ARIMA model and return fit + K
function fit_arima_model(x, m, d, D, params::NamedTuple, ic, trace, approximation, offset, xreg; kwargs...)
    # Unpack params
    i, j, I, J, K = params.i, params.j, params.I, params.J, params.K

    # Fit model using fit_custom_arima
    fit = fit_custom_arima(
        x, m;
        order = PDQ(i, d, j),
        seasonal = PDQ(I, D, J),
        constant = (K == 1),
        ic = ic,
        trace = trace,
        approximation = approximation,
        offset = offset,
        xreg = xreg,
        kwargs...
    )
    return (fit = fit, K = K)
end

# Helper: Find best model from grid (serial)
function find_best_arima_serial(x, m, d, D, grid, ic, trace, approximation, offset, xreg; kwargs...)
    best_ic = Inf
    bestfit = nothing
    constant = nothing
    for params in grid
        result = fit_arima_model(x, m, d, D, params, ic, trace, approximation, offset, xreg; kwargs...)
        fit = result.fit
        K = result.K
        ic_sym = Symbol(ic)
        # Use robust access and skip if missing or Inf
        fit_ic = hasproperty(fit, ic_sym) ? getproperty(fit, ic_sym) : nothing
        if fit_ic === nothing || fit_ic == Inf
            continue
        end
        if fit_ic < best_ic
            best_ic = fit_ic
            bestfit = fit
            constant = (K == 1)
        end
    end
    return (bestfit = bestfit, best_ic = best_ic, constant = constant)
end

@everywhere begin
    # Define PDQ, ArimaFit, etc. here, or import your module

    # Worker function for fitting
    function fit_arima_worker(x, m, d, D, params, ic, trace, approximation, offset, xreg; kwargs...)
        try
            result = fit_arima_model(x, m, d, D, params, ic, trace, approximation, offset, xreg; kwargs...)
            fit = result.fit
            K = result.K
            # Skip if IC value is missing
            fit_ic = getproperty(fit, Symbol(ic), nothing)
            if fit_ic === nothing
                return nothing
            end
            return (fit=fit, K=K, fit_ic=fit_ic)
        catch
            return nothing
        end
    end
end

# addprocs(4)  # or whatever needed

function find_best_arima_parallel(x, m, d, D, grid, ic, trace, approximation, offset, xreg, num_cores=nothing; kwargs...)
    # Ensure sufficient processes
    if isnothing(num_cores)
        num_cores = nprocs()
    elseif num_cores > nprocs()
        addprocs(num_cores - nprocs())
    end

    # Create closures that capture all local variables
    paramlist = grid

    # Use pmap for robust load-balancing
    results = pmap(params -> fit_arima_worker(x, m, d, D, params, ic, trace, approximation, offset, xreg; kwargs...), paramlist)

    # Filter out failures
    filtered = filter(!isnothing, results)

    # Find best model by IC
    best_ic = Inf
    bestfit = nothing
    constant = nothing
    for res in filtered
        fit = res.fit
        K = res.K
        fit_ic = res.fit_ic
        if fit_ic < best_ic
            best_ic = fit_ic
            bestfit = fit
            constant = (K == 1)
        end
    end

    return (bestfit=bestfit, best_ic=best_ic, constant=constant)
end

function refit_best_arima(x, m, bestfit, constant, ic, xreg; kwargs...)
    order    = PDQ(bestfit.arma[1], bestfit.arma[6], bestfit.arma[2])
    seasonal = PDQ(bestfit.arma[3], bestfit.arma[7], bestfit.arma[4])
    fit = fit_custom_arima(
        x, m;
        order = order,
        seasonal = seasonal,
        constant = constant,
        ic = ic,
        trace = false,
        approximation = false,
        xreg = xreg,
        kwargs...
    )
    return fit
end

function search_arima_base(
    x, m;
    d=NaN, D=NaN, max_p=5, max_q=5, max_P=2, max_Q=2, max_order=5,
    stationary=false, ic="aic", trace=false, approximation=false,
    xreg=nothing, offset=0,
    allowdrift=true, allowmean=true,
    parallel=false, num_cores=2,
    kwargs...
)
    # IC: handle vector
    ic = isa(ic, AbstractVector) ? ic[1] : ic

    # allowdrift/allowmean logic
    allowdrift = allowdrift && (d + D == 1)
    allowmean  = allowmean && (d + D == 0)
    maxK = Int(allowdrift || allowmean)

    # Generate model grid
    grid = generate_model_grid(max_p, max_q, max_P, max_Q, maxK, max_order)

    # Search for best model
    if !parallel
        result = find_best_arima_serial(x, m, d, D, grid, ic, trace, approximation, offset, xreg; kwargs...)
    else
        result = find_best_arima_parallel(x, m, d, D, grid, ic, trace, approximation, offset, xreg, num_cores; kwargs...)
    end
    bestfit   = result.bestfit
    best_ic   = result.best_ic
    constant  = result.constant

    # Approximation refit if requested and initial fit was with approximation
    if !isnothing(bestfit) && approximation
        if trace
            println("\n\n Now re-fitting the best model(s) without approximations...\n")
        end
        newbestfit = refit_best_arima(x, m, bestfit, constant, ic, xreg; kwargs...)
        ic_val = getproperty(newbestfit, Symbol(ic), nothing)
        if !isnothing(ic_val) && ic_val != Inf
            bestfit = newbestfit
        end
    end

    if isnothing(bestfit)
        return nothing
    end

    if hasproperty(bestfit, :ic)
        bestfit.ic = nothing
    end

    return bestfit
end

function search_arima(
    x, m;
    d=NaN, D=NaN, max_p=5, max_q=5, max_P=2, max_Q=2, max_order=5,
    stationary=false, ic="aic", trace=false, approximation=false,
    xreg=nothing, offset=0,
    allowdrift=true, allowmean=true,
    parallel=false, num_cores=2,
    max_fallback=1,
    kwargs...
)
    # Initial call
    bestfit = search_arima_base(
        x, m; d=d, D=D, max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
        max_order=max_order, stationary=stationary, ic=ic, trace=trace,
        approximation=approximation, xreg=xreg, offset=offset,
        allowdrift=allowdrift, allowmean=allowmean,
        parallel=parallel, num_cores=num_cores, kwargs...
    )

    # If approximation was requested and failed, do fallback (only once)
    if !isnothing(bestfit) && approximation && max_fallback > 0
        ic_str = isa(ic, AbstractVector) ? ic[1] : ic
        ic_val = getproperty(bestfit, Symbol(ic_str), nothing)
        if isnothing(ic_val) || ic_val == Inf
            if trace
                println("Refitting without approximation due to failed IC value or Inf.")
            end
            # Retry without approximation (decrement max_fallback)
            return search_arima(
                x, m; d=d, D=D, max_p=max_p, max_q=max_q, max_P=max_P, max_Q=max_Q,
                max_order=max_order, stationary=stationary, ic=ic, trace=trace,
                approximation=false, xreg=xreg, offset=offset,
                allowdrift=allowdrift, allowmean=allowmean,
                parallel=parallel, num_cores=num_cores,
                max_fallback=max_fallback-1, kwargs...
            )
        end
    end

    if isnothing(bestfit)
        error("No ARIMA model able to be estimated")
    end

    return bestfit
end


