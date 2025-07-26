function prepare_drift(model, x, xreg; drift_col=1)
    # Get drift vector
    drift_vec = model.xreg[:, drift_col]

    n_train = length(model.x)
    time_train = 1:n_train
    X = [ones(n_train) time_train]
    coef = X \ drift_vec

    n_new = length(x)
    time_new = 1:n_new
    newxreg = coef[1] .+ coef[2] .* time_new

    if xreg !== nothing && !isempty(xreg)
        if size(xreg, 1) != n_new
            error("Number of rows in xreg ($(size(xreg,1))) does not match length of x ($(n_new)).")
        end
        xreg = hcat(newxreg, xreg)
    else
        xreg = reshape(newxreg, :, 1)
    end
    return xreg
end

function refit_arima_model(x, model, xreg, method, use_intercept)
    # Extract model orders and seasonal info
    order = PDQ(model.arma[1], model.arma[6], model.arma[2]) # (p, d, q)
    seasonal_order = PDQ(model.arma[3], model.arma[7], model.arma[4]) # (P, D, Q)
    m = model.arma[5]
    fixed_coefs = model.coef

    # Prepare keyword arguments
    kwargs = Dict(
        :order => order,
        :seasonal => seasonal_order,
        :include_mean => use_intercept,
        :method => method,
        :fixed => fixed_coefs
    )

    if xreg !== nothing
        kwargs[:xreg] = xreg
    end

    refit = arima(x, m; kwargs...)

    # Set variance-covariance matrix of coefs to zeros, as in R
    if hasproperty(refit, :var_coef)
        refit.var_coef = zeros(length(refit.coef), length(refit.coef))
    end

    if xreg !== nothing && hasproperty(refit, :xreg)
        refit.xreg = xreg
    end

    if hasproperty(refit, :sigma2)
        refit.sigma2 = model.sigma2
    end

    return refit
end

function arima_rjh(y, m;
    order::PDQ = PDQ(0, 0, 0),
    seasonal::PDQ = PDQ(0, 0, 0),
    xreg = nothing,
    include_mean = true,
    include_drift = false,
    include_constant = nothing, # Should be nothing or bool
    lambda = nothing,
    biasadj = false,
    method = "CSS-ML",
    model = nothing, # should be an ArimaFit or nothing
    x = y,
    kwargs...
)
    origx = y
    method = match_arg(method, ["CSS-ML", "ML", "CSS"])

    if lambda !== nothing
        x = box_cox(x, m, lambda=lambda)
    end

    if xreg !== nothing
        if !(eltype(xreg) <: Number)
            error("xreg should be a numeric matrix or vector")
        end
        xreg = convert(Matrix{Float64}, xreg)
    end

    # Check Seasonal Model
    if m <=1 
        seasonal = PDQ(0,0,0)
        m = 1
        if length(x) <= order.d
            @error "Not enough data to fit the model: not enough data for regular differencing."
        end
    else
        if length(x) <= order.d + seasonal.d * m
            @error "Not enough data to fit the model: not enough data for seasonal differencing."
        end
    end

    # include_constant logic
    if include_constant !== nothing
        if include_constant
            include_mean = true
            if (order.d + seasonal.order.d) == 1
                include_drift = true
            end
        else
            include_mean = false
            include_drift = false
        end
    end

    if (order.d + seasonal.order.d) > 1 && include_drift
        @warn("No drift term fitted as the order of difference is 2 or more.")
        include_drift = false
    end

    if isa(model, ArimaFit)
        use_drift = "intercept" in propertynames(model.coef)
        use_intercept = "intercept" in propertynames(model.coef)
        if use_drift
            xreg = prepare_drift(model, x, xreg)
        end
        if get(model, :xreg, nothing) !== nothing && (xreg === nothing || size(xreg,2) != size(model.xreg,2))
            error("Regressors do not match those used for the fitted model")
        end
        fit = refit_arima_model(x, m, model, xreg, method, use_drift, use_intercept)
        fit.lambda = get(model, :lambda, nothing)
    else
        # Handle drift for new model
        if include_drift
            drift_vec = collect(1:length(x))
            if xreg !== nothing
                xreg = hcat(drift_vec, xreg)
            else
                xreg = reshape(drift_vec, :, 1)
            end
        end

        fit = arima(x, m;
            order=order,
            seasonal=seasonal.order,
            period=seasonal.period,
            include_mean=include_mean,
            method=method,
            xreg=xreg,
            kwargs...
        )
        fit.lambda = lambda

        npar = sum(fit.mask) + 1 
        missing = isnan.(fit.residuals)
        firstnonmiss = findfirst(!ismissing, fit.residuals)
        lastnonmiss = findlast(!ismissing, fit.residuals)
        n = sum(.!missing[firstnonmiss:lastnonmiss])
        nstar = n - order[2] - seasonal.order[2] * seasonal.period
        fit.sigma2 = sum(skipmissing(fit.residuals).^2) / (nstar - npar + 1)
    end

    fit.series = series
    fit.xreg = xreg
    fit.x = origx
    
    return fit
end