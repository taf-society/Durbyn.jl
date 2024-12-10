export ets

struct EtsRefit
    model::String
    alpha::Union{AbstractFloat,Nothing,Bool}
    beta::Union{AbstractFloat,Nothing,Bool}
    gamma::Union{AbstractFloat,Nothing,Bool}
    phi::Union{AbstractFloat,Nothing,Bool}
end

function process_parameters(y, model, damped, alpha, beta, gamma, phi, additive_only,
    lambda, lower, upper, opt_crit, nmse, bounds, ic, na_action_type)
    # Match the input arguments to allowed values
    opt_crit = match_arg(opt_crit, ["lik", "amse", "mse", "sigma", "mae"])
    bounds = match_arg(bounds, ["both", "usual", "admissible"])
    ic = match_arg(ic, ["aicc", "aic", "bic"])
    na_action_type = match_arg(na_action_type, ["na_contiguous", "na_interp", "na_fail"])

    # Remove missing values near the ends
    ny = length(y)
    y = na_action(y, na_action_type)

    if ny != length(y) && na_action_type == "na_contiguous"
        @warn "Missing values encountered. Using longest contiguous portion of time series"
        ny = length(y)
    end

    orig_y = y

    # Check if model is of type ETS and lambda is not set
    if typeof(model) == ETS && isnothing(lambda)
        lambda = model.lambda
    end

    # Apply Box-Cox transformation if lambda is provided
    if !isnothing(lambda)
        y, lambda = box_cox(y, length(y), lambda=lambda)  # Assuming m is length of y
        additive_only = true
    end

    # Check nmse range
    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse out of range"))
    end

    # Check if the bounds are valid
    if any(x -> x < 0, upper .- lower)
        throw(ArgumentError("Lower limits must be less than upper limits"))
    end

    # Return the necessary parameters
    return orig_y, y, lambda, damped, alpha, beta, gamma, phi, additive_only, opt_crit, bounds, ic, na_action_type, ny
end

function ets_refit(y::AbstractArray, m::Int, model::ETS; biasadj::Bool=false, use_initial_values::Bool=false, kwargs...)
    # Extract model parameters
    alpha = max(model.par["alpha"], 1e-10)
    beta = model.par["beta"]
    gamma = model.par["gamma"]
    phi = model.par["phi"]

    modelcomponents = string(model.components[1], model.components[2], model.components[3])
    damped = model.components[4]

    if use_initial_values
        errortype = string(modelcomponents[1])
        trendtype = string(modelcomponents[2])
        seasontype = string(modelcomponents[3])

        initstates = Vector(model.initstate[1, :])

        e = calculate_residuals(y, m, initstates, errortype, trendtype, seasontype, damped, alpha, beta, gamma, phi, nmse)

        lik = e["likelihood"]

        np = length(model.par) + 1
        loglik = -0.5 * lik
        aic = lik + 2 * np
        bic = lik + log(length(y)) * np
        aicc = model.aic + 2 * np * (np + 1) / (length(y) - np - 1)
        amse = e["amse"]
        mse = amse[1]
        amse = mean(amse)

        states = e["state"]
        if errortype == "A"
            fitted = y .- e["errors"]
        else
            fitted = y ./ (1 .+ e["errors"])
        end
        residuals = e["errors"]
        sigma2 = sum(residuals .^ 2) / (length(y) - np)
        x = y

        if biasadj
            model.fitted = invboxcox(model.fitted, model.lambda, biasadj, var(model.residuals))
        end
        model = EtsModel(
            fitted,
            residuals,
            model.components,
            x,
            model.par,
            loglik,
            model.initstate,
            states,
            model.SSE,
            sigma2,
            m,
            model.lambda,
            biasadj,
            aic,
            bic,
            aicc,
            mse,
            amse,
            model.fit,
            model.method)

        return model
    else
        model = modelcomponents
        @info "Model is being refit with current smoothing parameters but initial states are being re-estimated.\nSet 'use_initial_values=true' if you want to re-use existing initial values."
        return EtsRefit(modelcomponents, alpha, beta, gamma, phi)
    end
end

function validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    # Extract model components
    errortype = string(model[1])
    trendtype = string(model[2])
    seasontype = string(model[3])

    # Validate error, trend, and season types
    if !(errortype in ["M", "A", "Z"])
        throw(ArgumentError("Invalid error type"))
    end
    if !(trendtype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid trend type"))
    end
    if !(seasontype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid season type"))
    end

    # Adjust seasonality type based on the frequency `m` and length of `y`
    if m < 1 || length(y) <= m
        seasontype = "N"
    end
    if m == 1
        if seasontype in ["A", "M"]
            throw(ArgumentError("Nonseasonal data"))
        else
            seasontype = "N"
        end
    end
    if m > 24
        if seasontype in ["A", "M"]
            throw(ArgumentError("Frequency too high"))
        elseif seasontype == "Z"
            @warn "I can't handle data with frequency greater than 24. Seasonality will be ignored. Try stlf() if you need seasonal forecasts."
            seasontype = "N"
        end
    end

    # Check for forbidden model combinations based on the restrict flag
    if restrict
        if (errortype == "A" && (trendtype == "M" || seasontype == "M")) ||
           (errortype == "M" && trendtype == "M" && seasontype == "A") ||
           (additive_only && (errortype == "M" || trendtype == "M" || seasontype == "M"))
            throw(ArgumentError("Forbidden model combination"))
        end
    end

    # Check for non-positive data when error type is multiplicative
    data_positive = minimum(y) > 0
    if !data_positive && errortype == "M"
        throw(ArgumentError("Inappropriate model for data with negative or zero values"))
    end

    # Check if damped is provided and validate the model combination
    if !isnothing(damped)
        if damped && trendtype == "N"
            throw(ArgumentError("Forbidden model combination: Damped trend with no trend component"))
        end
    end

    # Calculate number of parameters (npars) based on model components
    n = length(y)
    npars = 2 # alpha + l0

    if trendtype in ["A", "M"]
        npars += 2 # beta + b0
    end
    if seasontype in ["A", "M"]
        npars += m # gamma + s
    end
    if !isnothing(damped)
        npars += damped ? 1 : 0
    end

    return errortype, trendtype, seasontype, npars, data_positive
end

function fit_small_dataset(y, m, alpha, beta, gamma, phi, trendtype, seasontype, lambda, biasadj)

    # Seasonal models handling
    if seasontype in ["A", "M"]
        try
            fit = holt_winters_conventional(y, m, alpha=alpha, beta=beta, gamma=gamma,
                seasonal=(seasontype == "M" ? "multiplicative" : "additive"), exponential=(trendtype == 'M'),
                phi=phi, lambda=lambda, biasadj=biasadj, warnings=false)
            return fit
        catch e
            @warn "Seasonal component could not be estimated: $e"
        end
    end

    # Trend models handling
    if trendtype in ["A", "M"]
        try
            fit = holt_winters_conventional(y, m, alpha=alpha, beta=beta, gamma=false,
                seasonal="additive", exponential=(trendtype == 'M'), phi=phi, lambda=lambda,
                biasadj=biasadj, warnings=false)
            return fit
        catch e
            @warn "Trend component could not be estimated: $e"
            return nothing
        end
    end

    # Non-trend and non-seasonal models
    if trendtype == "N" && seasontype == "N"
        try
            fit = holt_winters_conventional(y, m, alpha=alpha, beta=false, gamma=false,
                seasonal="additive", exponential=false, phi=nothing, lambda=lambda,
                biasadj=biasadj, warnings=false)
            return fit
        catch e
            @warn "Model without trend and seasonality could not be estimated: $e"
            return nothing
        end
    end

    # If none of the above fit, try Holt and SES models and select the best one
    fit1 = try
        holt_winters_conventional(y, m, alpha=alpha, beta=beta, gamma=false, seasonal="additive",
            exponential=(trendtype == 'M'), phi=phi, lambda=lambda, biasadj=biasadj, warnings=false)
    catch e
        nothing
    end

    fit2 = try
        holt_winters_conventional(y, m, alpha=alpha, beta=false, gamma=false, seasonal="additive",
            exponential=(trendtype == 'M'), phi=phi, lambda=lambda, biasadj=biasadj, warnings=false)
    catch e
        nothing
    end

    # Compare fits and return the best one
    fit = isnothing(fit1) ? fit2 : (isnothing(fit2) ? fit1 : (fit1.sigma2 < fit2.sigma2 ? fit1 : fit2))

    if isnothing(fit)
        error("Unable to estimate a model.")
    end

    return fit
end

function get_ic(fit, ic)
    if ic == "aic"
        return fit["aic"]
    elseif ic == "bic"
        return fit["bic"]
    elseif ic == "aicc"
        return fit["aicc"]
    else
        return Inf
    end
end

function generate_ets_grid_fixed(
    errortype::String,
    trendtype::String,
    seasontype::String,
    allow_multiplicative_trend::Bool,
    restrict::Bool,
    additive_only::Bool,
    data_positive::Bool,
    damped::Union{Bool,Nothing}
)
    errortype_vals = (errortype == "Z") ? ["A", "M"] : [errortype]
    trendtype_vals = if trendtype == "Z"
        allow_multiplicative_trend ? ["N", "A", "M"] : ["N", "A"]
    else
        [trendtype]
    end
    seasontype_vals = (seasontype == "Z") ? ["N", "A", "M"] : [seasontype]
    damped_vals = isnothing(damped) ? [true, false] : [damped]

    grid = []

    for e in errortype_vals
        for t in trendtype_vals
            for s in seasontype_vals
                for d in damped_vals
                    if t == "N" && d
                        continue
                    end
                    if restrict
                        if e == "A" && (t == "M" || s == "M")
                            continue
                        end
                        if e == "M" && t == "M" && s == "A"
                            continue
                        end
                        if additive_only && (e == "M" || t == "M" || s == "M")
                            continue
                        end
                    end
                    if !data_positive && e == "M"
                        continue
                    end
                    push!(grid, (e, t, s, d))
                end
            end
        end
    end

    return grid
end

function fit_ets_models(grid, y, m, alpha, beta, gamma, phi, lower, upper, opt_crit, nmse, bounds, ic, opt_method, iterations; kwargs...)
    best_ic = Inf
    best_model = nothing
    best_params = ()

    for combo in grid
        et, t, s, d = combo 
        the_fit_model = etsmodel(y, m, et, t, s, d, alpha, beta, gamma, phi, lower, upper, opt_crit, nmse, bounds, opt_method, iterations; kwargs...)
        if fit_ic < best_ic
            best_ic = fit_ic
            best_model = the_fit_model
            best_params = combo
        end
        # try
        #     the_fit_model = etsmodel(y, m, et, t, s, d, alpha, beta, gamma, phi, lower, upper, opt_crit, nmse, bounds, opt_method, iterations; kwargs...)
            
        #     fit_ic = get_ic(the_fit_model, ic)

        #     if fit_ic < best_ic
        #         best_ic = fit_ic
        #         best_model = the_fit_model
        #         best_params = combo
        #     end
        # catch e
        #     @warn "Error fitting model with combination: $combo"
        #     continue
        # end
    end
    return Dict("best_model" => best_model, "best_params" => best_params, "best_ic" => best_ic)
end

function fit_best_ets_model(y, m, errortype, trendtype, seasontype, damped, alpha, beta, gamma, phi,
    lower, upper, opt_crit, nmse, bounds, ic, data_positive; restrict=true, additive_only=false, 
    allow_multiplicative_trend=true, opt_method = NelderMead(), iterations = 2000, kwargs...)

    grid = generate_ets_grid_fixed(errortype, trendtype, seasontype, allow_multiplicative_trend, restrict, 
    additive_only,  data_positive,  damped)

    result = fit_ets_models(grid, y, m, alpha, beta, gamma, phi, lower, upper, opt_crit, nmse, 
    bounds, ic,  opt_method, iterations; kwargs...)

    best_model = result["best_model"]
    best_params = result["best_params"]
    best_ic = result["best_ic"]
    result = nothing
    println("best_params: ", best_params)
    best_e, best_t, best_s, best_d = best_params
    
    if best_ic == Inf
        throw(ModelFitError("No model able to be fitted"))
    end
    
    method = "ETS($(best_e),$(best_t)$(best_d ? "d" : ""),$(best_s))"
    components = [best_e, best_t, best_s, best_d]
    return Dict("model" => best_model, "method" => method, "components" => components)
end

function get_optimization_method(opt_method::String)
    opt_methods = Dict(
        "Nelder-Mead" => Optim.NelderMead(),
        "BFGS" => Optim.BFGS(),
        "CG" => Optim.ConjugateGradient(),
        "L-BFGS-B" => Optim.LBFGS(),
        "SANN" => Optim.SimulatedAnnealing(),
        "Brent" => Optim.Brent()
    )

    if haskey(opt_methods, opt_method)
        return opt_methods[opt_method]
    else
        error("Unsupported optimization method: $opt_method")
    end
end


function ets_base_model(y::AbstractArray, m::Int, model; damped::Union{Bool,Nothing}=nothing,
    alpha::Union{Float64,Bool,Nothing}=nothing, beta::Union{Float64,Bool,Nothing}=nothing,
    gamma::Union{Float64,Bool,Nothing}=nothing, phi::Union{Float64,Bool,Nothing}=nothing,
    additive_only::Bool=false,
    lambda::Union{Float64,Bool,Nothing, String}=nothing, biasadj::Bool=false,
    lower::AbstractArray=[0.0001, 0.0001, 0.0001, 0.8], upper::AbstractArray=[0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::String="lik", nmse::Int=3, bounds::String="both", ic::String="aicc", restrict::Bool=true,
    allow_multiplicative_trend::Bool=false, use_initial_values::Bool=false, na_action_type::String="na_contiguous",
    opt_method::String="Nelder-Mead", iterations::Int=2000, kwargs...)

    opt_method = match_arg(opt_method, ["Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN", "Brent"])

    opt_method = get_optimization_method(opt_method)

    orig_y, y, lambda, damped, alpha, beta, gamma, phi, additive_only, opt_crit, bounds, ic,
    na_action_type, ny = process_parameters(y, model, damped, alpha, beta, gamma, phi, additive_only, lambda, lower, upper,
        opt_crit, nmse, bounds, ic, na_action_type)

    if typeof(model) == ETS
        model = ets_refit(y, m, model, biasadj=biasadj, use_initial_values=use_initial_values; kwargs...)
        if typeof(model) == ETS
            return model
        end
    end

    errortype, trendtype, seasontype, npars, data_positive = validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    # Produce something non-optimized for tiny data sets
    if ny <= npars + 4

        if !isnothing(damped) && damped
            @warn "Not enough data to use damping"
        end

        return fit_small_dataset(orig_y, alpha, beta, gamma, phi, trendtype, seasontype, lambda, biasadj)
    end

    model = fit_best_ets_model(y, m, errortype, trendtype, seasontype, damped, alpha, beta, gamma, phi, lower, upper, opt_crit, nmse, bounds, ic, data_positive,
        restrict=additive_only, additive_only=additive_only, allow_multiplicative_trend=allow_multiplicative_trend, opt_method=opt_method, iterations=iterations; kwargs...)

    method = model["method"]
    components = model["components"]
    model = model["model"]
    np = length(model["par"])
    sigma2 = sum(skipmissing(model["residuals"] .^ 2)) / (ny - np)
    SSE = NaN

    if !isnothing(lambda)
        model["fitted"] = inv_box_cox(model["fitted"], lambda=lambda, biasadj=biasadj, fvar=sigma2)
    end

    initstates = transpose(model["states"])
    initstates = initstates[1, :]

    model = EtsModel(
        model["fitted"],
        model["residuals"],
        components,
        orig_y,
        model["par"],
        model["loglik"],
        initstates,
        transpose(model["states"]),
         ["cff"], # model["state_names"],
        SSE,
        sigma2,
        m,
        lambda,
        biasadj,
        model["aic"],
        model["bic"],
        model["aicc"],
        model["mse"],
        model["amse"],
        model["fit"],
        method,
    )
    return model
end

function ets(y::AbstractArray, 
    m::Int, 
    model::Union{String,ETS};
    damped::Union{Bool,Nothing}=nothing,
    alpha::Union{Float64,Bool,Nothing}=nothing,
    beta::Union{Float64,Bool,Nothing}=nothing,
    gamma::Union{Float64,Bool,Nothing}=nothing,
    phi::Union{Float64,Bool,Nothing}=nothing,
    additive_only::Bool=false,
    lambda::Union{Float64,Bool,Nothing, String}=nothing,
    biasadj::Bool=false,
    lower::AbstractArray=[0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray=[0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::String="lik",
    nmse::Int=3,
    bounds::String="both",
    ic::String="aicc",
    restrict::Bool=true,
    allow_multiplicative_trend::Bool=false,
    use_initial_values::Bool=false,
    na_action_type::String="na_contiguous",
    opt_method::String="Nelder-Mead",
    iterations::Int=2000,
    kwargs...)

    if model == "ZZZ" && is_constant(y)
        return ses(y, alpha=0.99999, initial="simple")
    end

    out = ets_base_model(y, m, model, damped=damped, alpha=alpha, beta=beta, gamma=gamma, phi=phi,
        additive_only=additive_only, lambda=lambda, biasadj=biasadj, lower=lower, upper=upper,
        opt_crit=opt_crit, nmse=nmse, bounds=bounds, ic=ic, restrict=restrict,
        allow_multiplicative_trend=allow_multiplicative_trend,
        use_initial_values=use_initial_values,
        na_action_type=na_action_type,
        opt_method=opt_method, iterations=iterations, kwargs...)

    return out
end
