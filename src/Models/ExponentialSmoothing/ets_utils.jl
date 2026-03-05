function create_params(
    optimized_params::AbstractArray,
    opt_alpha::Bool,
    opt_beta::Bool,
    opt_gamma::Bool,
    opt_phi::Bool,
)
    j = 1
    pars = Dict()

    if opt_alpha
        pars["alpha"] = optimized_params[j]
        j += 1
    end

    if opt_beta
        pars["beta"] = optimized_params[j]
        j += 1
    end

    if opt_gamma
        pars["gamma"] = optimized_params[j]
        j += 1
    end

    if opt_phi
        pars["phi"] = optimized_params[j]
        j += 1
    end

    result = optimized_params[j:end]
    return merge(pars, Dict("initstate" => result))
end


function objective_fun(
    par,
    y,
    nstate,
    errortype::Int,
    trendtype::Int,
    seasontype::Int,
    damped,
    lower,
    upper,
    opt_crit::Int,
    nmse,
    bounds::Int,
    m,
    init_alpha,
    init_beta,
    init_gamma,
    init_phi,
    opt_alpha,
    opt_beta,
    opt_gamma,
    opt_phi,
    workspace::ETSWorkspace,
)

    j = 1

    alpha = opt_alpha ? par[j] : init_alpha
    j += opt_alpha

    beta = nothing
    if trendtype != 0
        beta = opt_beta ? par[j] : init_beta
        j += opt_beta
    end

    gamma = nothing
    if seasontype != 0
        gamma = opt_gamma ? par[j] : init_gamma
        j += opt_gamma
    end

    phi = nothing
    if damped
        phi = opt_phi ? par[j] : init_phi
        j += opt_phi
    end

    if isnan(alpha)
        throw(ArgumentError("alpha must be numeric"))
    elseif !isnothing(beta) && isnan(beta)
        throw(ArgumentError("beta must be numeric"))
    elseif !isnothing(gamma) && isnan(gamma)
        throw(ArgumentError("gamma must be numeric"))
    elseif !isnothing(phi) && isnan(phi)
        throw(ArgumentError("phi must be numeric"))
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        return Inf
    end

    init_state = view(par, (length(par)-nstate+1):length(par))
    init_state_eval = init_state
    trend_slots = trendtype == 0 ? 0 : 1

    if seasontype != 0
        workspace_init = workspace.init_state
        @inbounds copyto!(workspace_init, 1, init_state, 1, nstate)
        seasonal_start = 2 + trend_slots
        seasonal_sum = sum(view(workspace_init, seasonal_start:nstate))
        workspace_init[nstate+1] = (seasontype == 2 ? m : 0.0) - seasonal_sum
        init_state_eval = view(workspace_init, 1:(nstate+1))
    end

    if seasontype == 2
        if minimum(view(init_state_eval, (2+trend_slots):length(init_state_eval))) < 0.0
            return Inf
        end
    end

    lik, _ = calculate_residuals!(
        workspace,
        y,
        m,
        init_state_eval,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )

    if isnan(lik) || abs(lik + 99999) < 1e-7
        lik = Inf
    elseif lik < -1e10
        # Clamp near-perfect fits to a finite floor.
        lik = -1e10
    end

    if opt_crit == OPT_CRIT_LIK
        return lik
    elseif opt_crit == OPT_CRIT_MSE
        return workspace.amse[1]
    elseif opt_crit == OPT_CRIT_AMSE
        return sum(view(workspace.amse, 1:nmse)) / nmse
    elseif opt_crit == OPT_CRIT_SIGMA
        return sum(abs2, view(workspace.e, 1:length(y))) / length(y)
    elseif opt_crit == OPT_CRIT_MAE
        return sum(abs, view(workspace.e, 1:length(y))) / length(y)
    else
        throw(ArgumentError("Unknown optimization criterion"))
    end
end

function optim_ets_base(
    opt_params,
    y,
    nstate,
    errortype,
    trendtype,
    seasontype,
    damped,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    m,
    initial_params,
    options)

    init_alpha = initial_params.alpha
    init_beta = initial_params.beta
    init_gamma = initial_params.gamma
    init_phi = initial_params.phi
    opt_alpha = !isnan(init_alpha)
    opt_beta = !isnan(init_beta)
    opt_gamma = !isnan(init_gamma)
    opt_phi = !isnan(init_phi)
    errortype_code = ets_model_type_code(errortype)
    trendtype_code = ets_model_type_code(trendtype)
    seasontype_code = ets_model_type_code(seasontype)
    opt_crit_int = opt_crit_code(opt_crit)
    bounds_int = bounds_code(bounds)
    max_state_len = nstate + (seasontype_code != 0 ? 1 : 0)
    workspace = ETSWorkspace(length(y), m, nmse, max_state_len)

    result = nelder_mead(par -> objective_fun(
            par,
            y,
            nstate,
            errortype_code,
            trendtype_code,
            seasontype_code,
            damped,
            lower,
            upper,
            opt_crit_int,
            nmse,
            bounds_int,
            m,
            init_alpha,
            init_beta,
            init_gamma,
            init_phi,
            opt_alpha,
            opt_beta,
            opt_gamma,
            opt_phi,
            workspace,
        ), opt_params,
        options)

    optimized_params = result.x_opt
    optimized_value = result.f_opt
    number_of_iterations = result.fncount

    optimized_params =
        create_params(optimized_params, opt_alpha, opt_beta, opt_gamma, opt_phi)

    return Dict(
        "optimized_params" => optimized_params,
        "optimized_value" => optimized_value,
        "number_of_iterations" => number_of_iterations,
    )
end

function etsmodel(
    y::Vector{Float64},
    m::Int,
    errortype::String,
    trendtype::String,
    seasontype::String,
    damped::Bool,
    alpha::Union{Float64,Nothing,Bool},
    beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool},
    phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64},
    upper::Vector{Float64},
    opt_crit::Symbol,
    nmse::Int,
    bounds::Symbol,
    options::NelderMeadOptions)

    if seasontype == "N"
        m = 1
    end

    if isa(alpha, Bool)
        if alpha
            alpha = 1.0 - 1e-10
        else
            alpha = 0.0
        end
    end

    if isa(beta, Bool)
        if beta
            beta = 1.0
        else
            beta = 0.0
        end
    end

    if isa(gamma, Bool)
        if gamma
            gamma = 1.0
        else
            gamma = 0.0
        end
    end

    if isa(phi, Bool)
        if phi
            phi = 1.0
        else
            phi = 0.0
        end
    end

    if !(isnothing(alpha) || (!isnothing(alpha) && isnan(alpha)))
        upper[2] = min(alpha, upper[2])
        upper[3] = min(1 - alpha, upper[3])
    end


    if !(isnothing(beta) || (!isnothing(beta) && isnan(beta)))
        lower[1] = max(beta, lower[1])
    end

    if !(isnothing(gamma) || (!isnothing(gamma) && isnan(gamma)))
        upper[1] = min(1 - gamma, upper[1])
    end

    par = initparam(
        alpha,
        beta,
        gamma,
        phi,
        trendtype,
        seasontype,
        damped,
        lower,
        upper,
        m,
        bounds,
        nothing_as_nan = true,
    )

    if !isnan(par.alpha)
        alpha = par.alpha
    end

    if !isnan(par.beta)
        beta = par.beta
    end

    if !isnan(par.gamma)
        gamma = par.gamma
    end

    if !isnan(par.phi)
        phi = par.phi
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        damped_str = damped ? "d" : ""
        throw(ArgumentError("For model `ETS($(errortype),\
         $(trendtype)$(damped_str), $(seasontype))` \
         parameters are out of range!"))
    end

    init_state = initialize_states(y, m, trendtype, seasontype)
    nstate = length(init_state)
    initial_params = par
    par = [par.alpha, par.beta, par.gamma, par.phi]
    par = dropmissing(par)
    par = vcat(par, init_state)

    lower = vcat(lower, fill(-Inf, nstate))
    upper = vcat(upper, fill(Inf, nstate))

    np = length(par)
    if np >= length(y) - 1
        return Dict(
            :aic => Inf,
            :bic => Inf,
            :aicc => Inf,
            :mse => Inf,
            :amse => Inf,
            :fit => nothing,
            :par => par,
            :states => init_state,
        )
    end

    init_state = nothing

    optimized_fit = optim_ets_base(
        par,
        y,
        nstate,
        errortype,
        trendtype,
        seasontype,
        damped,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        m,
        initial_params,
        options)

    fit_par = optimized_fit["optimized_params"]

    states = fit_par["initstate"]

    if seasontype != "N"
        states =
            vcat(states, m * (seasontype == "M") - sum(states[(2+(trendtype!="N")):nstate]))
    end

    if !isnan(initial_params.alpha)
        alpha = fit_par["alpha"]
    end
    if !isnan(initial_params.beta)
        beta = fit_par["beta"]
    end
    if !isnan(initial_params.gamma)
        gamma = fit_par["gamma"]
    end
    if !isnan(initial_params.phi)
        phi = fit_par["phi"]
    end

    lik, amse, e, states = calculate_residuals(
        y,
        m,
        states,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        nmse,
    )

    np += 1
    ny = length(y)
    aic = lik + 2 * np
    bic = lik + log(ny) * np
    aicc = aic + 2 * np * (np + 1) / (ny - np - 1)
    mse = amse[1]
    amse = mean(amse)

    if errortype == "A"
        fits = y .- e
    else
        fits = y ./ (1 .+ e)
    end

    out = Dict(
        "loglik" => -0.5 * lik,
        "aic" => aic,
        "bic" => bic,
        "aicc" => aicc,
        "mse" => mse,
        "amse" => amse,
        "fit" => optimized_fit,
        "residuals" => e,
        "fitted" => fits,
        "states" => states,
        "par" => fit_par,
    )
    return out
end

function process_parameters(
    y,
    m,
    model,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    lambda,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    missing_method::MissingMethod,
)

    opt_crit = _check_arg(opt_crit, (:lik, :amse, :mse, :sigma, :mae), "opt_crit")
    bounds = _check_arg(bounds, (:both, :usual, :admissible), "bounds")
    ic = _check_arg(ic, (:aicc, :aic, :bic), "ic")

    ny = length(y)
    y = handle_missing(y, missing_method; m=m)

    if ny != length(y) && missing_method isa Contiguous
        @warn "Missing values encountered. Using longest contiguous portion of time series"
        ny = length(y)
    end

    orig_y = y

    if typeof(model) == ETS && isnothing(lambda)
        lambda = model.lambda
    end

    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda = lambda)
        additive_only = true
    end

    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse out of range"))
    end

    if any(x -> x < 0, upper .- lower)
        throw(ArgumentError("Lower limits must be less than upper limits"))
    end

    return orig_y,
    y,
    lambda,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    opt_crit,
    bounds,
    ic,
    missing_method,
    ny
end

function ets_refit(
    y::AbstractArray,
    m::Int,
    model::ETS;
    biasadj::Bool = false,
    use_initial_values::Bool = false,
    nmse::Int = 3,
    kwargs...,
)
    alpha = max(model.par["alpha"], 1e-10)
    beta = get(model.par, "beta", nothing)
    gamma = get(model.par, "gamma", nothing)
    phi = get(model.par, "phi", nothing)

    modelcomponents = string(model.components[1], model.components[2], model.components[3])
    damped = parse(Bool, model.components[4])

    if use_initial_values
        errortype = string(modelcomponents[1])
        trendtype = string(modelcomponents[2])
        seasontype = string(modelcomponents[3])

        # Handle both 1D and 2D initstate arrays
        if ndims(model.initstate) == 1
            initstates = Vector(model.initstate)
        else
            initstates = Vector(model.initstate[1, :])
        end

        lik, amse, e, states = calculate_residuals(
            y,
            m,
            initstates,
            errortype,
            trendtype,
            seasontype,
            damped,
            alpha,
            beta,
            gamma,
            phi,
            nmse,
        )

        np = length(model.par) + 1
        loglik = -0.5 * lik
        aic = lik + 2 * np
        bic = lik + log(length(y)) * np
        aicc = model.aic + 2 * np * (np + 1) / (length(y) - np - 1)
        mse = amse[1]
        amse = mean(amse)

        if errortype == "A"
            fitted = y .- e
        else
            fitted = y ./ (1 .+ e)
        end

        sigma2 = sum(e .^ 2) / (length(y) - np)
        x = y

        if biasadj
            model.fitted = inv_box_cox(model.fitted, lambda=model.lambda, biasadj=biasadj, fvar=var(model.residuals))
        end
        model = EtsModel(
            fitted,
            e,
            model.components,
            x,
            model.par,
            loglik,
            model.initstate,
            states,
            String[],
            model.sse,
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
            model.method,
        )

        return model
    else
        model = modelcomponents
        @info "Model is being refit with current smoothing parameters but initial states are being re-estimated.\nSet 'use_initial_values=true' if you want to re-use existing initial values."
        return EtsRefit(modelcomponents, alpha, beta, gamma, phi)
    end
end

function validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    errortype = string(model[1])
    trendtype = string(model[2])
    seasontype = string(model[3])

    if !(errortype in ["M", "A", "Z"])
        throw(ArgumentError("Invalid error type"))
    end
    if !(trendtype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid trend type"))
    end
    if !(seasontype in ["N", "A", "M", "Z"])
        throw(ArgumentError("Invalid season type"))
    end

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

    if restrict
        if (errortype == "A" && (trendtype == "M" || seasontype == "M")) ||
           (errortype == "M" && trendtype == "M" && seasontype == "A") ||
           (additive_only && (errortype == "M" || trendtype == "M" || seasontype == "M"))
            throw(ArgumentError("Forbidden model combination"))
        end
    end

    data_positive = minimum(y) > 0
    if !data_positive
        if errortype == "M"
            throw(ArgumentError(
                "Multiplicative error models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
        if trendtype == "M"
            throw(ArgumentError(
                "Multiplicative trend models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
        if seasontype == "M"
            throw(ArgumentError(
                "Multiplicative seasonal models require strictly positive data. " *
                "Data contains zero or negative values."
            ))
        end
    end

    if !isnothing(damped)
        if damped && trendtype == "N"
            throw(
                ArgumentError(
                    "Forbidden model combination: Damped trend with no trend component",
                ),
            )
        end
    end

    n = length(y)
    npars = 2

    if trendtype in ["A", "M"]
        npars += 2
    end
    if seasontype in ["A", "M"]
        npars += m
    end
    if !isnothing(damped)
        npars += damped ? 1 : 0
    end

    return errortype, trendtype, seasontype, npars, data_positive
end

function fit_small_dataset(
    y,
    m,
    alpha,
    beta,
    gamma,
    phi,
    trendtype,
    seasontype,
    lambda,
    biasadj,
    options
)

    if seasontype in ["A", "M"]
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = beta,
                gamma = gamma,
                seasonal = (seasontype == "M" ? :multiplicative : :additive),
                exponential = (trendtype == "M"),
                phi = phi,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Seasonal component could not be estimated: $e"
        end
    end

    if trendtype in ["A", "M"]
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = beta,
                gamma = false,
                seasonal = :additive,
                exponential = (trendtype == "M"),
                phi = phi,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Trend component could not be estimated: $e"
            return nothing
        end
    end

    if trendtype == "N" && seasontype == "N"
        try
            fit = holt_winters_conventional(
                y,
                m,
                alpha = alpha,
                beta = false,
                gamma = false,
                seasonal = :additive,
                exponential = false,
                phi = nothing,
                lambda = lambda,
                biasadj = biasadj,
                warnings = false,
                options = options
            )
            return fit
        catch e
            @warn "Model without trend and seasonality could not be estimated: $e"
            return nothing
        end
    end

    fit1 = try
        holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = false,
            seasonal = :additive,
            exponential = (trendtype == "M"),
            phi = phi,
            lambda = lambda,
            biasadj = biasadj,
            warnings = false,
            options = options
        )
    catch e
        nothing
    end

    fit2 = try
        holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = false,
            gamma = false,
            seasonal = :additive,
            exponential = (trendtype == "M"),
            phi = phi,
            lambda = lambda,
            biasadj = biasadj,
            warnings = false,
            options = options
        )
    catch e
        nothing
    end

    fit =
        isnothing(fit1) ? fit2 :
        (isnothing(fit2) ? fit1 : (fit1.sigma2 < fit2.sigma2 ? fit1 : fit2))

    if isnothing(fit)
        error("Unable to estimate a model.")
    end

    return fit
end

function get_ic(fit, ic)
    if ic === :aic
        return fit["aic"]
    elseif ic === :bic
        return fit["bic"]
    elseif ic === :aicc
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
    damped::Union{Bool,Nothing},
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

function fit_ets_models(
    grid,
    y,
    m,
    alpha,
    beta,
    gamma,
    phi,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    options)

    best_ic = Inf
    best_model = nothing
    best_params = ()
    lower_scratch = similar(lower)
    upper_scratch = similar(upper)

    for combo in grid
        et, t, s, d = combo
        try
            copyto!(lower_scratch, lower)
            copyto!(upper_scratch, upper)
            the_fit_model = etsmodel(
                y,
                m,
                et,
                t,
                s,
                d,
                alpha,
                beta,
                gamma,
                phi,
                lower_scratch,
                upper_scratch,
                opt_crit,
                nmse,
                bounds,
                options)
            fit_ic = get_ic(the_fit_model, ic)
            if fit_ic < best_ic
                best_ic = fit_ic
                best_model = the_fit_model
                best_params = combo
            end
        catch e
            @warn "Error fitting model with combination: $combo" exception=(e, catch_backtrace())
            continue
        end
    end
    return Dict(
        "best_model" => best_model,
        "best_params" => best_params,
        "best_ic" => best_ic,
    )
end

function fit_best_ets_model(
    y,
    m,
    errortype,
    trendtype,
    seasontype,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    lower,
    upper,
    opt_crit,
    nmse,
    bounds,
    ic,
    data_positive;
    restrict = true,
    additive_only = false,
    allow_multiplicative_trend = true,
    options)

    grid = generate_ets_grid_fixed(
        errortype,
        trendtype,
        seasontype,
        allow_multiplicative_trend,
        restrict,
        additive_only,
        data_positive,
        damped,
    )

    result = fit_ets_models(
        grid,
        y,
        m,
        alpha,
        beta,
        gamma,
        phi,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        options)

    best_model = result["best_model"]
    best_params = result["best_params"]
    best_ic = result["best_ic"]
    result = nothing

    best_e, best_t, best_s, best_d = best_params

    if best_ic == Inf
        throw(ModelFitError("No model able to be fitted"))
    end

    method = "ETS($(best_e),$(best_t)$(best_d ? "d" : ""),$(best_s))"
    components = [best_e, best_t, best_s, string(best_d)]
    return Dict("model" => best_model, "method" => method, "components" => components)
end


function ets_base_model(
    y::AbstractArray,
    m::Int,
    model;
    damped::Union{Bool,Nothing} = nothing,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    additive_only::Bool = false,
    lambda::Union{Float64,Bool,Nothing,Symbol} = nothing,
    biasadj::Bool = false,
    lower::AbstractArray = [0.0001, 0.0001, 0.0001, 0.8],
    upper::AbstractArray = [0.9999, 0.9999, 0.9999, 0.98],
    opt_crit::Symbol = :lik,
    nmse::Int = 3,
    bounds::Symbol = :both,
    ic::Symbol = :aicc,
    restrict::Bool = true,
    allow_multiplicative_trend::Bool = false,
    use_initial_values::Bool = false,
    missing_method::MissingMethod = Contiguous(),
    options::NelderMeadOptions,
)

    orig_y,
    y,
    lambda,
    damped,
    alpha,
    beta,
    gamma,
    phi,
    additive_only,
    opt_crit,
    bounds,
    ic,
    missing_method,
    ny = process_parameters(
        y,
        m,
        model,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        additive_only,
        lambda,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        missing_method,
    )

    if model isa ETS
        refit_result = ets_refit(
            y,
            m,
            model,
            biasadj = biasadj,
            use_initial_values = use_initial_values,
            nmse = nmse,
        )
        if refit_result isa ETS
            return refit_result
        end
        # EtsRefit was returned, extract model string for further processing
        model = refit_result.model
    end

    errortype, trendtype, seasontype, npars, data_positive =
        validate_and_set_model_params(model, y, m, damped, restrict, additive_only)

    if ny <= npars + 4

        if !isnothing(damped) && damped
            @warn "Not enough data to use damping"
        end

        return fit_small_dataset(
            orig_y,
            m,
            alpha,
            beta,
            gamma,
            phi,
            trendtype,
            seasontype,
            lambda,
            biasadj,
            options
        )
    end

    model = fit_best_ets_model(
        Float64.(y),
        m,
        errortype,
        trendtype,
        seasontype,
        damped,
        alpha,
        beta,
        gamma,
        phi,
        lower,
        upper,
        opt_crit,
        nmse,
        bounds,
        ic,
        data_positive,
        restrict = restrict,
        additive_only = additive_only,
        allow_multiplicative_trend = allow_multiplicative_trend,
        options = options)

    method = model["method"]
    components = model["components"]
    model = model["model"]
    np = length(model["par"])
    sigma2 = sum(skipmissing(model["residuals"] .^ 2)) / (ny - np)
    sse = NaN

    if !isnothing(lambda)
        model["fitted"] =
            inv_box_cox(model["fitted"], lambda = lambda, biasadj = biasadj, fvar = sigma2)
    end

    initstates = model["states"][1, :]

    model = EtsModel(
        model["fitted"],
        model["residuals"],
        components,
        orig_y,
        model["par"],
        model["loglik"],
        initstates,
        model["states"],
        ["cff"],
        sse,
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

function simulate_ets(
    object::ETS,
    nsim::Union{Int,Nothing} = nothing;
    future::Bool = true,
    bootstrap::Bool = false,
    innov::Union{Vector{Float64},Nothing} = nothing,
)

    x = object.x
    m = object.m
    states = object.states
    residuals = object.residuals
    sigma2 = object.sigma2
    components = object.components
    par = object.par
    lambda = object.lambda
    biasadj = object.biasadj

    nsim = isnothing(nsim) ? length(x) : nsim

    if !isnothing(innov)
        nsim = length(innov)
    end

    if !all(ismissing.(x))
        if isnothing(m)
            m = 1
        end
    else
        if nsim == 0
            nsim = 100
        end
        x = [10]
        future = false
    end

    initstate = future ? states[end, :] : states[rand(1:size(states, 1)), :]

    if bootstrap
        res = filter(!ismissing, residuals) .- mean(filter(!ismissing, residuals))
        e = rand(res, nsim)
    elseif isnothing(innov)
        e = rand(Normal(0, sqrt(sigma2)), nsim)
    elseif length(innov) == nsim
        e = innov
    else
        throw(ArgumentError("Length of innov must be equal to nsim"))
    end

    if components[1] == "M"
        e = max.(-1, e)
    end

    y = zeros(nsim)
    errors = ets_model_type_code(components[1])
    trend = ets_model_type_code(components[2])
    season = ets_model_type_code(components[3])
    alpha = check_component(par, "alpha")
    # beta = ifelse(trend == "N", 0.0, check_component(par, "beta"))
    # gamma = ifelse(season == "N", 0.0, check_component(par, "gamma"))
    # phi = ifelse(!components[4], 1.0, check_component(par, "phi"))
    beta = (trend == 0) ? 0.0 : check_component(par, "beta")
    gamma = (season == 0) ? 0.0 : check_component(par, "gamma")
    phi = parse(Bool, components[4]) ? check_component(par, "phi") : 1.0
    simulate_ets_base(
        initstate,
        m,
        errors,
        trend,
        season,
        alpha,
        beta,
        gamma,
        phi,
        nsim,
        y,
        e,
    )

    if abs(y[1] - (-99999.0)) < 1e-7
        error("Problem with multiplicative damped trend")
    end

    if !isnothing(lambda)
        y = inv_box_cox(y, lambda=lambda)
    end

    return y
end
