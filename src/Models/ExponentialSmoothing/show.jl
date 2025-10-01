"""
Custom show methods for ETS models to display output similar to R's forecast package.
"""

function Base.show(io::IO, model::EtsModel)
    # Extract model components
    error_type = model.components[1]
    trend_type = model.components[2]
    season_type = model.components[3]
    damped = Bool(model.components[4])

    # Model name
    model_str = "ETS($error_type,$trend_type"
    if damped && trend_type != "N"
        model_str *= "_d"
    end
    model_str *= ",$season_type)"

    println(io, model_str)
    println(io)

    # Box-Cox transformation
    if model.lambda !== nothing && model.lambda !== false
        println(io, "  Box-Cox transformation: lambda= ", round(model.lambda, digits=4))
        println(io)
    end

    # Smoothing parameters
    println(io, "  Smoothing parameters:")
    if haskey(model.par, "alpha")
        println(io, "    alpha = ", round(model.par["alpha"], digits=4))
    end
    if haskey(model.par, "beta")
        println(io, "    beta  = ", round(model.par["beta"], digits=4))
    end
    if haskey(model.par, "gamma")
        println(io, "    gamma = ", round(model.par["gamma"], digits=4))
    end
    if haskey(model.par, "phi")
        println(io, "    phi   = ", round(model.par["phi"], digits=4))
    end
    println(io)

    # Initial states
    println(io, "  Initial states:")
    state_vals = model.initstate

    # Level
    idx = 1
    if idx <= length(state_vals)
        println(io, "    l = ", round(state_vals[idx], digits=4))
        idx += 1
    end

    # Trend
    if trend_type != "N" && idx <= length(state_vals)
        println(io, "    b = ", round(state_vals[idx], digits=4))
        idx += 1
    end

    # Seasonal components
    if season_type != "N" && idx <= length(state_vals)
        seasonal = state_vals[idx:end]
        print(io, "    s = ")
        # Print first 6 seasonal values on first line
        n_print = min(6, length(seasonal))
        for i in 1:n_print
            print(io, round(seasonal[i], digits=4), " ")
        end
        println(io)

        # Print remaining seasonal values on second line (if any)
        if length(seasonal) > 6
            print(io, "           ")
            for i in 7:length(seasonal)
                print(io, round(seasonal[i], digits=4), " ")
            end
            println(io)
        end
    end
    println(io)

    # Sigma
    sigma = sqrt(model.sigma2)
    println(io, "  sigma:  ", round(sigma, digits=4))
    println(io)

    # Information criteria
    println(io, "      AIC      AICc       BIC")
    aic_str = lpad(round(model.aic, digits=4), 9)
    aicc_str = lpad(round(model.aicc, digits=4), 10)
    bic_str = lpad(round(model.bic, digits=4), 11)
    println(io, aic_str, aicc_str, bic_str)
end

function Base.show(io::IO, model::HoltWintersConventional)
    # Determine model type
    if haskey(model.par, "gamma")
        seasonal = get(model.par, "seasonal", "additive")
        damped_str = haskey(model.par, "phi") ? "_d" : ""
        model_str = "Holt-Winters$(damped_str) ($(seasonal) seasonality)"
    elseif haskey(model.par, "beta")
        damped_str = haskey(model.par, "phi") ? " damped" : ""
        model_str = "Holt's$(damped_str) linear trend method"
    else
        model_str = "Simple exponential smoothing"
    end

    println(io, model_str)
    println(io)

    # Box-Cox transformation
    if model.lambda !== nothing && model.lambda !== false
        println(io, "  Box-Cox transformation: lambda= ", round(model.lambda, digits=4))
        println(io)
    end

    # Smoothing parameters
    println(io, "  Smoothing parameters:")
    if haskey(model.par, "alpha")
        println(io, "    alpha = ", round(model.par["alpha"], digits=4))
    end
    if haskey(model.par, "beta")
        println(io, "    beta  = ", round(model.par["beta"], digits=4))
    end
    if haskey(model.par, "gamma")
        println(io, "    gamma = ", round(model.par["gamma"], digits=4))
    end
    if haskey(model.par, "phi")
        println(io, "    phi   = ", round(model.par["phi"], digits=4))
    end
    println(io)

    # Initial states
    println(io, "  Initial states:")
    state_vals = model.initstate
    idx = 1

    # Level
    if idx <= length(state_vals)
        println(io, "    l = ", round(state_vals[idx], digits=4))
        idx += 1
    end

    # Trend
    if haskey(model.par, "beta") && idx <= length(state_vals)
        println(io, "    b = ", round(state_vals[idx], digits=4))
        idx += 1
    end

    # Seasonal components
    if haskey(model.par, "gamma") && idx <= length(state_vals)
        seasonal = state_vals[idx:end]
        print(io, "    s = ")
        # Print first 6 seasonal values on first line
        n_print = min(6, length(seasonal))
        for i in 1:n_print
            print(io, round(seasonal[i], digits=4), " ")
        end
        println(io)

        # Print remaining seasonal values on second line (if any)
        if length(seasonal) > 6
            print(io, "           ")
            for i in 7:length(seasonal)
                print(io, round(seasonal[i], digits=4), " ")
            end
            println(io)
        end
    end
    println(io)

    # Sigma
    sigma = sqrt(model.sigma2)
    println(io, "  sigma:  ", round(sigma, digits=4))
end

function Base.show(io::IO, model::SES)
    println(io, "Simple Exponential Smoothing")
    println(io)

    # Box-Cox transformation
    if model.lambda !== nothing && model.lambda !== false
        println(io, "  Box-Cox transformation: lambda= ", round(model.lambda, digits=4))
        println(io)
    end

    # Smoothing parameters
    println(io, "  Smoothing parameters:")
    if haskey(model.par, "alpha")
        println(io, "    alpha = ", round(model.par["alpha"], digits=4))
    end
    println(io)

    # Initial states
    println(io, "  Initial states:")
    state_vals = model.initstate

    # Level
    if length(state_vals) >= 1
        println(io, "    l = ", round(state_vals[1], digits=4))
    end
    println(io)

    # Sigma
    sigma = sqrt(model.sigma2)
    println(io, "  sigma:  ", round(sigma, digits=4))
    println(io)

    # Information criteria
    println(io, "      AIC      AICc       BIC")
    aic_str = lpad(round(model.aic, digits=4), 9)
    aicc_str = lpad(round(model.aicc, digits=4), 10)
    bic_str = lpad(round(model.bic, digits=4), 11)
    println(io, aic_str, aicc_str, bic_str)
end

function Base.show(io::IO, model::Holt)
    println(io, model.method)
    println(io)

    # Box-Cox transformation
    if model.lambda !== nothing && model.lambda !== false
        println(io, "  Box-Cox transformation: lambda= ", round(model.lambda, digits=4))
        println(io)
    end

    # Smoothing parameters
    println(io, "  Smoothing parameters:")
    if haskey(model.par, "alpha")
        println(io, "    alpha = ", round(model.par["alpha"], digits=4))
    end
    if haskey(model.par, "beta")
        println(io, "    beta  = ", round(model.par["beta"], digits=4))
    end
    if haskey(model.par, "phi")
        println(io, "    phi   = ", round(model.par["phi"], digits=4))
    end
    println(io)

    # Initial states
    println(io, "  Initial states:")
    state_vals = model.initstate

    # Level
    if length(state_vals) >= 1
        println(io, "    l = ", round(state_vals[1], digits=4))
    end

    # Trend
    if length(state_vals) >= 2
        println(io, "    b = ", round(state_vals[2], digits=4))
    end
    println(io)

    # Sigma
    sigma = sqrt(model.sigma2)
    println(io, "  sigma:  ", round(sigma, digits=4))
    println(io)

    # Information criteria
    println(io, "      AIC      AICc       BIC")
    aic_str = lpad(round(model.aic, digits=4), 9)
    aicc_str = lpad(round(model.aicc, digits=4), 10)
    bic_str = lpad(round(model.bic, digits=4), 11)
    println(io, aic_str, aicc_str, bic_str)
end
