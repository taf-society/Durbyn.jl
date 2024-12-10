function etsmodel(y::Vector{Float64}, m::Int, errortype::String, trendtype::String, seasontype::String, damped::Bool,
    alpha::Union{Float64,Nothing,Bool}, beta::Union{Float64,Nothing,Bool}, gamma::Union{Float64,Nothing,Bool}, phi::Union{Float64,Nothing,Bool}, lower::Vector{Float64}, upper::Vector{Float64},
    opt_crit::String, nmse::Int, bounds::String, optim_method=NelderMead(), maxit::Int=2000; kwargs...)

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

    if !(isnothing(alpha) || (alpha !== nothing && isnan(alpha)))
        upper[2] = min(alpha, upper[2])
        upper[3] = min(1 - alpha, upper[3])
    end


    if !(isnothing(beta) || (beta !== nothing && isnan(beta)))
        lower[1] = max(beta, lower[1])
    end

    if !(isnothing(gamma) || (gamma !== nothing && isnan(gamma)))
        upper[1] = min(1 - gamma, upper[1])
    end

    par = initparam(alpha, beta, gamma, phi, trendtype, seasontype, damped, lower, upper, m, bounds, nothing_as_nan=true)

    # Update alpha, beta, gamma, and phi based on the 'par' values
    if !isnan(par["alpha"])
        alpha = par["alpha"]
    end

    if !isnan(par["beta"])
        beta = par["beta"]
    end

    if !isnan(par["gamma"])
        gamma = par["gamma"]
    end

    if !isnan(par["phi"])
        phi = par["phi"]
    end

    # Check if parameters are valid
    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        damped_str = damped ? "d" : ""  # Assign the result of ternary to a variable
        println("Model: ETS($(errortype), $(trendtype)$(damped_str), $(seasontype))")
        throw(ArgumentError("Parameters out of range"))
    end


    init_state = initialize_states(y, m, trendtype, seasontype)
    nstate = length(init_state)
    initial_params = par
    par = [par["alpha"], par["beta"], par["gamma"], par["phi"]]
    par = na_omit(par)
    par = vcat(par, init_state)

    lower = vcat(lower, fill(-Inf, nstate))
    upper = vcat(upper, fill(Inf, nstate))

    np = length(par)
    if np >= length(y) - 1
        # Not enough data to continue
        return Dict(:aic => Inf, :bic => Inf, :aicc => Inf, :mse => Inf, :amse => Inf, :fit => nothing, :par => par, :states => init_state)
    end

    init_state = nothing

    println("par = ", par)
    println("y = ", y)
    println("nstate = ", nstate)
    println("errortype = ", errortype)
    println("trendtype = ", trendtype)
    println("seasontype = ", seasontype)
    println("damped = ", damped)
    println("lower = ", lower)
    println("upper = ", upper)
    println("opt_crit = ", opt_crit)
    println("nmse = ", nmse)
    println("bounds = ", bounds)
    println("m = ", m)
    println("initial_params = ", initial_params)
    println("fun = ", fun)
    println("iterations = ", iterations)
    println("kwargs = ", kwargs)

    optimized_fit = optim_ets_base(par, y, nstate, errortype, trendtype, seasontype, damped, lower,
        upper, opt_crit, nmse, bounds, m, initial_params, fun=optim_method, iterations=maxit, kwargs...)

    fit_par = optimized_fit["optimized_params"]

    states = fit_par["initstate"]

    if seasontype != "N"
        states = vcat(states, m * (seasontype == "M") - sum(states[(2+(trendtype!="N")):nstate]))
    end

    if !isnan(initial_params["alpha"])
        alpha = fit_par["alpha"]
    end
    if !isnan(initial_params["beta"])
        beta = fit_par["beta"]
    end
    if !isnan(initial_params["gamma"])
        gamma = fit_par["gamma"]
    end
    if !isnan(initial_params["phi"])
        phi = fit_par["phi"]
    end

    e = calculate_residuals(y, m, states, errortype, trendtype, seasontype, damped, alpha, beta, gamma, phi, nmse)

    lik = e["likelihood"]
    amse = e["amse"]
    states = e["state"]
    e = e["errors"]

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
        "par" => fit_par)
    return out
end