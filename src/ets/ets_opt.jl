function create_params(optimized_params::AbstractArray, opt_alpha::Bool, opt_beta::Bool, 
    opt_gamma::Bool, opt_phi::Bool)
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


function optim_ets_base(opt_params, y, nstate, errortype, trendtype, seasontype, damped, lower, upper,
    opt_crit, nmse, bounds, m, initial_params; fun=NelderMead(), iterations=2000, kwargs...)

    init_alpha = initial_params["alpha"]
    init_beta = initial_params["beta"]
    init_gamma = initial_params["gamma"]
    init_phi = initial_params["phi"]
    opt_alpha = !isnan(init_alpha)
    opt_beta = !isnan(init_beta)
    opt_gamma = !isnan(init_gamma)
    opt_phi = !isnan(init_phi)
    
    result = optimize(par -> objective_fun(par, y, nstate, errortype, trendtype, seasontype, damped, lower,
            upper, opt_crit, nmse, bounds, m, init_alpha, init_beta, init_gamma,
            init_phi, opt_alpha, opt_beta, opt_gamma, opt_phi), opt_params,
        fun, Optim.Options(iterations=iterations, kwargs...))

    optimized_params = Optim.minimizer(result)
    optimized_value = Optim.minimum(result)
    number_of_iterations = Optim.iterations(result)

    optimized_params = create_params(optimized_params, opt_alpha, opt_beta, opt_gamma, opt_phi)

    return Dict("optimized_params" => optimized_params, "optimized_value" => optimized_value, 
    "number_of_iterations" => number_of_iterations)
end

function objective_fun(par, y, nstate, errortype, trendtype, seasontype, damped, lower, upper, opt_crit,
    nmse, bounds, m, init_alpha, init_beta, init_gamma, init_phi, opt_alpha, opt_beta, opt_gamma, opt_phi)

    j = 1
    # Extract or set alpha
    if opt_alpha
        alpha = par[j]
        j = j + 1
    else
        alpha = init_alpha
    end

    if isnan(alpha)
        throw(ArgumentError("alpha problem. alpha must be a floating number!"))
    end

    # Extract or set beta
    if trendtype != "N"
        if opt_beta
            beta = par[j]
            j = j + 1
        else
            beta = init_beta
        end

        if isnan(beta)
            throw(ArgumentError("beta problem. beta must be a floating number!"))
        end
    else
        beta = NaN
    end
    # Extract or set gamma
    if seasontype != "N"
        if opt_gamma
            gamma = par[j]
            j = j + 1
        else
            gamma = init_gamma
        end

        if isnan(gamma)
            throw(ArgumentError("gamma problem. gamma must be a floating number!"))
        end
    else
        m = 1
        gamma = NaN
    end


    # Extract or set phi
    if damped
        if opt_phi
            phi = par[j]
            j = j + 1
        else
            phi = init_phi
        end

        if isnan(phi)
            throw(ArgumentError("phi problem. phi must be a floating number!"))
        end
    else
        phi = NaN
    end

    if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
        return 1e11
    end

    # Calculate the size of the state
    p = nstate + ifelse(seasontype != 0, 1, 0)
    n = length(y)

    # Initialize the state array
    states = zeros(p * (n + 1))
    states[1:nstate] = par[end-nstate+1:end]

    out = calculate_residuals(y, m, states, errortype, trendtype, seasontype, damped, alpha, beta, gamma, phi, nmse)

    lik = out["likelihood"]
    amse = out["amse"]
    e = out["errors"]
    state = out["state"]

    # Prevent perfect fit issues
    if lik < -1e10
        lik = -1e10
    end
    if isnan(lik)
        lik = Inf
    end
    if abs(lik + 99999) < 1e-7
        lik = Inf
    end

    # Calculate the objective value based on the optimization criterion
    if opt_crit == "lik"
        objval = lik
    elseif opt_crit == "mse"
        objval = amse[1]
    elseif opt_crit == "amse"
        objval = mean(amse[1:nmse])
    elseif opt_crit == "sigma"
        objval = mean(e .^ 2)
    elseif opt_crit == "mae"
        objval = mean(abs.(e))
    else
        error("Unknown optimization criterion")
    end

    return objval
end