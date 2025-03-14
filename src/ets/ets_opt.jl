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

    # result = optim_nm(par -> objective_fun(par, y, nstate, errortype, trendtype, seasontype, damped, lower,
    #         upper, opt_crit, nmse, bounds, m, init_alpha, init_beta, init_gamma,
    #         init_phi, opt_alpha, opt_beta, opt_gamma, opt_phi), opt_params, iter_max = iterations, kwargs...)

    # optimized_params = result[:par]
    # optimized_value = result[:value]
    # number_of_iterations = iterations

    optimized_params = create_params(optimized_params, opt_alpha, opt_beta, opt_gamma, opt_phi)

    return Dict("optimized_params" => optimized_params, "optimized_value" => optimized_value, 
    "number_of_iterations" => number_of_iterations)
end


function objective_fun(par, y, nstate, errortype, trendtype, seasontype, damped, lower, upper, opt_crit, 
    nmse, bounds, m, init_alpha, init_beta, init_gamma, init_phi, opt_alpha, opt_beta, 
    opt_gamma, opt_phi)

j = 1

alpha = opt_alpha ? par[j] : init_alpha
j += opt_alpha

beta = nothing
if trendtype != "N"
beta = opt_beta ? par[j] : init_beta
j += opt_beta
end

gamma = nothing
if seasontype != "N"
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
elseif beta !== nothing && isnan(beta)
throw(ArgumentError("beta must be numeric"))
elseif gamma !== nothing && isnan(gamma)
throw(ArgumentError("gamma must be numeric"))
elseif phi !== nothing && isnan(phi)
throw(ArgumentError("phi must be numeric"))
end

if !check_param(alpha, beta, gamma, phi, lower, upper, bounds, m)
return Inf
end

p = nstate + (seasontype != "N" ? 1 : 0)
states = zeros(p * (length(y) + 1))
states[1:nstate] = par[end-nstate+1:end]

out = calculate_residuals(y, m, states, errortype, trendtype, seasontype, damped, alpha, beta, gamma, phi, nmse)
lik, amse, e = out["likelihood"], out["amse"], out["errors"]

lik = ifelse((isnan(lik) || lik < -1e10 || abs(lik + 99999) < 1e-7), Inf, lik)

if opt_crit == "lik"
return lik
elseif opt_crit == "mse"
return amse[1]
elseif opt_crit == "amse"
return mean(amse[1:nmse])
elseif opt_crit == "sigma"
return mean(e .^ 2)
elseif opt_crit == "mae"
return mean(abs.(e))
else
error("Unknown optimization criterion")
end
end
