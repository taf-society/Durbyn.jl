using Random
using Distributions

function simulate_ets(object::ETS, 
    nsim::Union{Int,Nothing}=nothing;
    seed::Union{Int,Nothing}=nothing,
    future::Bool=true,
    bootstrap::Bool=false,
    innov::Union{Vector{Float64},Nothing}=nothing)

    # Access parameters and fields from the ETS object
    x = object.x
    m = object.m
    states = object.states
    residuals = object.residuals
    sigma2 = object.sigma2
    components = object.components
    par = object.par
    lambda = object.lambda
    biasadj = object.biasadj

    # Determine the number of simulations
    nsim = isnothing(nsim) ? length(x) : nsim

    # Set random seed if innov is not provided and seed is given
    if isnothing(innov)
        if !isnothing(seed)
            Random.seed!(seed)
        end
    else
        nsim = length(innov)
    end

    # Handle missing `x` or `m`
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

    # Select the initial state
    initstate = future ? states[end, :] : states[rand(1:size(states, 1)), :]

    # Generate errors based on bootstrap or normal distribution
    if bootstrap
        res = filter(!ismissing, residuals) .- mean(filter(!ismissing, residuals))
        e = sample(res, nsim, replace=true)
    elseif isnothing(innov)
        e = rand(Normal(0, sqrt(sigma2)), nsim)
    elseif length(innov) == nsim
        e = innov
    else
        error("Length of innov must be equal to nsim")
    end

    # Ensure errors are bounded for multiplicative models
    if components[1] == "M"
        e = max.(-1, e)
    end

    # Prepare parameters for the ETS simulation
    y = zeros(nsim)
    errors = switch(components[1])
    trend = switch(components[2])
    season = switch(components[3])
    alpha = par["alpha"]
    beta = ifelse(trend == "N", 0.0, par["beta"])
    gamma = ifelse(season == "N", 0.0, par["gamma"])
    phi = ifelse(!components[4], 1.0, par["phi"])

    # Simulate the time series
    etssimulate(initstate, m, errors, trend, season, alpha, beta, gamma, phi, nsim, y, e)

    # Check for any issues with the simulation output
    if isnan(y[1])
        error("Problem with multiplicative damped trend")
    end

    # Apply inverse Box-Cox transformation if lambda is provided
    if !isnothing(lambda)
        y = InvBoxCox(y, lambda=lambda)
    end

    return y
end