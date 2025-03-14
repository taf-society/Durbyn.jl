function simulate_ets(object::ETS, 
    nsim::Union{Int,Nothing}=nothing;
    seed::Union{Int,Nothing}=42,
    future::Bool=true,
    bootstrap::Bool=false,
    innov::Union{Vector{Float64},Nothing}=nothing)

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

    if isnothing(innov)
        if !isnothing(seed)
            seed!(seed)
        end
    else
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
        e = sample(res, nsim, replace=true)
    elseif isnothing(innov)
        e = rand(Normal(0, sqrt(sigma2)), nsim)
    elseif length(innov) == nsim
        e = innov
    else
        error("Length of innov must be equal to nsim")
    end

    if components[1] == "M"
        e = max.(-1, e)
    end

    y = zeros(nsim)
    errors = ets_model_type_code(components[1])
    trend = ets_model_type_code(components[2])
    season = ets_model_type_code(components[3])
    alpha = check_component(par, "alpha")
    beta = ifelse(trend == "N", 0.0, check_component(par, "beta"))
    gamma = ifelse(season == "N", 0.0, check_component(par, "gamma"))
    phi = ifelse(!components[4], 1.0, check_component(par, "phi"))

    simulate_ets_base(initstate, m, errors, trend, season, alpha, beta, gamma, phi, nsim, y, e)

    if isnan(y[1])
        error("Problem with multiplicative damped trend")
    end

    if !isnothing(lambda)
        y = InvBoxCox(y, lambda=lambda)
    end

    return y
end