function etssimulate(x, m, error, trend, season, alpha, beta, gamma, phi, h, y, e)
    m = max(1, m)
    if m > 24 && season > NONE
        return
    end
    
    olds = zeros(24)
    s = zeros(24)
    f = zeros(10)
    
    l, b, s = initialize_states(x, m, trend, season)
    
    for i in 1:h
        oldl, oldb, olds = l, b, copy(s)
        
        forecast_ets_base(oldl, oldb, olds, m, trend, season, phi, f, 1)
        
        if abs(f[1] - NA) < TOL
            y[1] = NA
            return
        end
        
        y[i] = compute_simulated_y(f[1], e[i], error)
        
        l, b, s = update_state(oldl, l, oldb, b, olds, s, m, trend, season, alpha, beta, gamma, phi, y[i])
    end
end

function compute_simulated_y(f, e, error)
    return error == ADD ? f + e : f * (1.0 + e)
end


function simulate_ets(object::ETS, 
    nsim::Union{Int,Nothing}=nothing;
    seed::Union{Int,Nothing}=nothing,
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
            Random.seed!(seed)
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

    etssimulate(initstate, m, errors, trend, season, alpha, beta, gamma, phi, nsim, y, e)

    if isnan(y[1])
        error("Problem with multiplicative damped trend")
    end

    if !isnothing(lambda)
        y = InvBoxCox(y, lambda=lambda)
    end

    return y
end