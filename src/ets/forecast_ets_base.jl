
export forecast_ets_base

function forecast_ets_base(object; h=nothing, level=[80, 95], fan=false, simulate=false, bootstrap=false, npaths=5000, PI=true, lambda=nothing, biasadj=nothing, kwargs...)
    m = object.m
    h = initialize_horizon(h, m)
    validate_horizon(h)
    biasadj = initialize_biasadj(biasadj, lambda, get_biasadj(object))
    simulate, bootstrap, fan, npaths, level = adjust_for_pi_and_biasadj(PI, biasadj, simulate, bootstrap, fan, npaths, level)
    level = process_level(level, fan)
    damped = Bool(object.components[4])
    n = length(object.x)
    simulate, bootstrap = adjust_simulation_flags(simulate, bootstrap)
    f = compute_forecast_data(object, h, simulate, npaths, level, bootstrap, damped, n)
    lower, upper = calculate_prediction_intervals(PI, biasadj, f, h, level)
    mean, lower, upper = apply_boxcox_transformation(f.mu, lower, upper, lambda, biasadj, PI)
    
    if !PI
        level = nothing
    end
    
    return Forecast(object, mean, level, object.x, upper, lower, object.fitted, object.residuals)
end

function get_biasadj(object, default=false)
    return hasfield(typeof(object), :biasadj) ? getfield(object, :biasadj) : default
end

function initialize_horizon(h, m)
    return h === nothing ? (m > 1 ? 2 * m : 10) : h
end

function validate_horizon(h)
    h <= 0 && throw(ArgumentError("Forecast horizon out of bounds"))
end

function initialize_biasadj(biasadj, lambda, object_biasadj)
    if lambda === nothing
        return false
    else
        return biasadj === nothing ? object_biasadj : (isa(biasadj, Bool) ? biasadj : warn_biasadj())
    end
end

function warn_biasadj()
    @warn "biasadj information not found, defaulting to FALSE."
    return false
end

function adjust_for_pi_and_biasadj(PI, biasadj, simulate, bootstrap, fan, npaths, level)
    if !PI && !biasadj
        return false, false, false, 2, [NaN]  # Ensure level is an array
    end
    return simulate, bootstrap, fan, npaths, level
end

function process_level(level, fan)
    if fan
        return 51:3:99
    else
        return validate_and_scale_level(level)
    end
end

function validate_and_scale_level(level)
    if minimum(level) > 0 && maximum(level) < 1
        return level * 100
    elseif minimum(level) < 0 || maximum(level) > 99.99
        error("Confidence limit out of range")
    end
    return sort(level)
end

function adjust_simulation_flags(simulate, bootstrap)
    return simulate || bootstrap, bootstrap
end

function compute_forecast_data(object, h, simulate, npaths, level, bootstrap, damped, n)
    if simulate
        println("I am here 1")
        return bootstrap_ets_forecast(object, h, npath=npaths, level=level, bootstrap=bootstrap)
    elseif object.components[1] == "A" && object.components[2] in ["A", "N"] && object.components[3] in ["N", "A"]
        println("I am here 2")
        return compute_forecast_case1(object, h, n)
    elseif object.components[1] == "M" && object.components[2] in ["A", "N"] && object.components[3] in ["N", "A"]
        println("I am here 3")
        return compute_forecast_case2(object, h, n)
    elseif object.components[1] == "M" && object.components[3] == "M" && object.components[2] != "M"
        println("I am here 4")
        return compute_forecast_case3(object, h, n)
    else
        println("I am here 5")
        return bootstrap_ets_forecast(object, h, npaths=npaths, level=level, bootstrap=bootstrap)
    end
end

function compute_forecast_case1(object, h, n)
    last_state = vec(object.states[n+1, :])
    trend_type, season_type, damped, m, sigma2, params = object.components[2], object.components[3], Bool(object.components[4]), object.m, object.sigma2, object.par
    return compute_forecast(h, last_state, trend_type, season_type, damped, m, sigma2, params)
end

function compute_forecast_case2(object, h, n)
    last_state = vec(object.states[n+1, :])
    trend_type, season_type, damped, m, sigma2, params = object.components[2], object.components[3], Bool(object.components[4]), object.m, object.sigma2, object.par
    return compute_theta(h, last_state, trend_type, season_type, damped, m, sigma2, params)
end

function compute_forecast_case3(object, h, n)
    last_state = vec(object.states[n+1, :])
    trend_type, season_type, damped, m, sigma2, params = object.components[2], object.components[3], Bool(object.components[4]), object.m, object.sigma2, object.par
    return compute_combined_forecast(h, last_state, trend_type, season_type, damped, m, sigma2, params)
end

function calculate_prediction_intervals(PI, biasadj, f, h, level)
    f_var = hasfield(typeof(f), :var) ? getfield(f, :var) : nothing
    
    if PI || biasadj
        if !isnothing(f_var)
            lower, upper = zeros(h, length(level)), zeros(h, length(level))
            for i in 1:length(level)
                marg_error = sqrt.(f_var) .* abs.(quantile(Normal(), (100 - level[i]) / 200))
                lower[:, i], upper[:, i] = f.mu .- marg_error, f.mu .+ marg_error
            end
        elseif hasfield(typeof(f), :lower)
            lower, upper = f.lower, f.upper
        else
            warn_no_intervals(PI, biasadj)
        end
    else
        lower, upper = nothing, nothing
    end
    return lower, upper
end

function warn_no_intervals(PI, biasadj)
    if PI
        @warn "No prediction intervals for this model"
    elseif biasadj
        @warn "No bias adjustment possible"
    end
end

function apply_boxcox_transformation(mean, lower, upper, lambda, biasadj, PI)
    if lambda != nothing
        mean = InvBoxCox(mean, lambda, biasadj, out)
        if PI
            lower, upper = InvBoxCox(lower, lambda), InvBoxCox(upper, lambda)
        end
    end

    if !PI
        lower, upper, level = nothing, nothing, nothing
    end

    return mean, lower, upper
end

function compute_forecast(h, last_state, trend_type, season_type, damped, m, sigma2, params)
    p = length(last_state)
    H = zeros(1, p)
    H[1] = 1
    if season_type == "A"
        H[p] = 1.0
    end
    if trend_type == "A"
        if damped
            H[2] = params["phi"]
        else
            H[2] = 1
        end
    end

    F = zeros(p, p)
    F[1, 1] = 1
    if trend_type == "A"
        if damped
            F[1, 2] = params["phi"]
            F[2, 2] = params["phi"]
        else
            F[1, 2] = 1
            F[2, 2] = 1
        end
    end
    if season_type == "A"
        F[p - m + 1, p] = 1
        F[(p - m + 2):p, (p - m + 1):(p - 1)] .= I(m - 1)
    end

    G = zeros(p, 1)
    G[1, 1] = params["alpha"]
    if trend_type == "A"
        G[2, 1] = params["beta"]
    end
    if season_type == "A"
        G[3, 1] = params["gamma"]
    end

    mu = zeros(h)
    Fj = I(p)
    cj = zeros(h - 1)
    if h > 1
        for i in 1:(h - 1)
            mu[i] = (H * Fj * last_state)[1]
            cj[i] = (H * Fj * G)[1]
            Fj = Fj * F
        end
        cj2 = cumsum(cj .^ 2)
        var = sigma2 * [1; 1 .+ cj2]
    else
        var = sigma2
    end
    mu[h] = (H * Fj * last_state)[1]

    return (mu = mu, var = var, cj = cj)
end

function compute_theta(h, last_state, trend_type, season_type, damped, m, sigma2, params)
    forecast = compute_forecast(h, last_state, trend_type, season_type, damped, m, sigma2, params)
    theta = zeros(h)
    theta[1] = forecast.mu[1] ^ 2
    if h > 1
        for j in 2:h
            theta[j] = forecast.mu[j] ^ 2 + sigma2 * sum(forecast.cj[1:(j - 1)] .^ 2 .* theta[(j - 1):-1:1])
        end
    end
    var = (1 + sigma2) * theta .- forecast.mu .^ 2
    return (mu = forecast.mu, var = var)
end

function compute_combined_forecast(h, last_state, trend_type, season_type, damped, m, sigma2, params)
    p = length(last_state)
    H1 = ones(1, 1 + (trend_type != "N"))
    H2 = zeros(1, m)
    H2[m] = 1

    alpha = get(params, "alpha", missing)
    beta = get(params, "beta", missing)
    gamma = get(params, "gamma", missing)
    phi = get(params, "phi", missing)
    
    if trend_type == "N"
        F1 = 1
        G1 = alpha
    else
        F1 = [1 1; 0 ifelse(damped, phi, 1)]
        G1 = [alpha alpha; beta beta]
    end
    F2 = vcat(hcat(zeros(m - 1)', 1), hcat(I(m - 1), zeros(m - 1)))
    G2 = zeros(m, m)
    G2[1, m] = gamma

    Mh = reshape(last_state[1:(p - m)], (p - m, 1)) * reshape(last_state[(p - m + 1):p], 1, m)
    Vh = zeros(length(Mh), length(Mh))

    H21 = kron(H2, H1)
    F21 = kron(F2, F1)
    G21 = kron(G2, G1)
    K = kron(G2, F1) + kron(F2, G1)

    mu = zeros(h)
    var = zeros(h)

    for i in 1:h
        mu[i] = dot(H1 * Mh, H2)
        var[i] = (1 + sigma2) * dot(H21 * Vh, H21) + sigma2 * mu[i] ^ 2
        vecMh = vec(Mh)
        Vh = F21 * Vh * F21' + sigma2 * (F21 * Vh * G21' + G21 * Vh * F21' + K * (Vh + vecMh * vecMh') * K' + sigma2 * G21 * (3 * Vh + 2 * vecMh * vecMh') * G21')
        Mh = F1 * Mh * F2' + G1 * Mh * G2' * sigma2
    end

    return (mu = mu, var = var)
end

function bootstrap_ets_forecast(obj, h; npaths, level, bootstrap)
    
    y_paths = fill(NaN, npaths, h)

    for i in 1:npaths
        y_paths[i, :] = simulate_ets(obj, h, future=true, bootstrap=bootstrap)
    end
    
    state = obj.states[size(obj.x, 1) + 1, :]
    m = obj.m
    component2 = switch(obj.components[2])
    component3 = switch(obj.components[3])
    phi = obj.components[4] == false ? 1.0 : obj.par["phi"]
    
    y_f = zeros(h)
    state = vec(as_matrix(state))
    etsforecast(state, m, component2, component3, phi, h, y_f)
    
    if abs(y_f[1] + 99999) < 1e-7
        error("Problem with multiplicative damped trend")
    end

    # Calculate lower and upper bounds, Check this part again
    if !all(isnan.(level))
        lower = [quantile_type8(y_paths[:, i], 0.5 - lvl / 200) for i in 1:size(y_paths, 2), lvl in level]
        upper = [quantile_type8(y_paths[:, i], 0.5 + lvl / 200) for i in 1:size(y_paths, 2), lvl in level]
    else
        lower, upper = nothing, nothing
    end
    
    return (mu = y_f, lower = lower, upper = upper)
end