function _static_regression_coefficients(y::AbstractVector{<:Real})
    n = length(y)
    y_sum = 0.0
    time_weighted_y_sum = 0.0

    @inbounds for t in 1:n
        y_t = Float64(y[t])
        y_sum += y_t
        time_weighted_y_sum += t * y_t
    end

    n_float = Float64(n)
    y_mean = y_sum / n_float
    weighted_average = time_weighted_y_sum / n_float
    slope = (6.0 * (2.0 * weighted_average - (n_float + 1.0) * y_mean)) / (n_float^2 - 1.0)
    intercept = y_mean - 0.5 * (n_float + 1.0) * slope
    return intercept, slope
end

function _simulate_static_theta_states!(states::Matrix{Float64}, residuals::Vector{Float64},
                                        y::Vector{Float64}, initial_level::Float64,
                                        alpha_value::Float64, theta_value::Float64,
                                        intercept::Float64, slope::Float64)
    n = length(y)
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    previous_level = initial_level
    previous_running_mean = 0.0
    q_power = 1.0
    geometric_sum = 1.0

    @inbounds for t in 1:n
        mu_t = previous_level + theta_multiplier * (q_power * intercept + geometric_sum * slope)
        y_t = y[t]
        residuals[t] = y_t - mu_t

        level_t = muladd(alpha_value, y_t, q * previous_level)
        running_mean_t = ((t - 1) * previous_running_mean + y_t) / t

        states[t, _LEVEL_COL] = level_t
        states[t, _RUNNING_MEAN_COL] = running_mean_t
        states[t, _INTERCEPT_COL] = intercept
        states[t, _SLOPE_COL] = slope
        states[t, _MEAN_FORECAST_COL] = mu_t

        previous_level = level_t
        previous_running_mean = running_mean_t

        q_power *= q
        geometric_sum = muladd(q, geometric_sum, 1.0)
    end

    return nothing
end

function _simulate_dynamic_theta_states!(states::Matrix{Float64}, residuals::Vector{Float64},
                                         y::Vector{Float64}, initial_level::Float64,
                                         alpha_value::Float64, theta_value::Float64)
    n = length(y)
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    previous_level = initial_level
    previous_running_mean = 0.0
    previous_intercept = 0.0
    previous_slope = 0.0
    q_power = 1.0
    geometric_sum = 1.0

    @inbounds for t in 1:n
        mu_t = previous_level + theta_multiplier * (q_power * previous_intercept + geometric_sum * previous_slope)
        y_t = y[t]
        residuals[t] = y_t - mu_t

        level_t = muladd(alpha_value, y_t, q * previous_level)
        running_mean_t = ((t - 1) * previous_running_mean + y_t) / t

        slope_t = if t == 1
            0.0
        else
            ((t - 2) * previous_slope + 6.0 * (y_t - previous_running_mean) / t) / (t + 1)
        end
        intercept_t = running_mean_t - 0.5 * (t + 1) * slope_t

        states[t, _LEVEL_COL] = level_t
        states[t, _RUNNING_MEAN_COL] = running_mean_t
        states[t, _INTERCEPT_COL] = intercept_t
        states[t, _SLOPE_COL] = slope_t
        states[t, _MEAN_FORECAST_COL] = mu_t

        previous_level = level_t
        previous_running_mean = running_mean_t
        previous_intercept = intercept_t
        previous_slope = slope_t

        q_power *= q
        geometric_sum = muladd(q, geometric_sum, 1.0)
    end

    return nothing
end

function _mean_horizon_mse(horizon_sse::Vector{Float64}, horizon_count::Vector{Int})
    total = 0.0
    used = 0

    @inbounds for h in eachindex(horizon_sse)
        count_h = horizon_count[h]
        if count_h > 0
            total += horizon_sse[h] / count_h
            used += 1
        end
    end

    return used == 0 ? Inf : total / used
end

function _multi_step_loss_static!(horizon_sse::Vector{Float64}, horizon_count::Vector{Int},
                                  y::Vector{Float64}, states::Matrix{Float64},
                                  alpha_value::Float64, theta_value::Float64,
                                  intercept::Float64, slope::Float64,
                                  nmse::Int)
    fill!(horizon_sse, 0.0)
    fill!(horizon_count, 0)

    n = length(y)
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    @inbounds for origin in 1:(n - 1)
        level = states[origin, _LEVEL_COL]
        q_power = q^origin
        geometric_sum = (1.0 - q_power * q) / alpha_value

        for h in 1:nmse
            target_index = origin + h
            target_index > n && break

            forecast_value = level + theta_multiplier * (q_power * intercept + geometric_sum * slope)
            error_value = y[target_index] - forecast_value
            horizon_sse[h] += error_value * error_value
            horizon_count[h] += 1

            level = muladd(alpha_value, forecast_value, q * level)
            q_power *= q
            geometric_sum = muladd(q, geometric_sum, 1.0)
        end
    end

    return _mean_horizon_mse(horizon_sse, horizon_count)
end

function _multi_step_loss_dynamic!(horizon_sse::Vector{Float64}, horizon_count::Vector{Int},
                                   y::Vector{Float64}, states::Matrix{Float64},
                                   alpha_value::Float64, theta_value::Float64,
                                   nmse::Int)
    fill!(horizon_sse, 0.0)
    fill!(horizon_count, 0)

    n = length(y)
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    @inbounds for origin in 1:(n - 1)
        level = states[origin, _LEVEL_COL]
        running_mean = states[origin, _RUNNING_MEAN_COL]
        intercept = states[origin, _INTERCEPT_COL]
        slope = states[origin, _SLOPE_COL]

        current_index = origin
        q_power = q^origin
        geometric_sum = (1.0 - q_power * q) / alpha_value

        for h in 1:nmse
            target_index = origin + h
            target_index > n && break

            forecast_value = level + theta_multiplier * (q_power * intercept + geometric_sum * slope)
            if target_index >= 3
                error_value = y[target_index] - forecast_value
                horizon_sse[h] += error_value * error_value
                horizon_count[h] += 1
            end

            next_index = current_index + 1
            level = muladd(alpha_value, forecast_value, q * level)
            next_running_mean = ((next_index - 1) * running_mean + forecast_value) / next_index

            next_slope = if next_index == 1
                0.0
            else
                ((next_index - 2) * slope + 6.0 * (forecast_value - running_mean) / next_index) / (next_index + 1)
            end
            next_intercept = next_running_mean - 0.5 * (next_index + 1) * next_slope

            current_index = next_index
            running_mean = next_running_mean
            slope = next_slope
            intercept = next_intercept

            q_power *= q
            geometric_sum = muladd(q, geometric_sum, 1.0)
        end
    end

    return _mean_horizon_mse(horizon_sse, horizon_count)
end

function _forecast_static_path!(forecasts::Vector{Float64}, fit::ThetaFit)
    n = size(fit.states, 1)
    alpha_value = fit.alpha
    theta_value = fit.theta
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    level = fit.states[n, _LEVEL_COL]
    intercept = fit.states[n, _INTERCEPT_COL]
    slope = fit.states[n, _SLOPE_COL]

    q_power = q^n
    geometric_sum = (1.0 - q_power * q) / alpha_value

    @inbounds for h in eachindex(forecasts)
        forecast_value = level + theta_multiplier * (q_power * intercept + geometric_sum * slope)
        forecasts[h] = forecast_value

        level = muladd(alpha_value, forecast_value, q * level)
        q_power *= q
        geometric_sum = muladd(q, geometric_sum, 1.0)
    end

    return nothing
end

function _forecast_dynamic_path!(forecasts::Vector{Float64}, fit::ThetaFit)
    n = size(fit.states, 1)
    alpha_value = fit.alpha
    theta_value = fit.theta
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    level = fit.states[n, _LEVEL_COL]
    running_mean = fit.states[n, _RUNNING_MEAN_COL]
    intercept = fit.states[n, _INTERCEPT_COL]
    slope = fit.states[n, _SLOPE_COL]

    current_index = n
    q_power = q^n
    geometric_sum = (1.0 - q_power * q) / alpha_value

    @inbounds for h in eachindex(forecasts)
        forecast_value = level + theta_multiplier * (q_power * intercept + geometric_sum * slope)
        forecasts[h] = forecast_value

        next_index = current_index + 1
        level = muladd(alpha_value, forecast_value, q * level)
        next_running_mean = ((next_index - 1) * running_mean + forecast_value) / next_index

        next_slope = ((next_index - 2) * slope + 6.0 * (forecast_value - running_mean) / next_index) / (next_index + 1)
        next_intercept = next_running_mean - 0.5 * (next_index + 1) * next_slope

        current_index = next_index
        running_mean = next_running_mean
        slope = next_slope
        intercept = next_intercept

        q_power *= q
        geometric_sum = muladd(q, geometric_sum, 1.0)
    end

    return nothing
end

function _forecast_mean!(forecasts::Vector{Float64}, fit::ThetaFit)
    if _is_dynamic_model(fit.model_type)
        _forecast_dynamic_path!(forecasts, fit)
    else
        _forecast_static_path!(forecasts, fit)
    end
    return nothing
end
