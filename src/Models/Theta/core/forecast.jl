function compute_prediction_intervals(fit::ThetaFit, h::Int, level::Vector{<:Real};
                                      n_samples::Int=200, seed::Int=0)
    n = length(fit.y)
    n <= 3 && throw(ArgumentError("Need at least 4 observations to compute prediction intervals"))

    residual_start = _is_dynamic_model(fit.model_type) ? min(3, n) : min(2, n)
    residual_slice = @view fit.residuals[residual_start:n]
    sigma = std(residual_slice, corrected = true)
    sigma = isfinite(sigma) ? sigma : 0.0

    alpha_value = fit.alpha
    theta_value = fit.theta
    q = 1.0 - alpha_value
    theta_multiplier = _theta_weight(theta_value)

    rng = MersenneTwister(seed)
    samples = Matrix{Float64}(undef, h, n_samples)

    final_level = fit.states[n, _LEVEL_COL]
    final_running_mean = fit.states[n, _RUNNING_MEAN_COL]
    final_intercept = fit.states[n, _INTERCEPT_COL]
    final_slope = fit.states[n, _SLOPE_COL]

    dynamic_model = _is_dynamic_model(fit.model_type)

    @inbounds for sample_idx in 1:n_samples
        level_state = final_level
        running_mean_state = final_running_mean
        intercept_state = final_intercept
        slope_state = final_slope

        current_index = n
        q_power = q^n
        geometric_sum = (1.0 - q_power * q) / alpha_value

        for h_idx in 1:h
            mean_forecast = level_state + theta_multiplier * (q_power * intercept_state + geometric_sum * slope_state)
            simulated_value = mean_forecast + sigma * randn(rng)
            samples[h_idx, sample_idx] = simulated_value

            next_index = current_index + 1
            level_state = muladd(alpha_value, simulated_value, q * level_state)

            if dynamic_model
                next_running_mean = ((next_index - 1) * running_mean_state + simulated_value) / next_index
                next_slope = ((next_index - 2) * slope_state + 6.0 * (simulated_value - running_mean_state) / next_index) / (next_index + 1)
                next_intercept = next_running_mean - 0.5 * (next_index + 1) * next_slope

                running_mean_state = next_running_mean
                slope_state = next_slope
                intercept_state = next_intercept
            end

            current_index = next_index
            q_power *= q
            geometric_sum = muladd(q, geometric_sum, 1.0)
        end
    end

    intervals = Dict{String, Vector{Float64}}()
    for level_value in level
        lower_q = (100.0 - level_value) / 200.0
        upper_q = lower_q + level_value / 100.0
        level_key = round(Int, level_value)

        intervals["lo_$(level_key)"] = [quantile(@view(samples[i, :]), lower_q) for i in 1:h]
        intervals["hi_$(level_key)"] = [quantile(@view(samples[i, :]), upper_q) for i in 1:h]
    end

    return intervals
end

function forecast(fit::ThetaFit; h::Int, level::Vector{<:Real}=[80, 95])
    forecasts = zeros(Float64, h)
    _forecast_mean!(forecasts, fit)

    intervals = compute_prediction_intervals(fit, h, level)

    if fit.decompose && !isnothing(fit.seasonal_component)
        seasonal_indices = fit.seasonal_component[1:fit.m]
        seasonal_forecast = repeat_seasonal(seasonal_indices, h)

        if fit.decomposition_type === :multiplicative
            forecasts .*= seasonal_forecast
            for interval_key in keys(intervals)
                intervals[interval_key] .*= seasonal_forecast
            end
        else
            forecasts .+= seasonal_forecast
            for interval_key in keys(intervals)
                intervals[interval_key] .+= seasonal_forecast
            end
        end
    end

    lower = hcat([intervals["lo_$(round(Int, level_value))"] for level_value in level]...)
    upper = hcat([intervals["hi_$(round(Int, level_value))"] for level_value in level]...)

    return Forecast(
        fit,
        "Theta($(fit.model_type))",
        forecasts,
        level,
        fit.y_original,
        upper,
        lower,
        fit.fitted,
        fit.residuals,
    )
end
