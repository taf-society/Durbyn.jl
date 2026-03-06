function _build_parameter_spec(y::AbstractVector{<:Real}, model_type::ThetaModelType;
                               initial_level::Union{Real, Nothing}=nothing,
                               alpha::Union{Real, Nothing}=nothing,
                               theta::Union{Real, Nothing}=nothing)
    level_value = isnothing(initial_level) ? 0.5 * Float64(y[1]) : Float64(initial_level)
    alpha_value = isnothing(alpha) ? 0.5 : Float64(alpha)

    if model_type === STM || model_type === DSTM
        theta_value = 2.0
        optimize_theta = false
    else
        theta_value = isnothing(theta) ? 2.0 : Float64(theta)
        optimize_theta = isnothing(theta)
    end

    optimize_level = isnothing(initial_level)
    optimize_alpha = isnothing(alpha)

    if !(_ALPHA_LOWER <= alpha_value <= _ALPHA_UPPER)
        throw(ArgumentError("alpha must be in [$_ALPHA_LOWER, $_ALPHA_UPPER]"))
    end

    if theta_value < _THETA_LOWER
        throw(ArgumentError("theta must be >= $_THETA_LOWER"))
    end

    if !isfinite(level_value)
        throw(ArgumentError("initial_level must be finite"))
    end

    return (
        initial_level = level_value,
        alpha = alpha_value,
        theta = theta_value,
        optimize_level = optimize_level,
        optimize_alpha = optimize_alpha,
        optimize_theta = optimize_theta,
    )
end

function _pack_parameters(spec)
    x0 = Float64[]
    lower = Float64[]
    upper = Float64[]

    if spec.optimize_level
        push!(x0, spec.initial_level)
        push!(lower, -_LEVEL_ABS_BOUND)
        push!(upper, _LEVEL_ABS_BOUND)
    end

    if spec.optimize_alpha
        push!(x0, spec.alpha)
        push!(lower, _ALPHA_LOWER)
        push!(upper, _ALPHA_UPPER)
    end

    if spec.optimize_theta
        push!(x0, spec.theta)
        push!(lower, _THETA_LOWER)
        push!(upper, _THETA_UPPER)
    end

    return x0, lower, upper
end

function _unpack_parameters(params::AbstractVector{<:Real}, spec)
    idx = 1
    level_value = spec.optimize_level ? Float64(params[idx]) : spec.initial_level
    idx += spec.optimize_level ? 1 : 0

    alpha_value = spec.optimize_alpha ? Float64(params[idx]) : spec.alpha
    idx += spec.optimize_alpha ? 1 : 0

    theta_value = spec.optimize_theta ? Float64(params[idx]) : spec.theta

    return level_value, alpha_value, theta_value
end

@inline function _validate_parameters(level_value::Float64, alpha_value::Float64, theta_value::Float64)
    if !isfinite(level_value) || abs(level_value) > _LEVEL_ABS_BOUND
        return false
    end
    if !(isfinite(alpha_value) && _ALPHA_LOWER <= alpha_value <= _ALPHA_UPPER)
        return false
    end
    if !(isfinite(theta_value) && _THETA_LOWER <= theta_value <= _THETA_UPPER)
        return false
    end
    return true
end

function _evaluate_theta_loss!(states::Matrix{Float64}, residuals::Vector{Float64},
                               horizon_sse::Vector{Float64}, horizon_count::Vector{Int},
                               y::Vector{Float64}, model_type::ThetaModelType,
                               initial_level::Float64, alpha_value::Float64,
                               theta_value::Float64, static_intercept::Float64,
                               static_slope::Float64, nmse::Int)
    if !_validate_parameters(initial_level, alpha_value, theta_value)
        return Inf
    end

    if _is_dynamic_model(model_type)
        _simulate_dynamic_theta_states!(states, residuals, y, initial_level, alpha_value, theta_value)
        return _multi_step_loss_dynamic!(horizon_sse, horizon_count, y, states,
                                         alpha_value, theta_value, nmse)
    else
        _simulate_static_theta_states!(states, residuals, y, initial_level, alpha_value,
                                       theta_value, static_intercept, static_slope)
        return _multi_step_loss_static!(horizon_sse, horizon_count, y, states, alpha_value,
                                        theta_value, static_intercept, static_slope, nmse)
    end
end

function _optimize_theta_parameters(y::Vector{Float64}, model_type::ThetaModelType,
                                    parameter_spec, static_intercept::Float64,
                                    static_slope::Float64, nmse::Int)
    x0, lower, upper = _pack_parameters(parameter_spec)

    if isempty(x0)
        return parameter_spec
    end

    n = length(y)
    objective_states = Matrix{Float64}(undef, n, 5)
    objective_residuals = Vector{Float64}(undef, n)
    objective_horizon_sse = zeros(Float64, nmse)
    objective_horizon_count = zeros(Int, nmse)

    function objective_fn(params)
        level_value, alpha_value, theta_value = _unpack_parameters(params, parameter_spec)
        return _evaluate_theta_loss!(objective_states, objective_residuals,
                                     objective_horizon_sse, objective_horizon_count,
                                     y, model_type, level_value, alpha_value,
                                     theta_value, static_intercept, static_slope, nmse)
    end

    result = optimize(objective_fn, x0, :lbfgsb;
                      lower = lower,
                      upper = upper,
                      max_iterations = 400,
                      reltol = sqrt(eps(Float64)),
                      factr = 1e9,
                      pgtol = 1e-8)

    level_value, alpha_value, theta_value = _unpack_parameters(result.minimizer, parameter_spec)

    return (
        initial_level = level_value,
        alpha = alpha_value,
        theta = theta_value,
        optimize_level = parameter_spec.optimize_level,
        optimize_alpha = parameter_spec.optimize_alpha,
        optimize_theta = parameter_spec.optimize_theta,
    )
end

function fit_theta_model(y::AbstractVector{<:Real}, m::Int, model_type::ThetaModelType;
                         initial_level::Union{Real, Nothing}=nothing,
                         alpha::Union{Real, Nothing}=nothing,
                         theta::Union{Real, Nothing}=nothing,
                         nmse::Int=3)
    y_values = Float64.(y)
    n = length(y_values)

    if n <= 3
        throw(ArgumentError("Time series too short: need at least 4 observations, got $n"))
    end
    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse must be between 1 and 30"))
    end

    static_intercept, static_slope = _static_regression_coefficients(y_values)
    parameter_spec = _build_parameter_spec(y_values, model_type;
                                           initial_level = initial_level,
                                           alpha = alpha,
                                           theta = theta)

    optimal_spec = _optimize_theta_parameters(y_values, model_type, parameter_spec,
                                              static_intercept, static_slope, nmse)

    fitted_states = Matrix{Float64}(undef, n, 5)
    fitted_residuals = Vector{Float64}(undef, n)
    horizon_sse = zeros(Float64, nmse)
    horizon_count = zeros(Int, nmse)

    model_mse = _evaluate_theta_loss!(fitted_states, fitted_residuals,
                                      horizon_sse, horizon_count,
                                      y_values, model_type,
                                      optimal_spec.initial_level,
                                      optimal_spec.alpha,
                                      optimal_spec.theta,
                                      static_intercept,
                                      static_slope,
                                      nmse)

    fitted_values = y_values .- fitted_residuals

    return ThetaFit(
        model_type,
        optimal_spec.alpha,
        optimal_spec.theta,
        optimal_spec.initial_level,
        fitted_states,
        fitted_residuals,
        fitted_values,
        model_mse,
        y_values,
        m,
        false,
        :none,
        nothing,
        y_values,
    )
end

function theta(y::AbstractVector{<:Real}, m::Int=1;
               model_type::ThetaModelType=OTM,
               initial_level::Union{Real, Nothing}=nothing,
               alpha::Union{Real, Nothing}=nothing,
               theta_param::Union{Real, Nothing}=nothing,
               nmse::Int=3)
    return fit_theta_model(y, m, model_type;
                           initial_level = initial_level,
                           alpha = alpha,
                           theta = theta_param,
                           nmse = nmse)
end

function _is_seasonal(y::Vector{Float64}, m::Int)
    n = length(y)
    if m < 4 || n < 2 * m
        return false
    end

    acf_values = acf(y, m, m).values
    if length(acf_values) < m + 1
        return false
    end

    denominator = sqrt((1.0 + 2.0 * sum(abs2, @view(acf_values[2:end-1]))) / n)
    if denominator <= 0.0 || !isfinite(denominator)
        return false
    end

    t_stat = abs(acf_values[end]) / denominator
    return t_stat > _SEASONAL_Z_90
end

function _seasonal_adjustment(y::Vector{Float64}, m::Int, decomposition_type::Symbol)
    decomposition = decompose(x = y, m = m, type = decomposition_type)
    seasonal_component = Float64.(decomposition.seasonal)

    if decomposition_type === :additive
        adjusted = y .- seasonal_component
    else
        adjusted = y ./ seasonal_component
    end

    return adjusted, seasonal_component
end

function auto_theta(y::AbstractVector{<:Real}, m::Int;
                    model::Union{ThetaModelType, Nothing}=nothing,
                    initial_level::Union{Real, Nothing}=nothing,
                    alpha::Union{Real, Nothing}=nothing,
                    theta_param::Union{Real, Nothing}=nothing,
                    nmse::Int=3,
                    decomposition_type::Symbol=:multiplicative)
    y_values = Float64.(y)
    n = length(y_values)

    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse must be between 1 and 30"))
    end
    if n <= 3
        throw(ArgumentError("Time series too short: need at least 4 observations, got $n"))
    end

    if is_constant(y_values)
        return theta(y_values, m;
                     model_type = STM,
                     initial_level = 0.5 * mean(y_values),
                     alpha = 0.5,
                     theta_param = 2.0,
                     nmse = nmse)
    end

    do_decompose = _is_seasonal(y_values, m)
    y_working = copy(y_values)
    seasonal_component::Union{Vector{Float64}, Nothing} = nothing
    used_decomposition = :none

    if do_decompose
        used_decomposition = decomposition_type

        if used_decomposition === :multiplicative && minimum(y_values) <= 0.0
            used_decomposition = :additive
        end

        y_working, seasonal_component = _seasonal_adjustment(y_values, m, used_decomposition)

        if used_decomposition === :multiplicative && any(x -> x < 0.01, seasonal_component)
            used_decomposition = :additive
            y_working, seasonal_component = _seasonal_adjustment(y_values, m, used_decomposition)
        end
    end

    model_candidates = isnothing(model) ? (STM, OTM, DSTM, DOTM) : (model,)

    best_fit::Union{ThetaFit, Nothing} = nothing
    best_mse = Inf

    for candidate in model_candidates
        try
            fitted_model = fit_theta_model(y_working, m, candidate;
                                           initial_level = initial_level,
                                           alpha = alpha,
                                           theta = theta_param,
                                           nmse = nmse)

            if isfinite(fitted_model.mse) && fitted_model.mse < best_mse
                best_mse = fitted_model.mse
                best_fit = fitted_model
            end
        catch
            continue
        end
    end

    isnothing(best_fit) && throw(ErrorException("No Theta model could be fitted"))

    if do_decompose
        adjusted_residuals = if used_decomposition === :multiplicative
            best_fit.residuals .* seasonal_component
        else
            best_fit.residuals .+ seasonal_component
        end

        return ThetaFit(
            best_fit.model_type,
            best_fit.alpha,
            best_fit.theta,
            best_fit.initial_level,
            best_fit.states,
            adjusted_residuals,
            y_values .- adjusted_residuals,
            best_fit.mse,
            y_working,
            m,
            true,
            used_decomposition,
            seasonal_component,
            y_values,
        )
    end

    return ThetaFit(
        best_fit.model_type,
        best_fit.alpha,
        best_fit.theta,
        best_fit.initial_level,
        best_fit.states,
        best_fit.residuals,
        best_fit.fitted,
        best_fit.mse,
        best_fit.y,
        m,
        false,
        :none,
        nothing,
        y_values,
    )
end
