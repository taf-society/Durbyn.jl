"""
    Theta

Implementation of the Theta forecasting method and its variants (STM, OTM, DSTM, DOTM).

The Theta method decomposes a time series into two "theta lines" - one capturing
the long-term trend and another the short-term dynamics - then combines their
forecasts. This implementation follows Assimakopoulos & Nikolopoulos (2000) and
the optimized variants from Fiorucci et al. (2016).

# Model Types
- `STM`: Simple Theta Model (theta=2, optimized alpha)
- `OTM`: Optimized Theta Model (optimized theta and alpha)
- `DSTM`: Dynamic Simple Theta Model (dynamic trend estimation)
- `DOTM`: Dynamic Optimized Theta Model (dynamic + optimized)

# References
- Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model: a decomposition
  approach to forecasting. International Journal of Forecasting, 16(4), 521-530.
- Fiorucci, J. A., Pellegrini, T. R., Louzada, F., Petropoulos, F., & Koehler, A. B.
  (2016). Models for optimising the theta method and their relationship to state
  space models. International Journal of Forecasting, 32(4), 1151-1161.
"""
module Theta

using Statistics
using LinearAlgebra
using Random: MersenneTwister, randn
using Distributions: Normal, quantile as dist_quantile
using Tables

import ..Utils: is_constant
import ..Optimize: optim
import ..Stats: acf, decompose
import ..Generics: forecast, Forecast
import ..Grammar: theta 
using ..Grammar: ModelFormula, ThetaTerm, _extract_single_term

export theta, auto_theta, ThetaFit, ThetaModelType
export STM, OTM, DSTM, DOTM

const THETA_TOL = 1e-10

"""
    ThetaModelType

Enum representing the four variants of the Theta model:
- `STM`: Simple Theta Model (theta=2 fixed, alpha optimized)
- `OTM`: Optimized Theta Model (both theta and alpha optimized)
- `DSTM`: Dynamic Simple Theta Model (theta=2, dynamic trend)
- `DOTM`: Dynamic Optimized Theta Model (optimized theta, dynamic trend)
"""
@enum ThetaModelType begin
    STM
    OTM
    DSTM
    DOTM
end

"""
    ThetaFit

Struct holding a fitted Theta model.

# Fields
- `model_type::ThetaModelType`: Type of Theta model (STM, OTM, DSTM, DOTM)
- `alpha::Float64`: Smoothing parameter (0 < alpha < 1)
- `theta::Float64`: Theta parameter (≥ 1)
- `initial_level::Float64`: Initial smoothed level
- `states::Matrix{Float64}`: State matrix (n × 5): [level, mean_y, A, B, mu]
- `residuals::Vector{Float64}`: In-sample residuals
- `fitted::Vector{Float64}`: In-sample fitted values
- `mse::Float64`: Mean squared error
- `y::Vector{Float64}`: Original series (possibly seasonally adjusted)
- `m::Int`: Seasonal period
- `decompose::Bool`: Whether seasonal decomposition was applied
- `decomposition_type::String`: "multiplicative" or "additive"
- `seasonal_component::Union{Vector{Float64}, Nothing}`: Seasonal factors
- `y_original::Vector{Float64}`: Original unadjusted series
"""
struct ThetaFit
    model_type::ThetaModelType
    alpha::Float64
    theta::Float64
    initial_level::Float64
    states::Matrix{Float64}
    residuals::Vector{Float64}
    fitted::Vector{Float64}
    mse::Float64
    y::Vector{Float64}
    m::Int
    decompose::Bool
    decomposition_type::String
    seasonal_component::Union{Vector{Float64}, Nothing}
    y_original::Vector{Float64}
end

function Base.show(io::IO, fit::ThetaFit)
    println(io, "Theta Model ($(fit.model_type))")
    println(io, "─────────────────────────────")
    println(io, "Observations: ", length(fit.y_original))
    println(io, "Seasonal period: ", fit.m)
    println(io, "Parameters:")
    println(io, "  α (smoothing): ", round(fit.alpha, digits=4))
    println(io, "  θ (theta):     ", round(fit.theta, digits=4))
    println(io, "  Initial level: ", round(fit.initial_level, digits=4))
    println(io, "MSE: ", round(fit.mse, digits=4))
    if fit.decompose
        println(io, "Seasonal decomposition: ", fit.decomposition_type)
    end
end

"""
    repeat_seasonal(season_vals, h) -> Vector

Repeat seasonal values to cover horizon h.
"""
function repeat_seasonal(season_vals::AbstractVector, h::Int)
    repeats = ceil(Int, h / length(season_vals))
    return repeat(season_vals, repeats)[1:h]
end

"""
    init_theta_state(y, model_type, initial_level, alpha, theta) -> Vector{Float64}

Initialize state vector [level, mean_y, A, B, mu] for Theta model.
"""
function init_theta_state(y::AbstractVector{T}, model_type::ThetaModelType,
                          initial_level::T, alpha::T, theta::T) where T
    n = length(y)

    if model_type == DSTM || model_type == DOTM
        A = y[1]
        B = T(0)
        mu = y[1]
    else
        y_mean = mean(y)
        weighted_avg = dot(y, 1:n) / n
        B = (6 * (2 * weighted_avg - (n + 1) * y_mean)) / (n^2 - 1)
        A = y_mean - (n + 1) * B / 2
        mu = initial_level + (1 - 1/theta) * (A + B)
    end

    level = alpha * y[1] + (1 - alpha) * initial_level

    return T[level, y[1], A, B, mu]
end

"""
    update_theta_state!(states, i, model_type, alpha, theta, y_val, use_mu)

Update state at time i given previous state and observation.
"""
function update_theta_state!(states::Matrix{T}, i::Int, model_type::ThetaModelType,
                             alpha::T, theta::T, y_val::T, use_mu::Bool) where T
    level = states[i-1, 1]
    mean_y = states[i-1, 2]
    A = states[i-1, 3]
    B = states[i-1, 4]

    idx = i - 1

    states[i, 5] = level + (1 - 1/theta) * (A * (1 - alpha)^idx +
                                             B * (1 - (1 - alpha)^(idx + 1)) / alpha)

    y_use = use_mu ? states[i, 5] : y_val

    states[i, 1] = alpha * y_use + (1 - alpha) * level

    states[i, 2] = (idx * mean_y + y_use) / (idx + 1)

    if model_type == DSTM || model_type == DOTM
        states[i, 4] = ((idx - 1) * B + 6 * (y_use - mean_y) / (idx + 1)) / (idx + 2)
        states[i, 3] = states[i, 2] - states[i, 4] * (idx + 2) / 2
    else
        states[i, 3] = A
        states[i, 4] = B
    end
end

"""
    forecast_theta_states!(states, n, model_type, forecasts, alpha, theta)

Generate h-step ahead forecasts by extending state space.
"""
function forecast_theta_states!(states::Matrix{T}, n::Int, model_type::ThetaModelType,
                                forecasts::AbstractVector{T}, alpha::T, theta::T) where T
    h = length(forecasts)
    extended_states = zeros(T, n + h, 5)
    extended_states[1:n, :] = states[1:n, :]

    for j in 1:h
        update_theta_state!(extended_states, n + j, model_type, alpha, theta, T(0), true)
        forecasts[j] = extended_states[n + j, 5]
    end
end

"""
    compute_theta_likelihood!(y, states, model_type, initial_level, alpha, theta,
                              residuals, amse, nmse) -> Float64

Compute likelihood (MSE) for Theta model given parameters.
Also fills residuals and average MSE arrays.
"""
function compute_theta_likelihood!(y::AbstractVector{T}, states::Matrix{T},
                                   model_type::ThetaModelType, initial_level::T,
                                   alpha::T, theta::T, residuals::AbstractVector{T},
                                   amse::AbstractVector{T}, nmse::Int) where T
    n = length(y)
    denom = zeros(T, nmse)
    forecasts = zeros(T, nmse)

    init_states = init_theta_state(y, model_type, initial_level, alpha, theta)
    states[1, :] = init_states

    fill!(amse, T(0))

    residuals[1] = y[1] - states[1, 5]

    for i in 2:n
        forecast_theta_states!(states, i - 1, model_type, forecasts, alpha, theta)

        if abs(forecasts[1] - (-99999.0)) < THETA_TOL
            return T(NaN)
        end

        residuals[i] = y[i] - forecasts[1]

        for j in 1:nmse
            if i + j - 1 <= n
                denom[j] += 1.0
                tmp = y[i + j - 1] - forecasts[j]
                amse[j] = (amse[j] * (denom[j] - 1.0) + tmp^2) / denom[j]
            end
        end

        update_theta_state!(states, i, model_type, alpha, theta, y[i], false)
    end

    mean_y = mean(abs.(y))
    if mean_y < THETA_TOL
        mean_y = THETA_TOL
    end

    return sum(residuals[4:end] .^ 2) / mean_y
end

"""
    compute_theta_residuals(y, model_type, initial_level, alpha, theta, nmse)

Compute residuals and states for given Theta parameters.
Returns (amse, residuals, states, mse).
"""
function compute_theta_residuals(y::AbstractVector{T}, model_type::ThetaModelType,
                                 initial_level::T, alpha::T, theta::T, nmse::Int) where T
    n = length(y)
    states = zeros(T, n, 5)
    residuals = zeros(T, n)
    amse = zeros(T, nmse)

    mse = compute_theta_likelihood!(y, states, model_type, initial_level, alpha, theta,
                                    residuals, amse, nmse)

    return (amse=amse, residuals=residuals, states=states, mse=mse)
end

"""
    init_theta_parameters(y, model_type; initial_level=nothing, alpha=nothing, theta=nothing)

Initialize Theta parameters and determine which to optimize.
"""
function init_theta_parameters(y::AbstractVector{T}, model_type::ThetaModelType;
                               initial_level::Union{T, Nothing}=nothing,
                               alpha::Union{T, Nothing}=nothing,
                               theta::Union{T, Nothing}=nothing) where T
    if model_type in (STM, DSTM)
        init_level = isnothing(initial_level) ? T(y[1] / 2) : initial_level
        init_alpha = isnothing(alpha) ? T(0.5) : alpha
        init_theta = T(2.0)

        opt_level = isnothing(initial_level)
        opt_alpha = isnothing(alpha)
        opt_theta = false
    else
        init_level = isnothing(initial_level) ? T(y[1] / 2) : initial_level
        init_alpha = isnothing(alpha) ? T(0.5) : alpha
        init_theta = isnothing(theta) ? T(2.0) : theta

        opt_level = isnothing(initial_level)
        opt_alpha = isnothing(alpha)
        opt_theta = isnothing(theta)
    end

    return (
        initial_level = init_level,
        alpha = init_alpha,
        theta = init_theta,
        optimize_level = opt_level,
        optimize_alpha = opt_alpha,
        optimize_theta = opt_theta
    )
end

"""
    optimize_theta_parameters(y, model_type, init_params, nmse) -> NamedTuple

Optimize Theta model parameters using Nelder-Mead.
"""
function optimize_theta_parameters(y::AbstractVector{T}, model_type::ThetaModelType,
                                   init_params::NamedTuple, nmse::Int) where T
    x0 = T[]
    lower = T[]
    upper = T[]

    if init_params.optimize_level
        push!(x0, init_params.initial_level)
        push!(lower, T(-1e10))
        push!(upper, T(1e10))
    end

    if init_params.optimize_alpha
        push!(x0, init_params.alpha)
        push!(lower, T(0.1))
        push!(upper, T(0.99))
    end

    if init_params.optimize_theta
        push!(x0, init_params.theta)
        push!(lower, T(1.0))
        push!(upper, T(1e10))
    end

    if isempty(x0)
        return init_params
    end

    function objective(params)
        j = 1
        level = init_params.optimize_level ? params[j] : init_params.initial_level
        j += init_params.optimize_level ? 1 : 0

        alpha = init_params.optimize_alpha ? params[j] : init_params.alpha
        j += init_params.optimize_alpha ? 1 : 0

        theta = init_params.optimize_theta ? params[j] : init_params.theta

        n = length(y)
        states = zeros(T, n, 5)
        residuals = zeros(T, n)
        amse = zeros(T, nmse)

        mse = compute_theta_likelihood!(y, states, model_type, level, alpha, theta,
                                        residuals, amse, nmse)

        if isnan(mse) || mse < -1e9
            return T(1e20)
        end

        return max(mse, T(-1e10))
    end

    # Use L-BFGS-B for bounded optimization
    result = optim(x0, objective;
                   method="L-BFGS-B",
                   lower=lower, upper=upper,
                   control=Dict("maxit" => 1000))

    opt_params = result.par
    j = 1

    opt_level = init_params.optimize_level ? opt_params[j] : init_params.initial_level
    j += init_params.optimize_level ? 1 : 0

    opt_alpha = init_params.optimize_alpha ? opt_params[j] : init_params.alpha
    j += init_params.optimize_alpha ? 1 : 0

    opt_theta = init_params.optimize_theta ? opt_params[j] : init_params.theta

    return (
        initial_level = opt_level,
        alpha = opt_alpha,
        theta = opt_theta,
        optimize_level = init_params.optimize_level,
        optimize_alpha = init_params.optimize_alpha,
        optimize_theta = init_params.optimize_theta
    )
end

"""
    fit_theta_model(y, m, model_type; initial_level=nothing, alpha=nothing,
                    theta=nothing, nmse=3) -> ThetaFit

Fit a specific Theta model variant to time series data.

# Arguments
- `y`: Time series data
- `m`: Seasonal period
- `model_type`: One of STM, OTM, DSTM, DOTM
- `initial_level`: Initial smoothed level (nothing = optimize)
- `alpha`: Smoothing parameter (nothing = optimize)
- `theta`: Theta parameter (nothing = optimize, except for STM/DSTM where theta=2)
- `nmse`: Number of steps for multi-step MSE calculation

# Returns
A `ThetaFit` struct containing the fitted model.
"""
function fit_theta_model(y::AbstractVector{T}, m::Int, model_type::ThetaModelType;
                         initial_level::Union{T, Nothing}=nothing,
                         alpha::Union{T, Nothing}=nothing,
                         theta::Union{T, Nothing}=nothing,
                         nmse::Int=3) where T<:Real
    y = collect(Float64.(y))

    init_params = init_theta_parameters(y, model_type;
                                        initial_level=initial_level,
                                        alpha=alpha, theta=theta)

    opt_params = optimize_theta_parameters(y, model_type, init_params, nmse)

    result = compute_theta_residuals(y, model_type,
                                     opt_params.initial_level,
                                     opt_params.alpha,
                                     opt_params.theta, nmse)

    n = length(y)
    fitted_vals = y .- result.residuals

    return ThetaFit(
        model_type,
        opt_params.alpha,
        opt_params.theta,
        opt_params.initial_level,
        result.states,
        result.residuals,
        fitted_vals,
        result.mse,
        y,
        m,
        false,
        "none",
        nothing,
        y
    )
end

"""
    theta(y, m; model_type=OTM, kwargs...) -> ThetaFit

Fit a Theta model to a time series.

# Arguments
- `y`: Time series data (vector)
- `m`: Seasonal period (use 1 for non-seasonal)
- `model_type`: Type of Theta model (STM, OTM, DSTM, DOTM). Default: OTM
- `initial_level`: Initial level (nothing = optimize)
- `alpha`: Smoothing parameter (nothing = optimize)
- `theta`: Theta parameter (nothing = optimize or 2.0 for STM/DSTM)
- `nmse`: Steps for multi-step MSE (default: 3)

# Returns
A `ThetaFit` object containing the fitted model.

# Example
```julia
y = randn(100) .+ collect(1:100) * 0.1  # Trend + noise
fit = theta(y, 1)
fc = forecast(fit, 12)
```
"""
function theta(y::AbstractVector{<:Real}, m::Int=1;
               model_type::ThetaModelType=OTM,
               initial_level::Union{Real, Nothing}=nothing,
               alpha::Union{Real, Nothing}=nothing,
               theta_param::Union{Real, Nothing}=nothing,
               nmse::Int=3)
    init_lvl = isnothing(initial_level) ? nothing : Float64(initial_level)
    alpha_val = isnothing(alpha) ? nothing : Float64(alpha)
    theta_val = isnothing(theta_param) ? nothing : Float64(theta_param)

    return fit_theta_model(Float64.(y), m, model_type;
                           initial_level=init_lvl, alpha=alpha_val,
                           theta=theta_val, nmse=nmse)
end

"""
    auto_theta(y, m; model=nothing, decomposition_type="multiplicative",
               nmse=3, kwargs...) -> ThetaFit

Automatically select and fit the best Theta model variant.

Performs automatic model selection by:
1. Testing for seasonality using ACF
2. Applying seasonal decomposition if needed
3. Fitting all model variants (or specified model) and selecting by MSE

# Arguments
- `y`: Time series data
- `m`: Seasonal period
- `model`: Specific model type to fit (nothing = try all and select best)
- `decomposition_type`: "multiplicative" or "additive" for seasonal adjustment
- `initial_level`, `alpha`, `theta`: Optional fixed parameter values
- `nmse`: Steps for multi-step MSE

# Returns
A `ThetaFit` object with the best fitting model.

# Example
```julia
# Monthly data with trend and seasonality
y = sin.(2π .* (1:120) ./ 12) .+ collect(1:120) .* 0.05 .+ randn(120) .* 0.1
fit = auto_theta(y, 12)
fc = forecast(fit, 24)
```
"""
function auto_theta(y::AbstractVector{<:Real}, m::Int;
                    model::Union{ThetaModelType, Nothing}=nothing,
                    initial_level::Union{Real, Nothing}=nothing,
                    alpha::Union{Real, Nothing}=nothing,
                    theta_param::Union{Real, Nothing}=nothing,
                    nmse::Int=3,
                    decomposition_type::String="multiplicative")
    y = collect(Float64.(y))
    n = length(y)

    if nmse < 1 || nmse > 30
        throw(ArgumentError("nmse must be between 1 and 30"))
    end

    if is_constant(y)
        return theta(y, m; model_type=STM, initial_level=mean(y)/2,
                     alpha=0.5, theta_param=2.0, nmse=nmse)
    end

    do_decompose = false
    y_work = copy(y)

    if m >= 4 && n >= 2 * m
        r = acf(y, m, m).values
        stat = sqrt((1 + 2 * sum(r[2:end-1] .^ 2)) / n)
        z_critical = dist_quantile(Normal(), 0.95)
        do_decompose = abs(r[end]) / stat > z_critical
    end

    seasonal_component = nothing
    data_positive = minimum(y) > 0

    if do_decompose
        if decomposition_type == "multiplicative" && !data_positive
            decomposition_type = "additive"
        end

        seasonal_component = decompose(x=y, m=m, type=decomposition_type).seasonal

        if decomposition_type == "multiplicative" && any(seasonal_component .< 0.01)
            decomposition_type = "additive"
            seasonal_component = decompose(x=y, m=m, type="additive").seasonal
        end

        y_work = if decomposition_type == "additive"
            y .- seasonal_component
        else
            y ./ seasonal_component
        end
    end

    model_types = isnothing(model) ? [STM, OTM, DSTM, DOTM] : [model]

    best_mse = Inf
    best_fit = nothing

    init_lvl = isnothing(initial_level) ? nothing : Float64(initial_level)
    alpha_val = isnothing(alpha) ? nothing : Float64(alpha)
    theta_val = isnothing(theta_param) ? nothing : Float64(theta_param)

    for mtype in model_types
        try
            fit = fit_theta_model(y_work, m, mtype;
                                  initial_level=init_lvl,
                                  alpha=alpha_val,
                                  theta=theta_val,
                                  nmse=nmse)

            if !isnan(fit.mse) && fit.mse < best_mse
                best_mse = fit.mse
                best_fit = fit
            end
        catch
            continue
        end
    end

    if isnothing(best_fit)
        throw(ErrorException("No model could be fitted"))
    end

    if do_decompose
        adjusted_residuals = if decomposition_type == "multiplicative"
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
            y .- adjusted_residuals,
            best_fit.mse,
            y_work,
            m,
            true,
            decomposition_type,
            seasonal_component,
            y
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
        "none",
        nothing,
        y
    )
end

"""
    compute_prediction_intervals(fit, h, level; n_samples=200) -> Dict

Compute prediction intervals using simulation.
"""
function compute_prediction_intervals(fit::ThetaFit, h::Int, level::Vector{Int};
                                      n_samples::Int=200, seed::Int=0)
    T = Float64
    n = length(fit.y)

    sigma = std(fit.residuals[4:end], corrected=true)
    mean_y = mean(fit.y)

    final_level = fit.states[n, 1]
    A = fit.states[n, 3]
    B = fit.states[n, 4]

    alpha = fit.alpha
    theta = fit.theta

    rng = MersenneTwister(seed)
    samples = zeros(T, h, n_samples)

    for s in 1:n_samples
        local_level = final_level
        local_mean = mean_y
        local_A = A
        local_B = B

        for i in 1:h
            idx = n + i - 1

            mu = local_level + (1 - 1/theta) * (
                local_A * (1 - alpha)^idx +
                local_B * (1 - (1 - alpha)^(idx + 1)) / alpha
            )

            samples[i, s] = mu + randn(rng) * sigma

            local_level = alpha * samples[i, s] + (1 - alpha) * local_level
            local_mean = (idx * local_mean + samples[i, s]) / (idx + 1)
            local_B = ((idx - 1) * local_B + 6 * (samples[i, s] - local_mean) / (idx + 1)) / (idx + 2)
            local_A = local_mean - local_B * (idx + 2) / 2
        end
    end

    intervals = Dict{String, Vector{T}}()
    for lv in level
        q_lo = (100 - lv) / 200
        q_hi = q_lo + lv / 100

        intervals["lo_$lv"] = [quantile(samples[i, :], q_lo) for i in 1:h]
        intervals["hi_$lv"] = [quantile(samples[i, :], q_hi) for i in 1:h]
    end

    return intervals
end

"""
    forecast(fit::ThetaFit; h, level=[80, 95]) -> Forecast

Generate forecasts from a fitted Theta model.

# Arguments
- `fit`: A fitted `ThetaFit` object
- `h::Int`: Forecast horizon (keyword argument)
- `level`: Confidence levels for prediction intervals (default: [80, 95])

# Returns
A `Forecast` object containing point forecasts and prediction intervals.

# Example
```julia
fit = auto_theta(y, 12)
fc = forecast(fit, h=24)
fc.mean        # Point forecasts
fc.lower[1]    # 80% lower bounds
fc.upper[2]    # 95% upper bounds
```
"""
function forecast(fit::ThetaFit; h::Int, level::Vector{Int}=[80, 95])
    T = Float64
    n = length(fit.y)

    forecasts = zeros(T, h)
    forecast_theta_states!(fit.states, n, fit.model_type, forecasts, fit.alpha, fit.theta)

    intervals = compute_prediction_intervals(fit, h, level)

    if fit.decompose && !isnothing(fit.seasonal_component)
        period = length(fit.seasonal_component) ÷ (n ÷ fit.m + 1)
        if period == 0
            period = fit.m
        end
        seasonal_indices = fit.seasonal_component[1:fit.m]
        seas_forecast = repeat_seasonal(seasonal_indices, h)

        if fit.decomposition_type == "multiplicative"
            forecasts .*= seas_forecast
            for key in keys(intervals)
                intervals[key] .*= seas_forecast
            end
        else
            forecasts .+= seas_forecast
            for key in keys(intervals)
                intervals[key] .+= seas_forecast
            end
        end
    end

    lower = [intervals["lo_$lv"] for lv in level]
    upper = [intervals["hi_$lv"] for lv in level]

    method = "Theta($(fit.model_type))"

    return Forecast(
        fit,
        method,
        forecasts,
        level,
        fit.y_original,
        upper,
        lower,
        fit.fitted,
        fit.residuals
    )
end

# Formula interface
include("theta_formula_interface.jl")

end
