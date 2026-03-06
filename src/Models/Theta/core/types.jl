const _ALPHA_LOWER = 1e-4
const _ALPHA_UPPER = 1.0 - 1e-4
const _THETA_LOWER = 1.0
const _THETA_UPPER = 1.0e6
const _LEVEL_ABS_BOUND = 1.0e12
const _SEASONAL_Z_90 = 1.6448536269514722

const _LEVEL_COL = 1
const _RUNNING_MEAN_COL = 2
const _INTERCEPT_COL = 3
const _SLOPE_COL = 4
const _MEAN_FORECAST_COL = 5

@enum ThetaModelType begin
    STM
    OTM
    DSTM
    DOTM
end

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
    decomposition_type::Symbol
    seasonal_component::Union{Vector{Float64}, Nothing}
    y_original::Vector{Float64}
end

function Base.show(io::IO, fit::ThetaFit)
    println(io, "Theta Model (", fit.model_type, ")")
    println(io, "------------------------------")
    println(io, "Observations: ", length(fit.y_original))
    println(io, "Seasonal period: ", fit.m)
    println(io, "alpha: ", round(fit.alpha, digits = 6))
    println(io, "theta: ", round(fit.theta, digits = 6))
    println(io, "initial_level: ", round(fit.initial_level, digits = 6))
    println(io, "mse: ", round(fit.mse, digits = 6))
    if fit.decompose
        println(io, "decomposition: ", fit.decomposition_type)
    end
end

@inline _is_dynamic_model(model_type::ThetaModelType) = model_type === DSTM || model_type === DOTM

@inline function _theta_weight(theta_value::Float64)
    return 1.0 - inv(theta_value)
end

function repeat_seasonal(seasonal_values::AbstractVector{<:Real}, horizon::Int)
    repeats = cld(horizon, length(seasonal_values))
    return repeat(seasonal_values, repeats)[1:horizon]
end
