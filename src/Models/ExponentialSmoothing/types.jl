export EtsModel, ETS

abstract type ETS end

"""
ETS model output

# Fields
- `fitted::Vector{Float64}`: The fitted values from the ETS model, representing the predicted values at each time point.
- `residuals::Vector{Float64}`: The residuals, which are the differences between the observed values and the fitted values.
- `components::Vector{Any}`: A collection of the model components such as level, trend, and seasonality.
- `x::Vector{Float64}`: The original time series data on which the ETS model was fitted.
- `par::Dict{String, Any}`: A dictionary containing the parameters of the ETS model, where the keys are parameter names and the values are the parameter values.
- `initstate::DataFrame`: A DataFrame containing the initial state estimates of the model.
- `states::DataFrame`: A DataFrame containing the state estimates of the model over time.
- `sse::Float64`: The sum of squared errors (SSE) of the model, a measure of the model's fit to the data.
- `sigma2::Float64`: The variance of the residuals, indicating the spread of the residuals around zero.
- `m::Int`: The frequency of the seasonal component, e.g., 12 for monthly data with yearly seasonality.
- `lambda::Float64`: The Box-Cox transformation parameter, used if the data were transformed before fitting the model.
- `biasadj::Bool`: A boolean flag indicating whether bias adjustment was applied to the model.
- `loglik::Float64`: Log-likelihood of the model.
- `aic::Float64`: Akaike Information Criterion (AIC) for model selection.
- bic::Float64: Bayesian Information Criterion (BIC) for model selection.
- aicc::Float64: Corrected Akaike Information Criterion (AICc) for small sample sizes.
- mse::Float64:  Mean Squared Error of the model fit.
- amse::Float64:  Average Mean Squared Error, typically used for forecasting accuracy.
- fit::Vector{Float64}: The fitted model.
- `method::String`: The method used for model fitting.

"""

struct EtsModel <: ETS
    fitted::AbstractArray
    residuals::AbstractArray
    components::Vector{String}
    x::AbstractArray
    par::Dict{String,Any}
    loglik::Float64
    initstate::AbstractArray
    states::AbstractArray
    state_names::Vector{String}
    sse::Float64
    sigma2::Float64
    m::Int
    lambda::Union{Float64,Bool,Nothing,Symbol}
    biasadj::Bool
    aic::Float64
    bic::Float64
    aicc::Float64
    mse::Float64
    amse::Float64
    fit::Union{Dict{String,Any}, Nothing}
    method::String
end

struct EtsRefit
    model::String
    alpha::Union{AbstractFloat,Nothing,Bool}
    beta::Union{AbstractFloat,Nothing,Bool}
    gamma::Union{AbstractFloat,Nothing,Bool}
    phi::Union{AbstractFloat,Nothing,Bool}
end

struct SimpleHoltWinters <: ETS
    sse::Float64
    fitted::Vector{Float64}
    residuals::Vector{Float64}
    level::Vector{Float64}
    trend::Vector{Float64}
    season::Vector{Float64}
    phi::Float64
end

struct HoltWintersConventional <: ETS
    fitted::AbstractArray
    residuals::AbstractArray
    components::Vector{String}
    x::AbstractArray
    par::Dict{String,Any}
    initstate::AbstractArray
    states::AbstractArray
    state_names::Vector{String}
    sse::Float64
    sigma2::Float64
    m::Int
    lambda::Union{Nothing,Float64}
    biasadj::Bool
    method::String
end

@inline function ets_model_type_code(x::AbstractString)
    if x == "N"
        return 0
    elseif x == "A"
        return 1
    elseif x == "M"
        return 2
    end
    throw(ArgumentError("Unknown ETS model type: $x"))
end

@inline function ets_model_type_code(x::Char)
    if x == 'N'
        return 0
    elseif x == 'A'
        return 1
    elseif x == 'M'
        return 2
    end
    throw(ArgumentError("Unknown ETS model type: $x"))
end

struct ETSWorkspace
    x::Vector{Float64}
    e::Vector{Float64}
    amse::Vector{Float64}
    olds::Vector{Float64}
    s::Vector{Float64}
    f::Vector{Float64}
    denom::Vector{Float64}
    init_state::Vector{Float64}
end

function ETSWorkspace(n::Int, m::Int, nmse::Int, max_state_len::Int)
    seasonal_len = max(24, m)
    amse_len = max(nmse, 30)
    return ETSWorkspace(
        zeros(Float64, max_state_len * (n + 1)),
        zeros(Float64, n),
        zeros(Float64, amse_len),
        zeros(Float64, seasonal_len),
        zeros(Float64, seasonal_len),
        zeros(Float64, 30),
        zeros(Float64, 30),
        zeros(Float64, max_state_len),
    )
end

# Integer codes for opt_crit (eliminates string comparison in hot path)
const OPT_CRIT_LIK = 0
const OPT_CRIT_MSE = 1
const OPT_CRIT_AMSE = 2
const OPT_CRIT_SIGMA = 3
const OPT_CRIT_MAE = 4

@inline function opt_crit_code(s::Symbol)
    s === :lik && return OPT_CRIT_LIK
    s === :mse && return OPT_CRIT_MSE
    s === :amse && return OPT_CRIT_AMSE
    s === :sigma && return OPT_CRIT_SIGMA
    s === :mae && return OPT_CRIT_MAE
    throw(ArgumentError("Unknown optimization criterion: :$s"))
end

# Integer codes for bounds
const BOUNDS_BOTH = 0
const BOUNDS_USUAL = 1
const BOUNDS_ADMISSIBLE = 2

@inline function bounds_code(s::Symbol)
    s === :both && return BOUNDS_BOTH
    s === :usual && return BOUNDS_USUAL
    s === :admissible && return BOUNDS_ADMISSIBLE
    throw(ArgumentError("Unknown bounds type: :$s"))
end
