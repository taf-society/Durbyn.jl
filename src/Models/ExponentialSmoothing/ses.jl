export ses, SES

"""
SES model output

# Fields
- `fitted::Vector{Float64}`: The fitted values from the ETS model, representing the predicted values at each time point.
- `residuals::Vector{Float64}`: The residuals, which are the differences between the observed values and the fitted values.
- `components::Vector{Any}`: A collection of the model components such as level, trend, and seasonality.
- `x::Vector{Float64}`: The original time series data on which the ETS model was fitted.
- `par::Dict{String, Any}`: A dictionary containing the parameters of the ETS model, where the keys are parameter names and the values are the parameter values.
- `initstate::DataFrame`: A DataFrame containing the initial state estimates of the model.
- `states::DataFrame`: A DataFrame containing the state estimates of the model over time.
- `SSE::Float64`: The sum of squared errors (SSE) of the model, a measure of the model's fit to the data.
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
struct SES
    fitted::AbstractArray
    residuals::AbstractArray
    components::Vector{Any}
    x::AbstractArray
    par
    loglik::Union{Float64, Int}
    initstate::AbstractArray
    states::AbstractArray
    state_names
    SSE::Union{Float64, Int}
    sigma2::Union{Float64, Int}
    m::Int
    lambda::Union{Float64, Bool, Nothing}
    biasadj::Bool
    aic::Union{Float64, Int}
    bic::Union{Float64, Int}
    aicc::Union{Float64, Int}
    mse::Union{Float64, Int}
    amse::Union{Float64, Int}
    fit
    method::String
end

function ses(y::AbstractArray, m::Int; initial::String="optimal",
    alpha::Union{Float64,Bool,Nothing}=nothing,
    lambda::Union{Float64,Bool,Nothing}=nothing, biasadj::Bool=false)

    initial = match_arg(initial, ["optimal", "simple"])
    model = nothing
    if initial == "optimal"
        model = ets_base_model(y, m, "ANN", alpha=alpha, opt_crit="mse", lambda=lambda, biasadj=biasadj)
    else
        model = holt_winters_conventional(y, m, alpha=alpha, beta=false, gamma=false, lambda=lambda, biasadj=biasadj)
    end

    return model
end