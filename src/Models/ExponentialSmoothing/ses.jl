export ses, SES

"""
    SES

Simple Exponential Smoothing model output structure.

SES is the simplest form of exponential smoothing (equivalent to ETS(A,N,N)),
with no trend or seasonality components. It is suitable for forecasting data
with no clear trend or seasonal pattern.

# Fields
- `fitted::AbstractArray`: The fitted values (one-step ahead predictions) at each time point.
- `residuals::AbstractArray`: The residuals (observed - fitted values).
- `components::Union{Vector{Any}, Any}`: Model components (level only for SES).
- `x::AbstractArray`: The original time series data.
- `par::Any`: Dictionary containing model parameters (alpha).
- `loglik::Float64`: Log-likelihood of the model.
- `initstate::AbstractArray`: Initial state estimate (initial level).
- `states::AbstractArray`: Level estimates over time.
- `state_names::Any`: Names of the state variables.
- `sse::Float64`: Sum of squared errors, a measure of model fit.
- `sigma2::Float64`: Residual variance (σ²).
- `m::Int`: Seasonal period (e.g., 12 for monthly data, 1 for non-seasonal).
- `lambda::Union{Float64,Bool,Nothing}`: Box-Cox transformation parameter (nothing if not used).
- `biasadj::Bool`: Boolean flag indicating whether bias adjustment was applied.
- `aic::Float64`: Akaike Information Criterion for model selection.
- `bic::Float64`: Bayesian Information Criterion for model selection.
- `aicc::Float64`: Corrected AIC for small sample sizes.
- `mse::Float64`: Mean Squared Error of the model fit.
- `amse::Float64`: Average Mean Squared Error.
- `fit::Any`: The fitted model object.
- `method::String`: The method used for model fitting ("Simple Exponential Smoothing").

# See also
[`ses`](@ref), [`ets`](@ref), [`forecast`](@ref)

"""
struct SES
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
    lambda::Union{Float64,Bool,Nothing}
    biasadj::Bool
    aic::Float64
    bic::Float64
    aicc::Float64
    mse::Float64
    amse::Float64
    fit::Union{Dict{String,Any}, Nothing}
    method::String
end

"""
    ses(y; initial=:optimal, alpha=nothing, lambda=nothing, biasadj=false, options=NelderMeadOptions())
    ses(y, m; initial=:optimal, alpha=nothing, lambda=nothing, biasadj=false, options=NelderMeadOptions())

Fit a Simple Exponential Smoothing (SES) model to a time series.

SES is the simplest form of exponential smoothing (equivalent to ETS(A,N,N)),
suitable for data with no trend or seasonal pattern. The model uses a single
smoothing parameter α to exponentially weight past observations.

# Arguments
- `y::AbstractArray`: Time series data to fit.
- `m::Int`: Seasonal period (optional, defaults to 1 for non-seasonal data).
             Use 12 for monthly data, 4 for quarterly, etc.

# Keyword Arguments
- `initial::Symbol=:optimal`: Initialization method:
  - `:optimal`: Uses state-space optimization via ETS framework (default).
  - `:simple`: Uses conventional Holt-Winters initialization.
- `alpha::Union{Float64,Nothing}=nothing`: Smoothing parameter (0 < α < 1).
  If `nothing`, α is estimated from the data.
- `lambda::Union{Float64,Bool,Nothing}=nothing`: Box-Cox transformation parameter.
  - `nothing`: No transformation (default).
  - `:auto` or `true`: Automatically select optimal λ.
  - `Float64`: Use specified λ value.
- `biasadj::Bool=false`: Apply bias adjustment for Box-Cox back-transformation.
- `options::NelderMeadOptions`: Optimization options for parameter estimation.

# Returns
- `SES`: Fitted SES model object containing fitted values, residuals, parameters,
  states, and information criteria (AIC, BIC, AICc when `initial=:optimal`).

# Model Formulation
The SES model in state-space form:
```
yₜ = ℓₜ₋₁ + εₜ
ℓₜ = ℓₜ₋₁ + α·εₜ
```
where ℓₜ is the level at time t, and εₜ ~ N(0, σ²).

The h-step ahead forecast is constant: ŷₜ₊ₕ = ℓₜ for all h ≥ 1.

# Examples
```julia
using Durbyn.ExponentialSmoothing

# Simple data
y = [10.5, 12.3, 11.8, 13.1, 12.9, 14.2, 13.8, 15.1, 14.7, 16.0]

# Fit with optimal initialization (default)
fit = ses(y)
println(fit)  # Display model summary

# Fit with specified alpha
fit_fixed = ses(y, alpha=0.3)

# Fit with Box-Cox transformation
fit_bc = ses(y, lambda=0.5, biasadj=true)

# Generate forecasts
fc = forecast(fit, h=6)

# For monthly seasonal data
monthly_data = randn(60) .+ 100
fit_monthly = ses(monthly_data, 12)  # m=12 for monthly frequency
fc_monthly = forecast(fit_monthly, h=12)
```

# See also
[`SES`](@ref), [`ets`](@ref), [`forecast`](@ref), [`holt`](@ref), [`hw`](@ref)
"""
function ses(
    y::AbstractArray;
    initial::Symbol = :optimal,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),)

    ses(y, 1, initial = initial, alpha = alpha, lambda = lambda, biasadj = biasadj, options = options)
end

function ses(
    y::AbstractArray,
    m::Int;
    initial::Symbol = :optimal,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),
)

    initial = _check_arg(initial, (:optimal, :simple), "initial")
    model = nothing
    if initial === :optimal
        model = ets_base_model(
            y,
            m,
            "ANN",
            alpha = alpha,
            opt_crit = :mse,
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
        loglik = model.loglik
        aic = model.aic
        bic = model.bic
        aicc = model.aicc
        mse = model.mse
        amse = model.amse
        fit = model.fit
    else
        model = holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = false,
            gamma = false,
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
        loglik = NaN
        aic = NaN
        bic = NaN
        aicc = NaN
        mse = NaN
        amse = NaN
        fit = nothing
    end
    
    return SES(
        model.fitted,
        model.residuals,
        model.components,
        model.x,
        model.par,
        loglik,
        model.initstate,
        model.states,
        model.state_names,
        model.sse,
        model.sigma2,
        model.m,
        model.lambda,
        model.biasadj,
        aic,
        bic,
        aicc,
        mse,
        amse,
        fit,
        "Simple Exponential Smoothing",
    )
end