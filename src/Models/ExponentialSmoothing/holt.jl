export holt, Holt

"""
    Holt

Holt's linear trend method model output structure.

Holt's method extends simple exponential smoothing to capture linear trends in
time series data. It uses two smoothing parameters: α for the level and β for
the trend component. The method can optionally include damping (φ) to prevent
forecasts from trending indefinitely.

# Fields
- `fitted::AbstractArray`: The fitted values (one-step ahead predictions) at each time point.
- `residuals::AbstractArray`: The residuals (observed - fitted values).
- `components::Vector{Any}`: Model components (level and trend).
- `x::AbstractArray`: The original time series data.
- `par::Any`: Dictionary containing model parameters (alpha, beta, phi if damped).
- `loglik::Union{Float64,Int}`: Log-likelihood of the model.
- `initstate::AbstractArray`: Initial state estimates (initial level and trend).
- `states::AbstractArray`: Level and trend estimates over time.
- `state_names::Any`: Names of the state variables.
- `SSE::Union{Float64,Int}`: Sum of squared errors, a measure of model fit.
- `sigma2::Union{Float64,Int}`: Residual variance (σ²).
- `m::Int`: Seasonal period (typically 1 for non-seasonal data).
- `lambda::Union{Float64,Bool,Nothing}`: Box-Cox transformation parameter (nothing if not used).
- `biasadj::Bool`: Boolean flag indicating whether bias adjustment was applied.
- `aic::Union{Float64,Int}`: Akaike Information Criterion for model selection.
- `bic::Union{Float64,Int}`: Bayesian Information Criterion for model selection.
- `aicc::Union{Float64,Int}`: Corrected AIC for small sample sizes.
- `mse::Union{Float64,Int}`: Mean Squared Error of the model fit.
- `amse::Union{Float64,Int}`: Average Mean Squared Error.
- `fit::Any`: The fitted model object.
- `method::String`: The method used for model fitting (e.g., "Holt's method", "Damped Holt's method").

# See also
[`holt`](@ref), [`ses`](@ref), [`hw`](@ref), [`ets`](@ref), [`forecast`](@ref)

"""
struct Holt
    fitted::AbstractArray
    residuals::AbstractArray
    components::Vector{Any}
    x::AbstractArray
    par::Any
    loglik::Union{Float64,Int}
    initstate::AbstractArray
    states::AbstractArray
    state_names::Any
    SSE::Union{Float64,Int}
    sigma2::Union{Float64,Int}
    m::Int
    lambda::Union{Float64,Bool,Nothing}
    biasadj::Bool
    aic::Union{Float64,Int}
    bic::Union{Float64,Int}
    aicc::Union{Float64,Int}
    mse::Union{Float64,Int}
    amse::Union{Float64,Int}
    fit::Any
    method::String
end

"""
    holt(y; damped=false, initial="optimal", exponential=false, alpha=nothing,
         beta=nothing, phi=nothing, lambda=nothing, biasadj=false,
         options=NelderMeadOptions())
    holt(y, m; damped=false, initial="optimal", exponential=false, alpha=nothing,
         beta=nothing, phi=nothing, lambda=nothing, biasadj=false,
         options=NelderMeadOptions())

Fit Holt's linear trend method to a time series.

Holt's method (also known as double exponential smoothing) extends simple exponential
smoothing to capture linear trends. The method forecasts data with a trend but no
seasonality using two smoothing equations: one for the level and one for the trend.

# Arguments
- `y::AbstractArray`: Time series data to fit.
- `m::Int`: Seasonal period (optional, defaults to 1). Since Holt's method doesn't capture
  seasonality, this parameter is typically omitted or set to 1.

# Keyword Arguments
- `damped::Bool=false`: If `true`, applies damping to the trend component using parameter φ.
  Damped trends prevent forecasts from trending indefinitely into the future.
- `initial::String="optimal"`: Initialization method:
  - `"optimal"`: Uses state-space optimization via ETS framework (default).
  - `"simple"`: Uses conventional Holt-Winters initialization.
  Note: Damped trends require `"optimal"` initialization.
- `exponential::Bool=false`: If `true`, uses exponential (multiplicative) trend instead of additive.
- `alpha::Union{Float64,Nothing}=nothing`: Level smoothing parameter (0 < α < 1).
  If `nothing`, α is estimated from the data.
- `beta::Union{Float64,Nothing}=nothing`: Trend smoothing parameter (0 < β < 1).
  If `nothing`, β is estimated from the data.
- `phi::Union{Float64,Nothing}=nothing`: Damping parameter (0 < φ ≤ 1). Only used when `damped=true`.
  If `nothing`, φ is estimated from the data.
- `lambda::Union{Float64,Bool,Nothing}=nothing`: Box-Cox transformation parameter.
  - `nothing`: No transformation (default).
  - `"auto"` or `true`: Automatically select optimal λ.
  - `Float64`: Use specified λ value.
- `biasadj::Bool=false`: Apply bias adjustment for Box-Cox back-transformation.
- `options::NelderMeadOptions`: Optimization options for parameter estimation.

# Returns
- `Holt`: Fitted Holt model object containing fitted values, residuals, parameters,
  states, and information criteria.

# Model Formulation

## Standard Holt's Method (Additive Trend)
```
yₜ = ℓₜ₋₁ + bₜ₋₁ + εₜ
ℓₜ = α·yₜ + (1-α)(ℓₜ₋₁ + bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)bₜ₋₁
```
where ℓₜ is the level, bₜ is the trend, and εₜ ~ N(0, σ²).

**h-step ahead forecast:** ŷₜ₊ₕ = ℓₜ + h·bₜ

## Damped Trend
```
yₜ = ℓₜ₋₁ + φ·bₜ₋₁ + εₜ
ℓₜ = α·yₜ + (1-α)(ℓₜ₋₁ + φ·bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)φ·bₜ₋₁
```

**h-step ahead forecast:** ŷₜ₊ₕ = ℓₜ + (φ + φ² + ... + φʰ)·bₜ

The damping parameter φ controls how quickly the trend dampens:
- φ = 1: Standard Holt (no damping)
- φ < 1: Damped trend (more conservative long-term forecasts)

## Exponential Trend
```
yₜ = ℓₜ₋₁·bₜ₋₁^φ + εₜ
ℓₜ = α·yₜ + (1-α)·ℓₜ₋₁·bₜ₋₁^φ
bₜ = β(ℓₜ/ℓₜ₋₁) + (1-β)·bₜ₋₁^φ
```

# Examples
```julia
using Durbyn.ExponentialSmoothing

# Simulate data with trend
t = 1:50
y = 100 .+ 2 .* t .+ randn(50) .* 5

# Standard Holt's method (m parameter optional)
fit = holt(y)
println(fit)
fc = forecast(fit, h=10)

# Holt's method with fixed parameters
fit_fixed = holt(y, alpha=0.8, beta=0.2)

# Damped trend (recommended for long-horizon forecasts)
fit_damped = holt(y, damped=true)
fc_damped = forecast(fit_damped, h=24)

# Exponential trend
fit_exp = holt(y, exponential=true)

# With Box-Cox transformation
fit_bc = holt(y, lambda=0.5, biasadj=true)

# Simple initialization
fit_simple = holt(y, initial="simple")

# Can also specify m explicitly (though typically not needed)
fit_explicit = holt(y, 1, damped=true)
```

# When to Use Holt's Method

Use Holt's linear trend method when:
- Data exhibits a clear linear trend
- No seasonal pattern is present
- You need to extrapolate the trend into the future
- The trend is expected to be relatively stable

**Damped trends** are recommended when:
- Long-horizon forecasts are needed (h > 10)
- The trend may not continue indefinitely
- You want more conservative forecasts

**Limitations:**
- Cannot capture seasonality (use `hw()` instead)
- Assumes linear trend (may not fit curved trends well)
- Forecasts can be unrealistic for long horizons without damping

# See also
[`Holt`](@ref), [`ses`](@ref), [`hw`](@ref), [`ets`](@ref), [`forecast`](@ref)

# References
- Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag: New York. http://www.exponentialsmoothing.net.
- Hyndman and Athanasopoulos (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. https://otexts.com/fpp2/
"""
function holt(
    y::AbstractArray;
    damped::Bool = false,
    initial::String = "optimal",
    exponential::Bool = false,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),
)
    holt(y, 1, damped=damped, initial=initial, exponential=exponential,
         alpha=alpha, beta=beta, phi=phi, lambda=lambda, biasadj=biasadj,
         options=options)
end

function holt(
    y::AbstractArray,
    m::Int;
    damped::Bool = false,
    initial::String = "optimal",
    exponential::Bool = false,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),

)

    initial = match_arg(initial, ["optimal", "simple"])
    model = nothing

    if length(y) <= 1
        throw(
            ArgumentError(
                "Holt's method needs at least two observations to estimate trend.",
            ),
        )
    end

    if initial == "optimal" || damped
        if exponential
            model = ets_base_model(
                y,
                m,
                "MMN",
                alpha = alpha,
                beta = beta,
                phi = phi,
                damped = damped,
                opt_crit = "mse",
                lambda = lambda,
                biasadj = biasadj,
                options = options,
            )
        else
            model = ets_base_model(
                y,
                m,
                "AAN",
                alpha = alpha,
                beta = beta,
                phi = phi,
                damped = damped,
                opt_crit = "mse",
                lambda = lambda,
                biasadj = biasadj,
                options = options
            )
        end
    else
        model = holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = false,
            phi = phi,
            exponential = exponential,
            lambda = lambda,
            biasadj = biasadj,
            options = options
        )
    end

    if damped
        method = "Damped Holt's method"
        if initial == "simple"
            @warn "Damped Holt's method requires optimal initialization"
        end
    else
        method = "Holt's method"
    end

    if exponential
        method = method * " with exponential trend"
    end

    return Holt(
        model.fitted,
        model.residuals,
        model.components,
        model.x,
        model.par,
        model.loglik,
        model.initstate,
        model.states,
        model.state_names,
        model.SSE,
        model.sigma2,
        model.m,
        model.lambda,
        model.biasadj,
        model.aic,
        model.bic,
        model.aicc,
        model.mse,
        model.amse,
        model.fit,
        method,
    )
end