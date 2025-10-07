export holt_winters, HoltWinters

"""
    HoltWinters

Holt-Winters' seasonal method model output structure.

Holt-Winters' method extends Holt's linear trend method to capture seasonal patterns
in time series data. It uses three smoothing parameters: α for the level, β for
the trend, and γ for the seasonal component. The method supports both additive and
multiplicative seasonality, and can optionally include damping (φ) and exponential trends.

# Fields
- `fitted::AbstractArray`: The fitted values (one-step ahead predictions) at each time point.
- `residuals::AbstractArray`: The residuals (observed - fitted values).
- `components::Vector{Any}`: Model components (level, trend, and seasonal).
- `x::AbstractArray`: The original time series data.
- `par::Any`: Dictionary containing model parameters (alpha, beta, gamma, phi if damped).
- `loglik::Union{Float64,Int}`: Log-likelihood of the model.
- `initstate::AbstractArray`: Initial state estimates (initial level, trend, and seasonal indices).
- `states::AbstractArray`: Level, trend, and seasonal estimates over time.
- `state_names::Any`: Names of the state variables.
- `SSE::Union{Float64,Int}`: Sum of squared errors, a measure of model fit.
- `sigma2::Union{Float64,Int}`: Residual variance (σ²).
- `m::Int`: Seasonal period (e.g., 12 for monthly data, 4 for quarterly).
- `lambda::Union{Float64,Bool,Nothing}`: Box-Cox transformation parameter (nothing if not used).
- `biasadj::Bool`: Boolean flag indicating whether bias adjustment was applied.
- `aic::Union{Float64,Int}`: Akaike Information Criterion for model selection.
- `bic::Union{Float64,Int}`: Bayesian Information Criterion for model selection.
- `aicc::Union{Float64,Int}`: Corrected AIC for small sample sizes.
- `mse::Union{Float64,Int}`: Mean Squared Error of the model fit.
- `amse::Union{Float64,Int}`: Average Mean Squared Error.
- `fit::Any`: The fitted model object.
- `method::String`: The method used for model fitting (e.g., "Holt-Winters' additive method").

# See also
[`holt_winters`](@ref), [`holt`](@ref), [`ses`](@ref), [`ets`](@ref), [`forecast`](@ref)

"""
struct HoltWinters
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
    holt_winters(y, m; seasonal="additive", damped=false, initial="optimal",
                 exponential=false, alpha=nothing, beta=nothing, gamma=nothing,
                 phi=nothing, lambda=nothing, biasadj=false,
                 options=NelderMeadOptions())

Fit Holt-Winters' seasonal method to a time series.

Holt-Winters' method (also known as triple exponential smoothing) extends Holt's linear
trend method to capture seasonal patterns. The method uses three smoothing equations: one
for the level, one for the trend, and one for the seasonal component.

# Arguments
- `y::AbstractArray`: Time series data to fit.
- `m::Int`: Seasonal period (e.g., 12 for monthly data, 4 for quarterly, 7 for daily with weekly seasonality).

# Keyword Arguments
- `seasonal::String="additive"`: Type of seasonal component:
  - `"additive"`: Seasonal variations are constant across levels (default).
  - `"multiplicative"`: Seasonal variations change proportionally with level.
- `damped::Bool=false`: If `true`, applies damping to the trend component using parameter φ.
  Damped trends prevent forecasts from trending indefinitely into the future.
- `initial::String="optimal"`: Initialization method:
  - `"optimal"`: Uses state-space optimization via ETS framework (default).
  - `"simple"`: Uses conventional Holt-Winters initialization.
  Note: Damped trends require `"optimal"` initialization.
- `exponential::Bool=false`: If `true`, uses exponential (multiplicative) trend instead of additive.
  Cannot be combined with additive seasonality.
- `alpha::Union{Float64,Nothing}=nothing`: Level smoothing parameter (0 < α < 1).
  If `nothing`, α is estimated from the data.
- `beta::Union{Float64,Nothing}=nothing`: Trend smoothing parameter (0 < β < 1).
  If `nothing`, β is estimated from the data.
- `gamma::Union{Float64,Nothing}=nothing`: Seasonal smoothing parameter (0 < γ < 1).
  If `nothing`, γ is estimated from the data.
- `phi::Union{Float64,Nothing}=nothing`: Damping parameter (0 < φ ≤ 1). Only used when `damped=true`.
  If `nothing`, φ is estimated from the data.
- `lambda::Union{Float64,Bool,Nothing}=nothing`: Box-Cox transformation parameter.
  - `nothing`: No transformation (default).
  - `"auto"` or `true`: Automatically select optimal λ.
  - `Float64`: Use specified λ value.
- `biasadj::Bool=false`: Apply bias adjustment for Box-Cox back-transformation.
- `options::NelderMeadOptions`: Optimization options for parameter estimation.

# Returns
- `HoltWinters`: Fitted Holt-Winters model object containing fitted values, residuals,
  parameters, states, and information criteria.

# Model Formulation

## Additive Seasonality (seasonal="additive")
```
yₜ = ℓₜ₋₁ + bₜ₋₁ + sₜ₋ₘ + εₜ
ℓₜ = α(yₜ - sₜ₋ₘ) + (1-α)(ℓₜ₋₁ + bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)bₜ₋₁
sₜ = γ(yₜ - ℓₜ₋₁ - bₜ₋₁) + (1-γ)sₜ₋ₘ
```
where ℓₜ is the level, bₜ is the trend, sₜ is the seasonal component, and εₜ ~ N(0, σ²).

**h-step ahead forecast:** ŷₜ₊ₕ = ℓₜ + h·bₜ + sₜ₊ₕ₋ₘ

## Multiplicative Seasonality (seasonal="multiplicative")
```
yₜ = (ℓₜ₋₁ + bₜ₋₁)·sₜ₋ₘ + εₜ
ℓₜ = α(yₜ/sₜ₋ₘ) + (1-α)(ℓₜ₋₁ + bₜ₋₁)
bₜ = β(ℓₜ - ℓₜ₋₁) + (1-β)bₜ₋₁
sₜ = γ(yₜ/(ℓₜ₋₁ + bₜ₋₁)) + (1-γ)sₜ₋ₘ
```

**h-step ahead forecast:** ŷₜ₊ₕ = (ℓₜ + h·bₜ)·sₜ₊ₕ₋ₘ

## Damped Trend
When `damped=true`, the trend component is modified:
```
yₜ = ℓₜ₋₁ + φ·bₜ₋₁ + sₜ₋ₘ + εₜ  (additive)
yₜ = (ℓₜ₋₁ + φ·bₜ₋₁)·sₜ₋ₘ + εₜ  (multiplicative)
```

**h-step ahead forecast:** ŷₜ₊ₕ = ℓₜ + (φ + φ² + ... + φʰ)·bₜ + sₜ₊ₕ₋ₘ

The damping parameter φ controls how quickly the trend dampens:
- φ = 1: Standard Holt-Winters (no damping)
- φ < 1: Damped trend (more conservative long-term forecasts)

# Examples
```julia
using Durbyn.ExponentialSmoothing

# Simulate seasonal data (quarterly with trend)
t = 1:100
seasonal_pattern = repeat([10, -5, -8, 3], 25)
y = 100 .+ 0.5 .* t .+ seasonal_pattern .+ randn(100) .* 3

# Additive Holt-Winters (default)
fit_add = holt_winters(y, 4)
println(fit_add)
fc_add = forecast(fit_add, h=12)

# Multiplicative seasonality
fit_mult = holt_winters(y, 4, seasonal="multiplicative")

# Damped trend (recommended for long-horizon forecasts)
fit_damped = holt_winters(y, 4, damped=true)
fc_damped = forecast(fit_damped, h=24)

# With fixed parameters
fit_fixed = holt_winters(y, 4, alpha=0.7, beta=0.1, gamma=0.3)

# Exponential trend with multiplicative seasonality
fit_exp = holt_winters(y, 4, seasonal="multiplicative", exponential=true)

# With Box-Cox transformation
fit_bc = holt_winters(y, 4, lambda=0.5, biasadj=true)

# Simple initialization
fit_simple = holt_winters(y, 4, initial="simple")

# Monthly data example
monthly_data = rand(120)  # 10 years of monthly data
fit_monthly = holt_winters(monthly_data, 12)
```

# When to Use Holt-Winters' Method

Use Holt-Winters' seasonal method when:
- Data exhibits both trend and seasonal patterns
- Seasonal pattern is relatively stable over time
- You need to forecast future values accounting for seasonality

**Choose additive seasonality when:**
- Seasonal variations are roughly constant regardless of level
- Seasonal fluctuations don't increase/decrease with the level of the series

**Choose multiplicative seasonality when:**
- Seasonal variations change proportionally with the level
- Seasonal fluctuations increase/decrease as the series level changes

**Damped trends** are recommended when:
- Long-horizon forecasts are needed (h > 10)
- The trend may not continue indefinitely
- You want more conservative forecasts

**Limitations:**
- Requires at least m+3 observations to fit
- Assumes seasonal pattern repeats with period m
- May not handle changing seasonal patterns well (consider ETS or more advanced models)

# See also
[`HoltWinters`](@ref), [`holt`](@ref), [`ses`](@ref), [`ets`](@ref), [`forecast`](@ref)

# References
- Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D. (2008) Forecasting with exponential smoothing: the state space approach, Springer-Verlag: New York. http://www.exponentialsmoothing.net.
- Hyndman and Athanasopoulos (2018) Forecasting: principles and practice, 2nd edition, OTexts: Melbourne, Australia. https://otexts.com/fpp2/
"""
function holt_winters(
    y::AbstractArray,
    m::Int;
    seasonal::String = "additive",
    damped::Bool = false,
    initial::String = "optimal",
    exponential::Bool = false,
    alpha::Union{Float64,Bool,Nothing} = nothing,
    beta::Union{Float64,Bool,Nothing} = nothing,
    gamma::Union{Float64,Bool,Nothing} = nothing,
    phi::Union{Float64,Bool,Nothing} = nothing,
    lambda::Union{Float64,Bool,Nothing} = nothing,
    biasadj::Bool = false,
    options::NelderMeadOptions = NelderMeadOptions(),
)

    initial = match_arg(initial, ["optimal", "simple"])
    seasonal = match_arg(seasonal, ["additive", "multiplicative"])

    if seasonal == "additive" && exponential
        throw(
            ArgumentError("Additive seasonality cannot be combined with exponential trend"),
        )
    end

    if m <= 1
        throw(ArgumentError("The time series should have frequency greater than 1."))
    end

    if length(y) <= m + 3
        throw(
            ArgumentError("I need at least $(m + 3) observations to estimate seasonality."),
        )
    end

    if initial == "optimal" || damped
        if seasonal == "additive" && exponential
            error("Forbidden model combination")
        end
        model_type = if seasonal == "additive"
            "AAA"
        elseif exponential
            "MMM"
        else
            "MAM"
        end

        model = ets_base_model(
            y,
            m,
            model_type,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            damped = damped,
            opt_crit = "mse",
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )

    else
        model = holt_winters_conventional(
            y,
            m,
            alpha = alpha,
            beta = beta,
            gamma = gamma,
            phi = phi,
            seasonal = seasonal,
            exponential = exponential,
            lambda = lambda,
            biasadj = biasadj,
            options = options,
        )
    end

    method = damped ? "Damped Holt-Winters'" : "Holt-Winters'"
    method *= seasonal == "additive" ? " additive method" : " multiplicative method"
    if exponential
        method *= " with exponential trend"
    end

    if damped && initial == "simple"
        @warn "Damped Holt-Winters' method requires optimal initialization"
    end

    # Handle fields that don't exist in HoltWintersConventional
    loglik = hasfield(typeof(model), :loglik) ? model.loglik : NaN
    aic = hasfield(typeof(model), :aic) ? model.aic : NaN
    bic = hasfield(typeof(model), :bic) ? model.bic : NaN
    aicc = hasfield(typeof(model), :aicc) ? model.aicc : NaN
    mse = hasfield(typeof(model), :mse) ? model.mse : NaN
    amse = hasfield(typeof(model), :amse) ? model.amse : NaN
    fit = hasfield(typeof(model), :fit) ? model.fit : Float64[]

    return HoltWinters(
        model.fitted,
        model.residuals,
        model.components,
        model.x,
        model.par,
        loglik,
        model.initstate,
        model.states,
        model.state_names,
        model.SSE,
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
        method,
    )
end