"""
    Naive Forecasting Methods

This module provides naive forecasting methods for time series:
- `naive`: Uses the last observation as forecast
- `snaive`: Uses the observation from m periods ago (seasonal naive)
- `rw`/`rwf`: Random walk, optionally with drift

These serve as simple benchmarks for more complex forecasting methods.
"""

import Statistics: var, mean
import Distributions: quantile, Normal

"""
    NaiveFit

Fitted naive forecasting model.

# Fields
- `x::AbstractVector` - Original time series data
- `fitted::AbstractVector{Union{Float64, Missing}}` - In-sample fitted values
- `residuals::AbstractVector{Union{Float64, Missing}}` - In-sample residuals
- `lag::Int` - Lag used for naive forecast (1 for naive/rw, m for snaive)
- `drift::Union{Float64, Nothing}` - Drift coefficient (only for rw with drift)
- `drift_se::Union{Float64, Nothing}` - Standard error of drift (only for rw with drift)
- `sigma2::Float64` - Residual variance
- `m::Int` - Seasonal period
- `method::String` - Method name for display
- `lambda::Union{Nothing, Float64}` - Box-Cox transformation parameter
- `biasadj::Bool` - Whether bias adjustment was applied
"""
struct NaiveFit
    x::AbstractVector
    fitted::AbstractVector{Union{Float64, Missing}}
    residuals::AbstractVector{Union{Float64, Missing}}
    lag::Int
    drift::Union{Float64, Nothing}
    drift_se::Union{Float64, Nothing}
    sigma2::Float64
    m::Int
    method::String
    lambda::Union{Nothing, Float64}
    biasadj::Bool
end

"""
    naive(y::AbstractVector, m::Int=1; lambda=nothing, biasadj::Bool=false)

Fit a naive forecasting model (random walk without drift).

The naive forecast uses the last observed value as the forecast for all future periods.
Formally: y_{T+h|T} = y_T for all h = 1, 2, ...

# Arguments
- `y::AbstractVector` - Time series data
- `m::Int=1` - Seasonal period (stored for reference, not used in naive)

# Keyword Arguments
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Apply bias adjustment for Box-Cox back-transformation

# Returns
`NaiveFit` - Fitted naive model

# Examples
```julia
y = randn(100)
fit = naive(y)
fc = forecast(fit, h=10)
```

# See Also
- [`snaive`](@ref) - Seasonal naive
- [`rw`](@ref) - Random walk with optional drift
"""
function naive(y::AbstractVector, m::Int=1;
               lambda::Union{Nothing, Float64}=nothing,
               biasadj::Bool=false)
    x = copy(y)

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda=lambda)
    end

    n = length(y)
    lag = 1

    # Fitted values: y_{t} = y_{t-1}
    fitted = Vector{Union{Float64, Missing}}(missing, n)
    for t in 2:n
        fitted[t] = y[t-1]
    end

    # Residuals
    residuals = Vector{Union{Float64, Missing}}(missing, n)
    for t in 2:n
        residuals[t] = y[t] - fitted[t]
    end

    # Residual variance (using non-missing values)
    valid_residuals = collect(skipmissing(residuals))
    sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

    return NaiveFit(x, fitted, residuals, lag, nothing, nothing, sigma2, m,
                    "Naive method", lambda, biasadj)
end

"""
    snaive(y::AbstractVector, m::Int; lambda=nothing, biasadj::Bool=false)

Fit a seasonal naive forecasting model.

The seasonal naive method uses the observation from m periods ago as the forecast.
Formally: y_{T+h|T} = y_{T+h-m*k} where k = ceil((h-1)/m) + 1

# Arguments
- `y::AbstractVector` - Time series data
- `m::Int` - Seasonal period (required)

# Keyword Arguments
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Apply bias adjustment for Box-Cox back-transformation

# Returns
`NaiveFit` - Fitted seasonal naive model

# Examples
```julia
# Monthly data with yearly seasonality
y = randn(120)
fit = snaive(y, 12)
fc = forecast(fit, h=24)
```

# See Also
- [`naive`](@ref) - Non-seasonal naive
- [`rw`](@ref) - Random walk with optional drift
"""
function snaive(y::AbstractVector, m::Int;
                lambda::Union{Nothing, Float64}=nothing,
                biasadj::Bool=false)
    m >= 1 || throw(ArgumentError("Seasonal period m must be >= 1, got $m"))

    x = copy(y)

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda=lambda)
    end

    n = length(y)
    n > m || throw(ArgumentError("Time series length ($n) must be greater than seasonal period ($m)"))
    lag = m

    # Fitted values: y_{t} = y_{t-m}
    fitted = Vector{Union{Float64, Missing}}(missing, n)
    for t in (m+1):n
        fitted[t] = y[t-m]
    end

    # Residuals
    residuals = Vector{Union{Float64, Missing}}(missing, n)
    for t in (m+1):n
        residuals[t] = y[t] - fitted[t]
    end

    # Residual variance
    valid_residuals = collect(skipmissing(residuals))
    sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

    return NaiveFit(x, fitted, residuals, lag, nothing, nothing, sigma2, m,
                    "Seasonal naive method", lambda, biasadj)
end

"""
    rw(y::AbstractVector, m::Int=1; drift::Bool=false, lambda=nothing, biasadj::Bool=false)

Fit a random walk forecasting model, optionally with drift.

Without drift, this is equivalent to `naive()`. With drift, the forecast includes
a linear trend based on the average change in the historical data.

Without drift: y_{T+h|T} = y_T
With drift: y_{T+h|T} = y_T + h * drift, where drift = (y_T - y_1) / (T-1)

# Arguments
- `y::AbstractVector` - Time series data
- `m::Int=1` - Seasonal period (stored for reference)

# Keyword Arguments
- `drift::Bool=false` - Include drift term
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Apply bias adjustment for Box-Cox back-transformation

# Returns
`NaiveFit` - Fitted random walk model

# Examples
```julia
# Random walk without drift (same as naive)
y = cumsum(randn(100))
fit1 = rw(y)

# Random walk with drift
fit2 = rw(y, drift=true)
fc = forecast(fit2, h=10)
```

# See Also
- [`naive`](@ref) - Naive method (equivalent to rw without drift)
- [`snaive`](@ref) - Seasonal naive
- [`rwf`](@ref) - Alias for `rw`
"""
function rw(y::AbstractVector, m::Int=1;
            drift::Bool=false,
            lambda::Union{Nothing, Float64}=nothing,
            biasadj::Bool=false)
    x = copy(y)

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda=lambda)
    end

    n = length(y)
    lag = 1

    if drift
        # Random walk with drift
        # Drift = average change = (y_T - y_1) / (T-1)
        drift_val = (y[n] - y[1]) / (n - 1)

        # Fitted values: y_{t} = y_{t-1} + drift
        fitted = Vector{Union{Float64, Missing}}(missing, n)
        for t in 2:n
            fitted[t] = y[t-1] + drift_val
        end

        # Residuals
        residuals = Vector{Union{Float64, Missing}}(missing, n)
        for t in 2:n
            residuals[t] = y[t] - fitted[t]
        end

        # Residual variance
        valid_residuals = collect(skipmissing(residuals))
        sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

        # Standard error of drift
        # SE(drift) = sigma / sqrt(T-1)
        drift_se = sqrt(sigma2 / (n - 1))

        method_name = "Random walk with drift"

        return NaiveFit(x, fitted, residuals, lag, drift_val, drift_se, sigma2, m,
                        method_name, lambda, biasadj)
    else
        # Random walk without drift (same as naive)
        fitted = Vector{Union{Float64, Missing}}(missing, n)
        for t in 2:n
            fitted[t] = y[t-1]
        end

        residuals = Vector{Union{Float64, Missing}}(missing, n)
        for t in 2:n
            residuals[t] = y[t] - fitted[t]
        end

        valid_residuals = collect(skipmissing(residuals))
        sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

        return NaiveFit(x, fitted, residuals, lag, nothing, nothing, sigma2, m,
                        "Random walk method", lambda, biasadj)
    end
end

"""
    rwf

Alias for `rw` (random walk forecast).
"""
const rwf = rw

function Base.show(io::IO, fit::NaiveFit)
    println(io, fit.method)
    println(io, "-------------------------------")
    println(io, "Lag: ", fit.lag)
    if !isnothing(fit.drift)
        println(io, "Drift: ", round(fit.drift, digits=6))
        println(io, "Drift SE: ", round(fit.drift_se, digits=6))
    end
    println(io, "Residual variance: ", round(fit.sigma2, digits=6))
    println(io, "Series length: ", length(fit.x))
    if !isnothing(fit.lambda)
        println(io, "Box-Cox lambda: ", fit.lambda)
    end
end
