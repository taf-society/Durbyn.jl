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
- `fitted::AbstractVector{Union{Float64, Missing}}` - In-sample fitted values (on original scale)
- `residuals::AbstractVector{Union{Float64, Missing}}` - In-sample residuals (on original scale)
- `lag::Int` - Lag used for naive forecast (1 for naive/rw, m for snaive)
- `drift::Union{Float64, Nothing}` - Drift coefficient (only for rw with drift, on transformed scale if lambda used)
- `drift_se::Union{Float64, Nothing}` - Standard error of drift (only for rw with drift)
- `sigma2::Float64` - Residual variance (on transformed scale if lambda used)
- `m::Int` - Seasonal period
- `method::String` - Method name for display
- `lambda::Union{Nothing, Float64}` - Box-Cox transformation parameter
- `biasadj::Bool` - Whether bias adjustment was applied
- `y_transformed::Union{Nothing, AbstractVector}` - Transformed data (if lambda used)
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
    y_transformed::Union{Nothing, AbstractVector}
end

"""
    naive(y::AbstractVector, m::Int=1; lambda=nothing, biasadj::Bool=false)

Fit a naive forecasting model (random walk without drift).

The naive forecast uses the last observed value as the forecast for all future periods.
Formally: y_{T+h|T} = y_T for all h = 1, 2, ...

# Arguments
- `y::AbstractVector` - Time series data (must have length >= 2)
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
    n = length(y)
    n >= 2 || throw(ArgumentError("Time series must have at least 2 observations, got $n"))

    x = collect(Float64, y)
    y_transformed = nothing

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        y_trans, lambda = box_cox(x, m, lambda=lambda)
        y_transformed = y_trans
    else
        y_trans = x
    end

    lag = 1

    # Fitted values on transformed scale: y_{t} = y_{t-1}
    fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(fitted_trans, missing)
    for t in 2:n
        fitted_trans[t] = y_trans[t-1]
    end

    # Residuals on transformed scale
    residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(residuals_trans, missing)
    for t in 2:n
        residuals_trans[t] = y_trans[t] - fitted_trans[t]
    end

    # Residual variance on transformed scale
    valid_residuals = collect(skipmissing(residuals_trans))
    sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

    # Convert fitted/residuals back to original scale for storage
    if !isnothing(lambda)
        fitted = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted, missing)
        residuals = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals, missing)
        for t in 2:n
            fitted[t] = inv_box_cox([fitted_trans[t]]; lambda=lambda)[1]
            residuals[t] = x[t] - fitted[t]
        end
    else
        fitted = fitted_trans
        residuals = residuals_trans
    end

    return NaiveFit(x, fitted, residuals, lag, nothing, nothing, sigma2, m,
                    "Naive method", lambda, biasadj, y_transformed)
end

"""
    snaive(y::AbstractVector, m::Int; lambda=nothing, biasadj::Bool=false)

Fit a seasonal naive forecasting model.

The seasonal naive method uses the observation from m periods ago as the forecast.
Formally: y_{T+h|T} = y_{T+h-m*k} where k = ceil((h-1)/m) + 1

# Arguments
- `y::AbstractVector` - Time series data
- `m::Int` - Seasonal period (required, must be >= 1)

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

    n = length(y)
    n > m || throw(ArgumentError("Time series length ($n) must be greater than seasonal period ($m)"))

    x = collect(Float64, y)
    y_transformed = nothing

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        y_trans, lambda = box_cox(x, m, lambda=lambda)
        y_transformed = y_trans
    else
        y_trans = x
    end

    lag = m

    # Fitted values on transformed scale: y_{t} = y_{t-m}
    fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(fitted_trans, missing)
    for t in (m+1):n
        fitted_trans[t] = y_trans[t-m]
    end

    # Residuals on transformed scale
    residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(residuals_trans, missing)
    for t in (m+1):n
        residuals_trans[t] = y_trans[t] - fitted_trans[t]
    end

    # Residual variance on transformed scale
    valid_residuals = collect(skipmissing(residuals_trans))
    sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

    # Convert fitted/residuals back to original scale for storage
    if !isnothing(lambda)
        fitted = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted, missing)
        residuals = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals, missing)
        for t in (m+1):n
            fitted[t] = inv_box_cox([fitted_trans[t]]; lambda=lambda)[1]
            residuals[t] = x[t] - fitted[t]
        end
    else
        fitted = fitted_trans
        residuals = residuals_trans
    end

    return NaiveFit(x, fitted, residuals, lag, nothing, nothing, sigma2, m,
                    "Seasonal naive method", lambda, biasadj, y_transformed)
end

"""
    rw(y::AbstractVector, m::Int=1; drift::Bool=false, lambda=nothing, biasadj::Bool=false)

Fit a random walk forecasting model, optionally with drift.

Without drift, this is equivalent to `naive()`. With drift, the forecast includes
a linear trend based on the average change in the historical data.

Without drift: y_{T+h|T} = y_T
With drift: y_{T+h|T} = y_T + h * drift, where drift = (y_T - y_1) / (T-1)

# Arguments
- `y::AbstractVector` - Time series data (must have length >= 2)
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
    n = length(y)
    n >= 2 || throw(ArgumentError("Time series must have at least 2 observations, got $n"))

    x = collect(Float64, y)
    y_transformed = nothing

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        y_trans, lambda = box_cox(x, m, lambda=lambda)
        y_transformed = y_trans
    else
        y_trans = x
    end

    lag = 1

    if drift
        # Random walk with drift
        # Drift = average change = (y_T - y_1) / (T-1) on transformed scale
        drift_val = (y_trans[n] - y_trans[1]) / (n - 1)

        # Fitted values on transformed scale: y_{t} = y_{t-1} + drift
        fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted_trans, missing)
        for t in 2:n
            fitted_trans[t] = y_trans[t-1] + drift_val
        end

        # Residuals on transformed scale
        residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals_trans, missing)
        for t in 2:n
            residuals_trans[t] = y_trans[t] - fitted_trans[t]
        end

        # Residual variance on transformed scale
        valid_residuals = collect(skipmissing(residuals_trans))
        sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

        # Standard error of drift
        # SE(drift) = sigma / sqrt(T-1)
        drift_se = sqrt(sigma2 / (n - 1))

        method_name = "Random walk with drift"

        # Convert fitted/residuals back to original scale for storage
        if !isnothing(lambda)
            fitted = Vector{Union{Float64, Missing}}(undef, n)
            fill!(fitted, missing)
            residuals = Vector{Union{Float64, Missing}}(undef, n)
            fill!(residuals, missing)
            for t in 2:n
                fitted[t] = inv_box_cox([fitted_trans[t]]; lambda=lambda)[1]
                residuals[t] = x[t] - fitted[t]
            end
        else
            fitted = fitted_trans
            residuals = residuals_trans
        end

        return NaiveFit(x, fitted, residuals, lag, drift_val, drift_se, sigma2, m,
                        method_name, lambda, biasadj, y_transformed)
    else
        # Random walk without drift (same as naive)
        fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted_trans, missing)
        for t in 2:n
            fitted_trans[t] = y_trans[t-1]
        end

        residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals_trans, missing)
        for t in 2:n
            residuals_trans[t] = y_trans[t] - fitted_trans[t]
        end

        valid_residuals = collect(skipmissing(residuals_trans))
        sigma2 = isempty(valid_residuals) ? 0.0 : var(valid_residuals, corrected=true)

        # Convert fitted/residuals back to original scale for storage
        if !isnothing(lambda)
            fitted = Vector{Union{Float64, Missing}}(undef, n)
            fill!(fitted, missing)
            residuals = Vector{Union{Float64, Missing}}(undef, n)
            fill!(residuals, missing)
            for t in 2:n
                fitted[t] = inv_box_cox([fitted_trans[t]]; lambda=lambda)[1]
                residuals[t] = x[t] - fitted[t]
            end
        else
            fitted = fitted_trans
            residuals = residuals_trans
        end

        return NaiveFit(x, fitted, residuals, lag, nothing, nothing, sigma2, m,
                        "Random walk method", lambda, biasadj, y_transformed)
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
