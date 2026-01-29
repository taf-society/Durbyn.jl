"""
    Naive Forecasting Methods

This module provides naive forecasting methods for time series:
- `naive`: Uses the last observation as forecast
- `snaive`: Uses the observation from m periods ago (seasonal naive)
- `rw`/`rwf`: Random walk, optionally with drift

These serve as simple benchmarks for more complex forecasting methods.
"""

import Statistics: var, mean, std
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
- `y::AbstractVector` - Time series data (must have at least 2 non-missing observations)
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
               lambda::Union{Nothing, Float64, String}=nothing,
               biasadj::Bool=false)
    n = length(y)
    n >= 2 || throw(ArgumentError("Time series must have at least 2 observations, got $n"))

    # Convert to Float64, preserving structure (missings become NaN for processing)
    x = Vector{Float64}(undef, n)
    for i in 1:n
        x[i] = ismissing(y[i]) ? NaN : Float64(y[i])
    end

    # Count non-missing for validation
    n_valid = count(!isnan, x)
    n_valid >= 2 || throw(ArgumentError("Time series must have at least 2 non-missing observations, got $n_valid"))

    y_transformed = nothing

    # Apply Box-Cox transformation if specified (only to non-missing values)
    if !isnothing(lambda)
        # Get non-missing values
        valid_mask = .!isnan.(x)

        # Pre-filter values based on lambda:
        # - lambda = "auto": need positive values for lambda estimation
        # - lambda <= 0: requires positive values (log for lambda=0, or domain restriction for lambda<0)
        # - lambda > 0: signed transform handles negatives AND zeros
        #   For x=0: (sign(0) * |0|^lambda - 1) / lambda = -1/lambda (finite)
        if lambda isa String
            # Auto lambda - need positive values for estimation
            transform_mask = valid_mask .& (x .> 0)
        elseif lambda <= 0
            transform_mask = valid_mask .& (x .> 0)
            n_invalid = count(valid_mask) - count(transform_mask)
            if n_invalid > 0
                @warn "$n_invalid non-positive value(s) invalid for Box-Cox transformation with lambda=$lambda, treated as missing"
            end
        else
            # For lambda > 0, signed transform handles negatives and zeros (like R)
            transform_mask = valid_mask
        end

        x_to_transform = x[transform_mask]
        count(transform_mask) >= 2 || throw(ArgumentError("Less than 2 valid observations for Box-Cox transformation with lambda=$lambda"))

        x_trans_values, lambda = box_cox(x_to_transform, m, lambda=lambda)

        # Check for any NaN/Inf introduced by transformation itself
        bc_invalid = .!isfinite.(x_trans_values)
        n_bc_invalid = count(bc_invalid)
        if n_bc_invalid > 0
            @warn "$n_bc_invalid additional value(s) invalid after Box-Cox transformation"
            x_trans_values[bc_invalid] .= NaN
        end

        # Create transformed array with NaN for missing/invalid positions
        y_trans = fill(NaN, n)
        y_trans[transform_mask] = x_trans_values
        y_transformed = copy(y_trans)

        # Validate enough valid values remain after transformation
        n_valid_trans = count(isfinite, y_trans)
        n_valid_trans >= 2 || throw(ArgumentError("Less than 2 valid observations remain after Box-Cox transformation with lambda=$lambda"))
    else
        y_trans = x
    end

    lag = 1

    # Fitted values on transformed scale: y_{t} = y_{t-1}
    # R forward-fills missing fitted values with earlier lagged values
    fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(fitted_trans, missing)
    for t in 2:n
        if isfinite(y_trans[t-1])
            fitted_trans[t] = y_trans[t-1]
        end
    end

    # Forward-fill missing fitted values (like R's lagwalk)
    # If fitted[t] is missing and y[t] is not missing, fill with previous fitted value
    for t in 2:n
        if ismissing(fitted_trans[t]) && !isnan(x[t])
            # Find most recent non-missing fitted value
            for k in (t-1):-1:2
                if !ismissing(fitted_trans[k])
                    fitted_trans[t] = fitted_trans[k]
                    break
                end
            end
        end
    end

    # Residuals on transformed scale (only when both fitted and actual are valid)
    residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(residuals_trans, missing)
    for t in 2:n
        if !ismissing(fitted_trans[t]) && isfinite(y_trans[t])
            residuals_trans[t] = y_trans[t] - fitted_trans[t]
        end
    end

    # Residual variance on transformed scale
    # R uses two different variance measures:
    # 1. sigma2 = mean(res^2) for prediction intervals (MSE)
    # 2. var(res) (centered variance) for biasadj in InvBoxCox
    valid_residuals = collect(skipmissing(residuals_trans))
    n_resid = length(valid_residuals)
    if n_resid == 0
        sigma2 = 0.0
        biasadj_var = 0.0
        @warn "No valid residuals available for variance estimation, using sigma2=0"
    else
        # sigma2 = MSE for prediction intervals (R: sigma <- sqrt(mean(res^2, na.rm=TRUE)))
        sigma2 = mean(valid_residuals .^ 2)
        # biasadj_var = centered variance for bias adjustment (R: var(res))
        biasadj_var = n_resid > 1 ? var(valid_residuals) : 0.0
    end

    # Convert fitted/residuals back to original scale for storage
    if !isnothing(lambda)
        fitted = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted, missing)
        residuals = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals, missing)

        # Batch back-transform non-missing fitted values
        # R uses var(res) (centered variance) for biasadj: InvBoxCox(fitted, lambda, biasadj, var(res))
        fitted_indices = findall(!ismissing, fitted_trans)
        if !isempty(fitted_indices)
            fitted_vals = Float64[fitted_trans[i] for i in fitted_indices]
            if biasadj && biasadj_var > 0
                fitted_back = inv_box_cox(fitted_vals; lambda=lambda, biasadj=true, fvar=biasadj_var)
            else
                fitted_back = inv_box_cox(fitted_vals; lambda=lambda)
            end
            for (j, i) in enumerate(fitted_indices)
                fitted[i] = fitted_back[j]
            end
        end
        # R's lagwalk: residuals stay on TRANSFORMED scale (res = y - fitted where both are transformed)
        residuals = residuals_trans
    else
        fitted = fitted_trans
        residuals = residuals_trans
    end

    # Store x with NaN converted back to original form for consistency
    x_store = Vector{Float64}(undef, n)
    for i in 1:n
        x_store[i] = ismissing(y[i]) ? NaN : Float64(y[i])
    end

    return NaiveFit(x_store, fitted, residuals, lag, nothing, nothing, sigma2, m,
                    "Naive method", lambda, biasadj, y_transformed)
end

"""
    snaive(y::AbstractVector, m::Int; lambda=nothing, biasadj::Bool=false)

Fit a seasonal naive forecasting model.

The seasonal naive method uses the observation from m periods ago as the forecast.
This is equivalent to an ARIMA(0,0,0)(0,1,0)_m model.

Formally: y_{T+h|T} = y_{T+h-m*k} where k = ceil(h/m)

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

# Quarterly data
y_quarterly = randn(20)
fit = snaive(y_quarterly, 4)
fc = forecast(fit, h=8)
```

# See Also
- [`naive`](@ref) - Non-seasonal naive
- [`rw`](@ref) - Random walk with optional drift
"""
function snaive(y::AbstractVector, m::Int;
                lambda::Union{Nothing, Float64, String}=nothing,
                biasadj::Bool=false)
    m >= 1 || throw(ArgumentError("Seasonal period m must be >= 1, got $m"))

    n = length(y)
    n > m || throw(ArgumentError("Time series length ($n) must be greater than seasonal period ($m)"))

    # Convert to Float64, preserving structure (missings become NaN for processing)
    x = Vector{Float64}(undef, n)
    for i in 1:n
        x[i] = ismissing(y[i]) ? NaN : Float64(y[i])
    end

    # Count non-missing for validation
    n_valid = count(!isnan, x)
    n_valid > m || throw(ArgumentError("Number of non-missing observations ($n_valid) must be greater than seasonal period ($m)"))

    y_transformed = nothing

    # Apply Box-Cox transformation if specified (only to non-missing values)
    if !isnothing(lambda)
        # Get non-missing values
        valid_mask = .!isnan.(x)

        # Pre-filter values based on lambda:
        # - lambda = "auto": need positive values for lambda estimation
        # - lambda <= 0: requires positive values (log for lambda=0, or domain restriction for lambda<0)
        # - lambda > 0: signed transform handles negatives AND zeros
        #   For x=0: (sign(0) * |0|^lambda - 1) / lambda = -1/lambda (finite)
        if lambda isa String
            # Auto lambda - need positive values for estimation
            transform_mask = valid_mask .& (x .> 0)
        elseif lambda <= 0
            transform_mask = valid_mask .& (x .> 0)
            n_invalid = count(valid_mask) - count(transform_mask)
            if n_invalid > 0
                @warn "$n_invalid non-positive value(s) invalid for Box-Cox transformation with lambda=$lambda, treated as missing"
            end
        else
            # For lambda > 0, signed transform handles negatives and zeros (like R)
            transform_mask = valid_mask
        end

        x_to_transform = x[transform_mask]
        count(transform_mask) > m || throw(ArgumentError("$(count(transform_mask)) valid observations for Box-Cox, need more than m=$m"))

        x_trans_values, lambda = box_cox(x_to_transform, m, lambda=lambda)

        # Check for any NaN/Inf introduced by transformation itself
        bc_invalid = .!isfinite.(x_trans_values)
        n_bc_invalid = count(bc_invalid)
        if n_bc_invalid > 0
            @warn "$n_bc_invalid additional value(s) invalid after Box-Cox transformation"
            x_trans_values[bc_invalid] .= NaN
        end

        y_trans = fill(NaN, n)
        y_trans[transform_mask] = x_trans_values
        y_transformed = copy(y_trans)

        # Validate enough valid values remain after transformation
        n_valid_trans = count(isfinite, y_trans)
        n_valid_trans > m || throw(ArgumentError("$n_valid_trans valid observations remain after Box-Cox transformation, need more than m=$m"))
    else
        y_trans = x
    end

    lag = m

    # Fitted values on transformed scale: y_{t} = y_{t-m}
    # R forward-fills missing fitted values with earlier lagged values
    fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(fitted_trans, missing)
    for t in (m+1):n
        if isfinite(y_trans[t-m])
            fitted_trans[t] = y_trans[t-m]
        end
    end

    # Forward-fill missing fitted values (like R's lagwalk)
    # If fitted[t] is missing and y[t] is not missing, fill with value from m periods earlier
    for t in (m+1):n
        if ismissing(fitted_trans[t]) && !isnan(x[t])
            # Find most recent non-missing fitted value at same seasonal position
            for k in (t-m):-m:(m+1)
                if !ismissing(fitted_trans[k])
                    fitted_trans[t] = fitted_trans[k]
                    break
                end
            end
        end
    end

    # Residuals on transformed scale (only when both fitted and actual are valid)
    residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
    fill!(residuals_trans, missing)
    for t in (m+1):n
        if !ismissing(fitted_trans[t]) && isfinite(y_trans[t])
            residuals_trans[t] = y_trans[t] - fitted_trans[t]
        end
    end

    # Residual variance on transformed scale
    # R uses two different variance measures:
    # 1. sigma2 = mean(res^2) for prediction intervals (MSE)
    # 2. var(res) (centered variance) for biasadj in InvBoxCox
    valid_residuals = collect(skipmissing(residuals_trans))
    n_resid = length(valid_residuals)
    if n_resid == 0
        sigma2 = 0.0
        biasadj_var = 0.0
        @warn "No valid residuals available for variance estimation, using sigma2=0"
    else
        # sigma2 = MSE for prediction intervals (R: sigma <- sqrt(mean(res^2, na.rm=TRUE)))
        sigma2 = mean(valid_residuals .^ 2)
        # biasadj_var = centered variance for bias adjustment (R: var(res))
        biasadj_var = n_resid > 1 ? var(valid_residuals) : 0.0
    end

    # Convert fitted/residuals back to original scale for storage
    if !isnothing(lambda)
        fitted = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted, missing)
        residuals = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals, missing)

        # Batch back-transform non-missing fitted values
        # R uses var(res) (centered variance) for biasadj: InvBoxCox(fitted, lambda, biasadj, var(res))
        fitted_indices = findall(!ismissing, fitted_trans)
        if !isempty(fitted_indices)
            fitted_vals = Float64[fitted_trans[i] for i in fitted_indices]
            if biasadj && biasadj_var > 0
                fitted_back = inv_box_cox(fitted_vals; lambda=lambda, biasadj=true, fvar=biasadj_var)
            else
                fitted_back = inv_box_cox(fitted_vals; lambda=lambda)
            end
            for (j, i) in enumerate(fitted_indices)
                fitted[i] = fitted_back[j]
            end
        end
        # R's lagwalk: residuals stay on TRANSFORMED scale (res = y - fitted where both are transformed)
        residuals = residuals_trans
    else
        fitted = fitted_trans
        residuals = residuals_trans
    end

    # Store x with NaN for missing positions
    x_store = Vector{Float64}(undef, n)
    for i in 1:n
        x_store[i] = ismissing(y[i]) ? NaN : Float64(y[i])
    end

    return NaiveFit(x_store, fitted, residuals, lag, nothing, nothing, sigma2, m,
                    "Seasonal naive method", lambda, biasadj, y_transformed)
end

"""
    rw(y::AbstractVector, m::Int=1; drift::Bool=false, lambda=nothing, biasadj::Bool=false)

Fit a random walk forecasting model, optionally with drift.

Without drift, this is equivalent to `naive()`. With drift, the forecast includes
a linear trend based on the average change in the historical data.

Without drift: y_{T+h|T} = y_T
With drift: y_{T+h|T} = y_T + h * drift, where drift = mean of first differences

# Arguments
- `y::AbstractVector` - Time series data (must have at least 2 non-missing observations)
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
            lambda::Union{Nothing, Float64, String}=nothing,
            biasadj::Bool=false)
    n = length(y)
    n >= 2 || throw(ArgumentError("Time series must have at least 2 observations, got $n"))

    # Convert to Float64, preserving structure (missings become NaN for processing)
    x = Vector{Float64}(undef, n)
    for i in 1:n
        x[i] = ismissing(y[i]) ? NaN : Float64(y[i])
    end

    # Count non-missing for validation
    n_valid = count(!isnan, x)
    n_valid >= 2 || throw(ArgumentError("Time series must have at least 2 non-missing observations, got $n_valid"))

    y_transformed = nothing

    # Apply Box-Cox transformation if specified (only to non-missing values)
    if !isnothing(lambda)
        # Get non-missing values
        valid_mask = .!isnan.(x)

        # Pre-filter values based on lambda:
        # - lambda = "auto": need positive values for lambda estimation
        # - lambda <= 0: requires positive values (log for lambda=0, or domain restriction for lambda<0)
        # - lambda > 0: signed transform handles negatives AND zeros
        #   For x=0: (sign(0) * |0|^lambda - 1) / lambda = -1/lambda (finite)
        if lambda isa String
            # Auto lambda - need positive values for estimation
            transform_mask = valid_mask .& (x .> 0)
        elseif lambda <= 0
            transform_mask = valid_mask .& (x .> 0)
            n_invalid = count(valid_mask) - count(transform_mask)
            if n_invalid > 0
                @warn "$n_invalid non-positive value(s) invalid for Box-Cox transformation with lambda=$lambda, treated as missing"
            end
        else
            # For lambda > 0, signed transform handles negatives and zeros (like R)
            transform_mask = valid_mask
        end

        x_to_transform = x[transform_mask]
        count(transform_mask) >= 2 || throw(ArgumentError("Less than 2 valid observations for Box-Cox transformation with lambda=$lambda"))

        x_trans_values, lambda = box_cox(x_to_transform, m, lambda=lambda)

        # Check for any NaN/Inf introduced by transformation itself
        bc_invalid = .!isfinite.(x_trans_values)
        n_bc_invalid = count(bc_invalid)
        if n_bc_invalid > 0
            @warn "$n_bc_invalid additional value(s) invalid after Box-Cox transformation"
            x_trans_values[bc_invalid] .= NaN
        end

        y_trans = fill(NaN, n)
        y_trans[transform_mask] = x_trans_values
        y_transformed = copy(y_trans)

        # Validate enough valid values remain after transformation
        n_valid_trans = count(isfinite, y_trans)
        n_valid_trans >= 2 || throw(ArgumentError("Less than 2 valid observations remain after Box-Cox transformation with lambda=$lambda"))
    else
        y_trans = x
    end

    lag = 1

    if drift
        # Random walk with drift
        # R's approach: drift = mean of first differences via lm(diff ~ 1)
        # This matches R forecast package's lagwalk() function

        # Compute first differences where both y[t] and y[t-1] are valid
        differences = Float64[]
        for t in 2:n
            if isfinite(y_trans[t]) && isfinite(y_trans[t-1])
                push!(differences, y_trans[t] - y_trans[t-1])
            end
        end

        n_diff = length(differences)
        n_diff >= 1 || throw(ArgumentError("Need at least 1 valid first difference for drift estimation"))

        # Drift = mean of first differences (equivalent to lm(diff ~ 1) intercept)
        drift_val = mean(differences)

        # Fitted values on transformed scale: y_{t} = y_{t-1} + drift
        # R forward-fills missing fitted values
        fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted_trans, missing)
        for t in 2:n
            if isfinite(y_trans[t-1])
                fitted_trans[t] = y_trans[t-1] + drift_val
            end
        end

        # Forward-fill missing fitted values (like R's lagwalk)
        for t in 2:n
            if ismissing(fitted_trans[t]) && !isnan(x[t])
                for k in (t-1):-1:2
                    if !ismissing(fitted_trans[k])
                        # Use last fitted value + drift for the time elapsed
                        fitted_trans[t] = fitted_trans[k] + (t - k) * drift_val
                        break
                    end
                end
            end
        end

        # Residuals on transformed scale (only when both fitted and actual are valid)
        # These are: y[t] - (y[t-1] + drift) = diff[t] - drift
        residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals_trans, missing)
        for t in 2:n
            if !ismissing(fitted_trans[t]) && isfinite(y_trans[t])
                residuals_trans[t] = y_trans[t] - fitted_trans[t]
            end
        end

        # Residual variance on transformed scale
        # R uses (n-1) divisor for drift case (corrected=true)
        valid_residuals = collect(skipmissing(residuals_trans))
        n_resid = length(valid_residuals)
        if n_resid <= 1
            sigma2 = 0.0
            if n_resid == 0
                @warn "No valid residuals available for variance estimation, using sigma2=0"
            end
        else
            sigma2 = var(valid_residuals, corrected=true)  # Match R: divide by (n-1) for drift
        end

        # Standard error of drift = SE of mean = sd(differences) / sqrt(n_diff)
        # This matches R's lm() coefficient standard error
        if n_diff <= 1
            drift_se = 0.0
        else
            drift_se = std(differences) / sqrt(n_diff)
        end

        method_name = "Random walk with drift"

        # Convert fitted/residuals back to original scale for storage
        if !isnothing(lambda)
            fitted = Vector{Union{Float64, Missing}}(undef, n)
            fill!(fitted, missing)
            residuals = Vector{Union{Float64, Missing}}(undef, n)
            fill!(residuals, missing)

            # Batch back-transform non-missing fitted values
            # Apply bias adjustment if requested (R does: InvBoxCox(fitted, lambda, biasadj, var(res)))
            fitted_indices = findall(!ismissing, fitted_trans)
            if !isempty(fitted_indices)
                fitted_vals = Float64[fitted_trans[i] for i in fitted_indices]
                if biasadj && sigma2 > 0
                    fitted_back = inv_box_cox(fitted_vals; lambda=lambda, biasadj=true, fvar=sigma2)
                else
                    fitted_back = inv_box_cox(fitted_vals; lambda=lambda)
                end
                for (j, i) in enumerate(fitted_indices)
                    fitted[i] = fitted_back[j]
                end
            end
            # R's lagwalk: residuals stay on TRANSFORMED scale
            residuals = residuals_trans
        else
            fitted = fitted_trans
            residuals = residuals_trans
        end

        # Store x with NaN for missing positions
        x_store = Vector{Float64}(undef, n)
        for i in 1:n
            x_store[i] = ismissing(y[i]) ? NaN : Float64(y[i])
        end

        return NaiveFit(x_store, fitted, residuals, lag, drift_val, drift_se, sigma2, m,
                        method_name, lambda, biasadj, y_transformed)
    else
        # Random walk without drift (same as naive)
        # R forward-fills missing fitted values
        fitted_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(fitted_trans, missing)
        for t in 2:n
            if isfinite(y_trans[t-1])
                fitted_trans[t] = y_trans[t-1]
            end
        end

        # Forward-fill missing fitted values (like R's lagwalk)
        for t in 2:n
            if ismissing(fitted_trans[t]) && !isnan(x[t])
                for k in (t-1):-1:2
                    if !ismissing(fitted_trans[k])
                        fitted_trans[t] = fitted_trans[k]
                        break
                    end
                end
            end
        end

        residuals_trans = Vector{Union{Float64, Missing}}(undef, n)
        fill!(residuals_trans, missing)
        for t in 2:n
            if !ismissing(fitted_trans[t]) && isfinite(y_trans[t])
                residuals_trans[t] = y_trans[t] - fitted_trans[t]
            end
        end

        # Residual variance on transformed scale
        # R uses two different variance measures:
        # 1. sigma2 = mean(res^2) for prediction intervals (MSE)
        # 2. var(res) (centered variance) for biasadj in InvBoxCox
        valid_residuals = collect(skipmissing(residuals_trans))
        n_resid = length(valid_residuals)
        if n_resid == 0
            sigma2 = 0.0
            biasadj_var = 0.0
            @warn "No valid residuals available for variance estimation, using sigma2=0"
        else
            # sigma2 = MSE for prediction intervals (R: sigma <- sqrt(mean(res^2, na.rm=TRUE)))
            sigma2 = mean(valid_residuals .^ 2)
            # biasadj_var = centered variance for bias adjustment (R: var(res))
            biasadj_var = n_resid > 1 ? var(valid_residuals) : 0.0
        end

        # Convert fitted/residuals back to original scale for storage
        if !isnothing(lambda)
            fitted = Vector{Union{Float64, Missing}}(undef, n)
            fill!(fitted, missing)

            # Batch back-transform non-missing fitted values
            # R uses var(res) (centered variance) for biasadj: InvBoxCox(fitted, lambda, biasadj, var(res))
            fitted_indices = findall(!ismissing, fitted_trans)
            if !isempty(fitted_indices)
                fitted_vals = Float64[fitted_trans[i] for i in fitted_indices]
                if biasadj && biasadj_var > 0
                    fitted_back = inv_box_cox(fitted_vals; lambda=lambda, biasadj=true, fvar=biasadj_var)
                else
                    fitted_back = inv_box_cox(fitted_vals; lambda=lambda)
                end
                for (j, i) in enumerate(fitted_indices)
                    fitted[i] = fitted_back[j]
                end
            end
            # R's lagwalk: residuals stay on TRANSFORMED scale
            residuals = residuals_trans
        else
            fitted = fitted_trans
            residuals = residuals_trans
        end

        # Store x with NaN for missing positions
        x_store = Vector{Float64}(undef, n)
        for i in 1:n
            x_store[i] = ismissing(y[i]) ? NaN : Float64(y[i])
        end

        return NaiveFit(x_store, fitted, residuals, lag, nothing, nothing, sigma2, m,
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
    if fit.lag > 1
        # Seasonal naive - show seasonal period
        println(io, "Seasonal period: ", fit.m)
    end
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
