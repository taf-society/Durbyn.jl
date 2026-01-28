"""
    Forecast methods for naive models

Provides forecast generation for NaiveFit objects with proper
prediction intervals that account for forecast horizon and method-specific
variance characteristics.
"""

# Helper functions for finding valid values (moved outside forecast for performance)

"""
    _find_last_valid(arr)

Find the last non-NaN value in an array, searching backwards.
Returns NaN if all values are NaN.
"""
function _find_last_valid(arr)
    for i in length(arr):-1:1
        if !isnan(arr[i])
            return arr[i]
        end
    end
    return NaN  # All values are NaN
end

"""
    _find_last_valid_at_season(arr, pos, period)

Find the last non-NaN value at a specific seasonal position.

# Arguments
- `arr` - Array to search
- `pos` - Position in seasonal cycle (1-indexed, 1 to period)
- `period` - Seasonal period

Returns NaN if no valid value found at this seasonal position.
"""
function _find_last_valid_at_season(arr, pos, period)
    n = length(arr)
    # Search backwards through the array at this seasonal position
    # Start from the most recent occurrence of this position
    start_idx = n - mod(n - pos, period)
    for idx in start_idx:-period:1
        if idx >= 1 && !isnan(arr[idx])
            return arr[idx]
        end
    end
    return NaN
end

"""
    forecast(object::NaiveFit; h::Int=10, level::Vector{<:Real}=[80, 95], fan::Bool=false)

Generate forecasts from a fitted naive model.

# Arguments
- `object::NaiveFit` - Fitted naive model

# Keyword Arguments
- `h::Int=10` - Forecast horizon (number of periods ahead)
- `level::Vector{<:Real}=[80, 95]` - Confidence levels for prediction intervals
- `fan::Bool=false` - If true, generate fan chart levels (51, 54, ..., 99)

# Returns
`Forecast` object containing:
- `mean` - Point forecasts
- `lower` - Lower prediction bounds for each level
- `upper` - Upper prediction bounds for each level
- `level` - Confidence levels used
- `x` - Original data
- `fitted` - In-sample fitted values
- `residuals` - In-sample residuals

# Prediction Interval Formulas

**Naive/Random Walk (without drift)**:
- Variance grows linearly with horizon: Var(h) = h * sigma2
- SE(h) = sqrt(h) * sigma

**Seasonal Naive**:
- Variance increases with complete seasons: Var(h) = ceil(h/m) * sigma2
- SE(h) = sqrt(ceil(h/m)) * sigma

**Random Walk with Drift**:
- Includes drift uncertainty: Var(h) = h * sigma2 + h^2 * SE(drift)^2
- SE(h) = sqrt(h * sigma2 + h^2 * drift_se^2)

# Examples
```julia
y = randn(100)
fit = naive(y)
fc = forecast(fit, h=12)

# Access results
fc.mean         # Point forecasts
fc.upper[2]     # 95% upper bound
fc.lower[1]     # 80% lower bound
```

# See Also
- [`naive`](@ref)
- [`snaive`](@ref)
- [`rw`](@ref)
"""
function forecast(object::NaiveFit;
                  h::Int=10,
                  level::Vector{<:Real}=[80, 95],
                  fan::Bool=false)
    h >= 1 || throw(ArgumentError("Forecast horizon h must be >= 1, got $h"))

    x = object.x
    n = length(x)
    sigma2 = object.sigma2
    m = object.m
    lag = object.lag
    lambda = object.lambda
    biasadj = object.biasadj

    # Adjust levels for fan or percentage
    if fan
        level = collect(51.0:3:99.0)
    else
        # Determine if levels are fractions (0,1] or percentages (1,100]
        all_fraction = all(lv -> 0.0 < lv <= 1.0, level)
        all_percent = all(lv -> 1.0 < lv <= 99.99, level)

        if !all_fraction && !all_percent
            # Check for mixed units or invalid values
            if any(lv -> lv <= 0.0 || lv > 99.99, level)
                error("Confidence levels must be in (0, 1] (fractions) or (1, 99.99] (percentages)")
            else
                error("Mixed confidence level units detected. Use all fractions (0, 1] or all percentages (1, 99.99]")
            end
        end

        if all_fraction
            level = 100.0 .* level
        end
    end

    nconf = length(level)

    # Get transformed data for forecasting (if lambda was used)
    if !isnothing(lambda) && !isnothing(object.y_transformed)
        y_trans = object.y_transformed
    else
        y_trans = x
    end

    # Generate point forecasts on transformed scale
    if !isnothing(object.drift)
        # RW with drift: forecast = last valid transformed value + h * drift
        last_trans = _find_last_valid(y_trans)
        f_trans = [last_trans + i * object.drift for i in 1:h]
    elseif lag == 1
        # Naive: forecast = last valid transformed value
        last_trans = _find_last_valid(y_trans)
        f_trans = fill(Float64(last_trans), h)
    else
        # Seasonal naive: forecast = last valid value at each seasonal position
        f_trans = Vector{Float64}(undef, h)
        for i in 1:h
            # Position in seasonal cycle (1 to m)
            pos = mod1(i, m)
            # First try the simple approach: look at position n - m + pos
            simple_idx = n - m + pos
            if simple_idx >= 1 && !isnan(y_trans[simple_idx])
                f_trans[i] = y_trans[simple_idx]
            else
                # Fall back to finding last valid at this seasonal position
                f_trans[i] = _find_last_valid_at_season(y_trans, pos, m)
            end
        end
    end

    # Calculate standard errors on transformed scale
    # Handle NaN sigma2 (can happen with single residual)
    if isnan(sigma2) || sigma2 < 0
        sigma2 = 0.0
    end
    sigma = sqrt(sigma2)
    se = Vector{Float64}(undef, h)

    for i in 1:h
        if !isnothing(object.drift) && !isnothing(object.drift_se)
            # RW with drift: SE^2 = h * sigma2 + h^2 * drift_se^2
            drift_se = isnan(object.drift_se) ? 0.0 : object.drift_se
            se[i] = sqrt(i * sigma2 + i^2 * drift_se^2)
        elseif lag == 1
            # Naive/RW without drift: SE = sqrt(h) * sigma
            se[i] = sqrt(i) * sigma
        else
            # Seasonal naive: SE based on complete seasons
            # SE = sqrt(k) * sigma, where k = ceil(h/m)
            k = ceil(Int, i / m)
            se[i] = sqrt(k) * sigma
        end
    end

    # Calculate prediction intervals using Normal quantiles on transformed scale
    lower_trans = Vector{Vector{Float64}}(undef, nconf)
    upper_trans = Vector{Vector{Float64}}(undef, nconf)

    for j in 1:nconf
        alpha = (100 - level[j]) / 200
        z = quantile(Normal(), 1 - alpha)

        lower_trans[j] = f_trans .- z .* se
        upper_trans[j] = f_trans .+ z .* se
    end

    # Apply inverse Box-Cox transformation if needed
    if !isnothing(lambda)
        # Compute forecast variance for bias adjustment
        fvar = se .^ 2

        if biasadj
            # Bias-adjusted back-transformation for point forecast
            f = inv_box_cox(f_trans; lambda=lambda, biasadj=true, fvar=fvar)
        else
            f = inv_box_cox(f_trans; lambda=lambda)
        end

        # Back-transform intervals (without bias adjustment for bounds)
        lower = Vector{Vector{Float64}}(undef, nconf)
        upper = Vector{Vector{Float64}}(undef, nconf)
        for j in 1:nconf
            lower[j] = inv_box_cox(lower_trans[j]; lambda=lambda)
            upper[j] = inv_box_cox(upper_trans[j]; lambda=lambda)
        end
    else
        f = f_trans
        lower = lower_trans
        upper = upper_trans
    end

    # Convert to Float64 vectors
    mean_vec = Float64.(f)
    x_data = Float64.(object.x)

    # Warn if all forecasts are NaN (indicates no valid data to forecast from)
    if all(isnan, mean_vec)
        @warn "All forecast values are NaN. This typically indicates all observations " *
              "at the required lag positions are missing."
    end

    # Fitted values and residuals (already on original scale in NaiveFit)
    fitted_vals = object.fitted
    residuals_vals = object.residuals

    # Keep level as-is (don't round to avoid mislabeling e.g., 99.5 → 100)
    level_out = Float64.(level)

    return Forecast(
        object,
        object.method,
        mean_vec,
        level_out,
        x_data,
        upper,
        lower,
        fitted_vals,
        residuals_vals
    )
end
