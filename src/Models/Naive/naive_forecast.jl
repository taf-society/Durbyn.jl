"""
    Forecast methods for naive models

Provides forecast generation for NaiveFit objects with proper
prediction intervals that account for forecast horizon and method-specific
variance characteristics.
"""

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
    elseif any(lv -> lv > 1.0, level)
        if minimum(level) < 0.0 || maximum(level) > 99.99
            error("Confidence limit out of range")
        end
    else
        level = 100.0 .* level
    end

    nconf = length(level)
    level_int = round.(Int, level)

    # Get transformed data for forecasting (if lambda was used)
    if !isnothing(lambda) && !isnothing(object.y_transformed)
        y_trans = object.y_transformed
    else
        y_trans = x
    end

    # Generate point forecasts on transformed scale
    if !isnothing(object.drift)
        # RW with drift: forecast = last transformed value + h * drift
        last_trans = y_trans[n]
        f_trans = [last_trans + i * object.drift for i in 1:h]
    elseif lag == 1
        # Naive: forecast = last transformed value
        last_trans = y_trans[n]
        f_trans = fill(Float64(last_trans), h)
    else
        # Seasonal naive: forecast = value from m periods ago, cycling
        f_trans = Vector{Float64}(undef, h)
        for i in 1:h
            # Use last m observations cyclically
            idx = mod1(i, m)
            f_trans[i] = y_trans[n - m + idx]
        end
    end

    # Calculate standard errors on transformed scale
    se = Vector{Float64}(undef, h)
    sigma = sqrt(sigma2)

    for i in 1:h
        if !isnothing(object.drift) && !isnothing(object.drift_se)
            # RW with drift: SE^2 = h * sigma2 + h^2 * drift_se^2
            se[i] = sqrt(i * sigma2 + i^2 * object.drift_se^2)
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

    # Fitted values and residuals (already on original scale in NaiveFit)
    fitted_vals = object.fitted
    residuals_vals = object.residuals

    return Forecast(
        object,
        object.method,
        mean_vec,
        level_int,
        x_data,
        upper,
        lower,
        fitted_vals,
        residuals_vals
    )
end
