"""
    MeanFit

Fitted mean forecasting model.

# Fields
- `x::Vector{Float64}` - Original time series data (with missings/NaN/invalid Box-Cox removed)
- `mu::Float64` - Mean on transformed scale (if lambda used) for forecasting
- `mu_original::Float64` - Mean on original scale for display
- `sd::Float64` - Standard deviation on transformed scale
- `lambda::Union{Nothing, Float64}` - Box-Cox transformation parameter
- `fitted::Vector{Float64}` - Fitted values on original scale
- `residuals::Vector{Float64}` - Residuals on original scale
- `n::Int` - Number of valid observations used
- `m::Int` - Seasonal period
"""
struct MeanFit
    x::Vector{Float64}
    mu::Float64
    mu_original::Float64
    sd::Float64
    lambda::Union{Nothing, Float64}
    fitted::Vector{Float64}
    residuals::Vector{Float64}
    n::Int
    m::Int
end

"""
    meanf(y::AbstractArray, m::Int, lambda=nothing)

Fit a mean forecasting model.

The mean method uses the sample mean as the forecast for all future periods.
"""
function meanf(y::AbstractArray, m::Int, lambda::Union{Nothing, Float64, String}=nothing)
    # Collect non-missing values and filter NaN (consistent with naive/snaive/rw)
    x_no_missing = collect(Float64, skipmissing(y))
    x = filter(!isnan, x_no_missing)
    n_before_transform = length(x)
    n_before_transform >= 1 || throw(ArgumentError("Time series must have at least 1 non-missing/non-NaN observation"))

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        # Pre-filter non-positive values for Box-Cox (required for log when lambda=0, and generally)
        positive_mask = x .> 0
        n_nonpositive = count(!, positive_mask)
        if n_nonpositive > 0
            @warn "$n_nonpositive non-positive value(s) invalid for Box-Cox transformation, treated as missing"
        end
        x_positive = x[positive_mask]

        length(x_positive) >= 1 || throw(ArgumentError("No positive observations for Box-Cox transformation with lambda=$lambda"))

        x_trans_valid, lambda = box_cox(x_positive, m, lambda=lambda)

        # Also check for any NaN/Inf introduced by transformation itself
        bc_valid_mask = isfinite.(x_trans_valid)
        n_bc_invalid = count(!, bc_valid_mask)
        if n_bc_invalid > 0
            @warn "$n_bc_invalid additional value(s) invalid after Box-Cox transformation"
            x_positive = x_positive[bc_valid_mask]
            x_trans_valid = x_trans_valid[bc_valid_mask]
        end

        x_valid = x_positive
        n = length(x_valid)

        n >= 1 || throw(ArgumentError("No valid observations remain after Box-Cox transformation with lambda=$lambda"))
    else
        x_valid = x
        x_trans_valid = x
        n = length(x)
    end

    # Compute mean and sd on transformed scale (using only valid values)
    mu_trans = mean(x_trans_valid)
    sd_trans = n > 1 ? std(x_trans_valid) : 0.0

    # Fitted values and residuals on transformed scale
    fitted_trans = fill(mu_trans, n)
    residuals_trans = x_trans_valid .- fitted_trans

    # Back-transform to original scale for storage
    if !isnothing(lambda)
        mu_original = inv_box_cox([mu_trans]; lambda=lambda)[1]
        fitted = inv_box_cox(fitted_trans; lambda=lambda)
        # Residuals on original scale: only for valid transformed observations
        residuals = x_valid .- fitted
    else
        mu_original = mu_trans
        fitted = fitted_trans
        residuals = residuals_trans
    end

    return MeanFit(x_valid, mu_trans, mu_original, sd_trans, lambda, fitted, residuals, n, m)
end

"""
    forecast(object::MeanFit; h=10, level=[80.0, 95.0], fan=false, bootstrap=false, npaths=5000)

Generate forecasts from a fitted mean model.

Returns a `Forecast` object for consistency with other forecasting methods.
"""
function forecast(object::MeanFit;
                  h::Int=10,
                  level::Vector{<:Real}=[80.0, 95.0],
                  fan::Bool=false,
                  bootstrap::Bool=false,
                  npaths::Int=5000)
    mu_trans = object.mu
    sd_trans = object.sd
    lambda = object.lambda
    m = object.m
    n = object.n

    # Point forecasts on transformed scale
    f_trans = fill(mu_trans, h)

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

    # Use Vector{Vector} for intervals (consistent with naive_forecast)
    lower_trans = Vector{Vector{Float64}}(undef, nconf)
    upper_trans = Vector{Vector{Float64}}(undef, nconf)

    if bootstrap
        # Residuals on transformed scale for bootstrap
        if !isnothing(lambda)
            x_trans, _ = box_cox(object.x, m, lambda=lambda)
            # Filter invalid transformed values
            valid_mask = isfinite.(x_trans)
            x_trans_valid = x_trans[valid_mask]
            residuals_trans = x_trans_valid .- mu_trans
        else
            residuals_trans = object.x .- mu_trans
        end
        e = residuals_trans .- mean(residuals_trans)
        ne = length(e)

        # Bootstrap simulation - sample with replacement from residuals
        for i in 1:nconf
            lower_h = Vector{Float64}(undef, h)
            upper_h = Vector{Float64}(undef, h)
            for j in 1:h
                sim = [mu_trans + e[rand(1:ne)] for _ in 1:npaths]
                lower_h[j] = quantile(sim, 0.5 - level[i] / 200.0)
                upper_h[j] = quantile(sim, 0.5 + level[i] / 200.0)
            end
            lower_trans[i] = lower_h
            upper_trans[i] = upper_h
        end
    else
        # t-distribution intervals on transformed scale
        # Handle n == 1 case: use infinite intervals (like R)
        for i in 1:nconf
            if n > 1
                tfrac = quantile(TDist(n - 1), 0.5 - level[i] / 200.0)
                w = -tfrac * sd_trans * sqrt(1.0 + 1.0 / n)
            else
                # n == 1: infinite intervals (no variance estimate possible)
                w = Inf
            end
            lower_trans[i] = f_trans .- w
            upper_trans[i] = f_trans .+ w
        end
    end

    # Back-transform to original scale
    if !isnothing(lambda)
        f = inv_box_cox(f_trans; lambda=lambda)
        lower = Vector{Vector{Float64}}(undef, nconf)
        upper = Vector{Vector{Float64}}(undef, nconf)
        for i in 1:nconf
            lower[i] = inv_box_cox(lower_trans[i]; lambda=lambda)
            upper[i] = inv_box_cox(upper_trans[i]; lambda=lambda)
        end
    else
        f = f_trans
        lower = lower_trans
        upper = upper_trans
    end

    # Convert level to Float64 (don't round to preserve exact values like 99.5)
    level_out = Float64.(level)

    # Return Forecast object for consistency with other methods
    return Forecast(
        object,
        "Mean method",
        Float64.(f),
        level_out,
        Float64.(object.x),
        upper,
        lower,
        object.fitted,
        object.residuals
    )
end

# Keep backward-compatible positional argument version
function forecast(object::MeanFit, h::Int, level::Vector{Float64}=[80.0, 95.0], fan::Bool=false, bootstrap::Bool=false, npaths::Int=5000)
    forecast(object; h=h, level=level, fan=fan, bootstrap=bootstrap, npaths=npaths)
end
