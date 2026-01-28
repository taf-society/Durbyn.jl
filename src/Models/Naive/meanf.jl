"""
    MeanFit

Fitted mean forecasting model.

# Fields
- `x::Vector{Float64}` - Original time series data (with missings removed)
- `mu::Float64` - Mean on transformed scale (if lambda used) for forecasting
- `mu_original::Float64` - Mean on original scale for display
- `sd::Float64` - Standard deviation on transformed scale
- `lambda::Union{Nothing, Float64}` - Box-Cox transformation parameter
- `fitted::Vector{Float64}` - Fitted values on original scale
- `residuals::Vector{Float64}` - Residuals on original scale
- `n::Int` - Number of non-missing observations
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
    # Collect non-missing values
    x = collect(Float64, skipmissing(y))
    n = length(x)
    n >= 1 || throw(ArgumentError("Time series must have at least 1 non-missing observation"))

    # Apply Box-Cox transformation if specified
    if !isnothing(lambda)
        x_trans, lambda = box_cox(x, m, lambda=lambda)
    else
        x_trans = x
    end

    # Compute mean and sd on transformed scale
    mu_trans = mean(x_trans)
    sd_trans = n > 1 ? std(x_trans) : 0.0

    # Fitted values and residuals on transformed scale
    fitted_trans = fill(mu_trans, n)
    residuals_trans = x_trans .- fitted_trans

    # Back-transform to original scale for storage
    if !isnothing(lambda)
        mu_original = inv_box_cox([mu_trans]; lambda=lambda)[1]
        fitted = inv_box_cox(fitted_trans; lambda=lambda)
        residuals = x .- fitted
    else
        mu_original = mu_trans
        fitted = fitted_trans
        residuals = residuals_trans
    end

    return MeanFit(x, mu_trans, mu_original, sd_trans, lambda, fitted, residuals, n, m)
end

"""
    forecast(object::MeanFit, h=10, level=[80.0, 95.0], fan=false, bootstrap=false, npaths=5000)

Generate forecasts from a fitted mean model.
"""
function forecast(object::MeanFit, h::Int=10, level::Vector{Float64}=[80.0, 95.0], fan::Bool=false, bootstrap::Bool=false, npaths::Int=5000)
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
    elseif any(level .> 1.0)
        if minimum(level) < 0.0 || maximum(level) > 99.99
            error("Confidence limit out of range")
        end
    else
        level = 100.0 .* level
    end

    nconf = length(level)
    lower_trans = Matrix{Float64}(undef, h, nconf)
    upper_trans = Matrix{Float64}(undef, h, nconf)

    if bootstrap
        # Residuals on transformed scale for bootstrap
        if !isnothing(lambda)
            x_trans, _ = box_cox(object.x, m, lambda=lambda)
            residuals_trans = x_trans .- mu_trans
        else
            # Compute residuals on transformed scale (which equals original when no lambda)
            residuals_trans = object.x .- mu_trans
        end
        e = residuals_trans .- mean(residuals_trans)
        ne = length(e)

        # Bootstrap simulation per horizon - sample with replacement from residuals
        for j in 1:h
            # Sample npaths residuals with replacement and add to mean
            sim = [mu_trans + e[rand(1:ne)] for _ in 1:npaths]
            for i in 1:nconf
                lower_trans[j, i] = quantile(sim, 0.5 - level[i] / 200.0)
                upper_trans[j, i] = quantile(sim, 0.5 + level[i] / 200.0)
            end
        end
    else
        # t-distribution intervals on transformed scale
        for i in 1:nconf
            tfrac = quantile(TDist(n - 1), 0.5 - level[i] / 200.0)
            w = -tfrac * sd_trans * sqrt(1.0 + 1.0 / n)
            lower_trans[:, i] .= f_trans .- w
            upper_trans[:, i] .= f_trans .+ w
        end
    end

    # Back-transform to original scale
    if !isnothing(lambda)
        f = inv_box_cox(f_trans; lambda=lambda)
        lower = inv_box_cox(lower_trans; lambda=lambda)
        upper = inv_box_cox(upper_trans; lambda=lambda)
    else
        f = f_trans
        lower = lower_trans
        upper = upper_trans
    end

    return Dict("mean" => f, "lower" => lower, "upper" => upper, "level" => level, "m" => m)
end