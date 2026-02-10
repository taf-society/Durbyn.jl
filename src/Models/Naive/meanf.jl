"""
    MeanFit

Fitted mean forecasting model.

# Fields
- `x::Vector{Float64}` - Original time series data (preserves original length, NaN for missing/invalid)
- `mu::Float64` - Mean on transformed scale (if lambda used) for forecasting
- `mu_original::Float64` - Mean on original scale for display
- `sd::Float64` - Standard deviation on transformed scale
- `lambda::Union{Nothing, Float64}` - Box-Cox transformation parameter
- `biasadj::Bool` - Whether bias adjustment is applied
- `fitted::Vector{Union{Float64, Missing}}` - Fitted values on original scale (preserves original length)
- `residuals::Vector{Union{Float64, Missing}}` - Residuals on original scale (preserves original length)
- `n::Int` - Number of valid observations used
- `m::Int` - Seasonal period
"""
struct MeanFit
    x::Vector{Float64}
    mu::Float64
    mu_original::Float64
    sd::Float64
    lambda::Union{Nothing, Float64}
    biasadj::Bool
    fitted::Vector{Union{Float64, Missing}}
    residuals::Vector{Union{Float64, Missing}}
    n::Int
    m::Int
end

"""
    meanf(y::AbstractArray, m::Int; lambda=nothing, biasadj::Bool=false)

Fit a mean forecasting model.

The mean method uses the sample mean as the forecast for all future periods.

# Arguments
- `y::AbstractArray` - Time series data
- `m::Int` - Seasonal period

# Keyword Arguments
- `lambda::Union{Nothing, Float64}=nothing` - Box-Cox transformation parameter
- `biasadj::Bool=false` - Apply bias adjustment for Box-Cox back-transformation
"""
function meanf(y::AbstractArray, m::Int;
               lambda::Union{Nothing, Float64, String}=nothing,
               biasadj::Bool=false)
    n_orig = length(y)

    x_orig = Vector{Float64}(undef, n_orig)
    for i in 1:n_orig
        x_orig[i] = ismissing(y[i]) ? NaN : Float64(y[i])
    end

    valid_mask = .!isnan.(x_orig)
    n_valid = count(valid_mask)
    n_valid >= 1 || throw(ArgumentError("Time series must have at least 1 non-missing/non-NaN observation"))

    if !isnothing(lambda)
        if lambda isa String
            transform_mask = valid_mask .& (x_orig .> 0)
        elseif lambda <= 0
            transform_mask = valid_mask .& (x_orig .> 0)
            n_invalid = count(valid_mask) - count(transform_mask)
            if n_invalid > 0
                @warn "$n_invalid non-positive value(s) invalid for Box-Cox transformation with lambda=$lambda, treated as missing"
            end
        else
            transform_mask = valid_mask
        end

        x_to_transform = x_orig[transform_mask]
        count(transform_mask) >= 1 || throw(ArgumentError("No valid observations for Box-Cox transformation with lambda=$lambda"))

        x_trans_values, lambda = box_cox(x_to_transform, m, lambda=lambda)

        bc_valid_mask = isfinite.(x_trans_values)
        n_bc_invalid = count(!, bc_valid_mask)
        if n_bc_invalid > 0
            @warn "$n_bc_invalid additional value(s) invalid after Box-Cox transformation"
        end

        y_trans = fill(NaN, n_orig)
        transform_indices = findall(transform_mask)
        for (j, i) in enumerate(transform_indices)
            if bc_valid_mask[j]
                y_trans[i] = x_trans_values[j]
            end
        end

        n = count(isfinite, y_trans)
        n >= 1 || throw(ArgumentError("No valid observations remain after Box-Cox transformation with lambda=$lambda"))

        x_trans_valid = filter(isfinite, y_trans)
    else
        y_trans = x_orig
        x_trans_valid = filter(!isnan, x_orig)
        n = length(x_trans_valid)
    end

    mu_trans = mean(x_trans_valid)
    sd_trans = n > 1 ? std(x_trans_valid) : 0.0

    residuals_trans_valid = x_trans_valid .- mu_trans
    sigma2 = n > 0 ? mean(residuals_trans_valid .^ 2) : 0.0

    fitted = Vector{Union{Float64, Missing}}(undef, n_orig)
    residuals = Vector{Union{Float64, Missing}}(undef, n_orig)
    fill!(fitted, missing)
    fill!(residuals, missing)

    if !isnothing(lambda)
        if biasadj && sigma2 > 0
            mu_original = inv_box_cox([mu_trans]; lambda=lambda, biasadj=true, fvar=sigma2)[1]
        else
            mu_original = inv_box_cox([mu_trans]; lambda=lambda)[1]
        end

        fitted_backtrans = inv_box_cox([mu_trans]; lambda=lambda)[1]
        for i in 1:n_orig
            if isfinite(y_trans[i])
                fitted[i] = fitted_backtrans
                residuals[i] = y_trans[i] - mu_trans
            end
        end
    else
        mu_original = mu_trans
        for i in 1:n_orig
            if !isnan(x_orig[i])
                fitted[i] = mu_trans
                residuals[i] = x_orig[i] - mu_trans
            end
        end
    end

    return MeanFit(x_orig, mu_trans, mu_original, sd_trans, lambda, biasadj, fitted, residuals, n, m)
end

function meanf(y::AbstractArray, m::Int, lambda::Union{Nothing, Float64, String})
    meanf(y, m; lambda=lambda, biasadj=false)
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
    biasadj = object.biasadj
    m = object.m
    n = object.n

    f_trans = fill(mu_trans, h)

    if fan
        level = collect(51.0:3:99.0)
    else
        all_fraction = all(lv -> 0.0 < lv < 1.0, level)

        if all_fraction
            level = 100.0 .* level
        elseif any(lv -> lv <= 0.0 || lv > 99.99, level)
            error("Confidence levels must be in (0, 1) (fractions) or (0, 99.99] (percentages)")
        end
    end

    nconf = length(level)

    valid_x = filter(!isnan, object.x)
    if !isnothing(lambda)
        if lambda <= 0
            x_to_transform = filter(x -> x > 0, valid_x)
        else
            x_to_transform = valid_x
        end
        if isempty(x_to_transform)
            residuals_trans_valid = Float64[]
        else
            x_trans_valid, _ = box_cox(x_to_transform, m, lambda=lambda)
            x_trans_valid = filter(isfinite, x_trans_valid)
            residuals_trans_valid = x_trans_valid .- mu_trans
        end
    else
        residuals_trans_valid = valid_x .- mu_trans
    end
    sigma2 = length(residuals_trans_valid) > 0 ? mean(residuals_trans_valid .^ 2) : 0.0

    lower_trans = Vector{Vector{Float64}}(undef, nconf)
    upper_trans = Vector{Vector{Float64}}(undef, nconf)

    if bootstrap
        e = residuals_trans_valid .- mean(residuals_trans_valid)
        ne = length(e)

        if ne == 0
            @warn "No valid residuals for bootstrap, using analytical intervals"
            bootstrap = false
        else
            sim = Matrix{Float64}(undef, npaths, h)
            for i in 1:npaths
                for j in 1:h
                    sim[i, j] = mu_trans + e[rand(1:ne)]
                end
            end

            for i in 1:nconf
                lower_trans[i] = [quantile(sim[:, j], 0.5 - level[i] / 200.0) for j in 1:h]
                upper_trans[i] = [quantile(sim[:, j], 0.5 + level[i] / 200.0) for j in 1:h]
            end
        end
    end

    if !bootstrap
        for i in 1:nconf
            if n > 1
                tfrac = quantile(TDist(n - 1), 0.5 - level[i] / 200.0)
                w = -tfrac * sd_trans * sqrt(1.0 + 1.0 / n)
            else
                w = Inf
            end
            lower_trans[i] = f_trans .- w
            upper_trans[i] = f_trans .+ w
        end
    end

    if !isnothing(lambda)
        if biasadj && n > 1
            max_idx = argmax(level)
            fvar_info = Dict(:level => [level[max_idx]],
                             :upper => upper_trans[max_idx],
                             :lower => lower_trans[max_idx])
            f = inv_box_cox(f_trans; lambda=lambda, biasadj=true, fvar=fvar_info)
        else
            f = inv_box_cox(f_trans; lambda=lambda)
        end

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

    level_out = Float64.(level)

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

function forecast(object::MeanFit, h::Int, level::Vector{Float64}=[80.0, 95.0], fan::Bool=false, bootstrap::Bool=false, npaths::Int=5000)
    forecast(object; h=h, level=level, fan=fan, bootstrap=bootstrap, npaths=npaths)
end
