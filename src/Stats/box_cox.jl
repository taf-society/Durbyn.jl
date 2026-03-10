function _nanmean(v)
    vals = filter(!isnan, v)
    isempty(vals) ? NaN : mean(vals)
end

function _nanstd(v)
    vals = filter(!isnan, v)
    length(vals) < 2 ? NaN : std(vals)
end

function _guerrero_cv(lam::Float64, x::Vector{Float64}, m::Int; nonseasonal_length::Int=2)
    period = max(nonseasonal_length, m)
    n_obs = length(x)
    n_years = div(n_obs, period)
    n_trimmed = n_years * period
    period_matrix = reshape(x[(end-n_trimmed+1):end], period, n_years)
    period_means = [_nanmean(period_matrix[:, j]) for j in 1:n_years]
    period_sds = [_nanstd(period_matrix[:, j]) for j in 1:n_years]
    cv_ratios = period_sds ./ period_means .^ (1 - lam)
    valid_ratios = filter(!isnan, cv_ratios)
    length(valid_ratios) < 2 && return Inf
    return _nanstd(valid_ratios) / _nanmean(valid_ratios)
end

function _guerrero_lambda(x::AbstractVector{<:Number}, m::Int, lower::Real=-1.0, upper::Real=2.0, nonseasonal_length::Int=2)
    x_float = Vector{Float64}([ismissing(v) ? NaN : Float64(v) for v in x])
    if any(v -> !isnan(v) && v <= 0, x_float)
        println("Warning: Guerrero's method for selecting a Box-Cox parameter (lambda) is given for strictly positive data.")
    end

    result = Optimize.brent(lam -> _guerrero_cv(lam, x_float, m, nonseasonal_length=nonseasonal_length), Float64(lower), Float64(upper))
    return result.x_opt
end

function _qr_residuals(X, y)
    Q, R = qr(X)
    a = Q' * y
    b = R \ a[1:size(R, 2)]
    c = X * b
    return y .- c
end

function _box_cox_loglik(x::AbstractArray, m::Int; lower::Float64 = -1.0, upper::Float64 = 2.0, is_ts::Bool = true)
    n_orig = length(x)
    good = [!ismissing(v) && !(v isa AbstractFloat && isnan(v)) for v in x]
    x_clean = Vector{Float64}([Float64(x[i]) for i in 1:n_orig if good[i]])

    if any(x_clean .<= 0)
        throw(DomainError(x, "Box-Cox transformation requires positive values"))
    end

    logx = log.(x_clean)
    geometric_mean = exp(mean(logx))

    if !is_ts
        X_full = ones(n_orig, 1)
    else
        t = collect(1:n_orig)
        if m > 1
            s = mod1.(t, m)
            D = zeros(n_orig, m-1)
            for j in 2:m
                D[:, j-1] .= (s .== j)
            end
            X_full = hcat(ones(n_orig), t, D)
        else
            X_full = hcat(ones(n_orig), t)
        end
    end

    X = X_full[good, :]

    lambda_grid = collect(lower:0.05:upper)
    loglik = similar(lambda_grid)

    for (i, λ) in enumerate(lambda_grid)
        xt = if abs(λ) > 0.02
            (x_clean .^ λ .- 1) ./ λ
        else
            logx .* (1 .+ (λ .* logx)/2 .* (1 .+ (λ .* logx)/3 .* (1 .+ (λ .* logx)/4)))
        end

        z = xt ./ geometric_mean^(λ - 1)
        β = X \ z
        r = z .- X*β

        loglik[i] = -n_orig/2 * log(sum(abs2, r))
    end

    return lambda_grid[argmax(loglik)]
end

"""
    box_cox_lambda(x::AbstractVector, m::Int; method::Symbol = :guerrero, lower::Real = -1, upper::Real = 2)

Automatic selection of Box-Cox transformation parameter.

# Description
If `method === :guerrero`, Guerrero's (1993) method is used,
where λ minimizes the coefficient of variation for subseries of `x`.

# Arguments
- `x::AbstractVector{<:Number}`: A numeric vector or time series.
- `m::Int`: The frequency of the data.
- `method::Symbol`: Choose the method to be used in calculating λ. Options are `:guerrero` or `:loglik`.
- `lower::Float64`: Lower limit for possible λ values.
- `upper::Float64`: Upper limit for possible λ values.
- `nonseasonal_length::Int` Length of non-seasonal components. Do not need to change.
- `is_ts::Bool` Is data time series?

# Details
If `method === :loglik`, the value of λ is chosen to maximize the profile
log likelihood of a linear model fitted to `x`. For non-seasonal data,
a linear time trend is fitted while for seasonal data, a linear
     time trend with seasonal dummy variables is used.

# Returns
A number indicating the Box-Cox transformation parameter.

# References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed), OTexts. <https://otexts.com/fpp3/>
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations.
JRSS B, 26, 211–246.
- Guerrero, V.M. (1993). Time-series analysis supported by power
transformations. Journal of Forecasting, 12, 37–48.
"""

function box_cox_lambda(x::AbstractVector{<:Number}, m::Int;
    method::Symbol=:guerrero, lower::Float64=-1.0, upper::Float64=2.0,
    nonseasonal_length::Int=2, is_ts::Bool=true)
    if any(v -> !ismissing(v) && !(v isa AbstractFloat && isnan(v)) && v <= 0, x)
        lower = max(lower, 0.0)
    end

    if length(x) <= 2 * m
        return 1.0
    end

    if method === :loglik
        return _box_cox_loglik(x, m, lower=lower, upper=upper, is_ts=is_ts)
    elseif method === :guerrero
        return _guerrero_lambda(x, m, lower, upper, nonseasonal_length)
    else
        throw(ArgumentError("Unknown method: $method. Choose either :loglik or :guerrero."))
    end
end

"""
    box_cox(x::AbstractVector{<:Number}, m::Int; lambda::Union{Symbol, Number}=:auto)

box_cox returns a transformation of the input variable using a Box-Cox transformation.

# Arguments
- `x::AbstractVector`: A numeric vector or time series.
- `lambda::Union{Symbol, Number}`: Transformation parameter.
If `:auto`, then the transformation parameter λ is chosen using `box_cox_lambda`
with a lower bound of -0.9.

# Details
The Box-Cox transformation as given by Bickel & Doksum 1981.

# Returns
A numeric vector of the same length as `x`.

# References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed), OTexts. <https://otexts.com/fpp3/>
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations. JRSS B, 26, 211–246.
- Bickel, P. J. and Doksum, K. A. (1981). An Analysis of Transformations Revisited. JASA, 76, 296–311.
"""
function box_cox(x::AbstractVector{<:Number}, m::Int; lambda::Union{Symbol,Number}=:auto)
    x = copy(x)
    if lambda === :auto
        lambda = box_cox_lambda(x, m, lower=-0.9)
    end
    if lambda < 0
        x[x.<0] .= NaN
    end

    if lambda == 0
        transformed = log.(x)
    else
        transformed = (sign.(x) .* abs.(x) .^ lambda .- 1) ./ lambda
    end

    return transformed, lambda
end

"""
    box_cox!(output::AbstractVector, x::AbstractVector, m::Int; lambda::Real)

In-place version of box_cox that writes to a pre-allocated output buffer.
Identical mathematical behavior but eliminates intermediate allocations.

# Arguments
- `output::AbstractVector`: Pre-allocated output buffer
- `x::AbstractVector`: Input vector
- `m::Int`: Frequency parameter (kept for API consistency)
- `lambda::Real`: Transformation parameter (must be numeric, not `:auto`)

# Returns
Tuple of (output, lambda) where output contains the transformed values.

# Note
This is a performance optimization that avoids ~6MB of allocations per call.
Use this in tight loops where box_cox is called repeatedly.
"""
function box_cox!(output::AbstractVector, x::AbstractVector, m::Int; lambda::Real)
    if lambda < 0
        copyto!(output, x)
        @inbounds for i in eachindex(output)
            if output[i] < 0
                output[i] = NaN
            end
        end
        x_work = output
    else
        x_work = x
    end

    if lambda == 0
        @. output = log(x_work)
    else
        @. output = (sign(x_work) * abs(x_work)^lambda - 1) / lambda
    end

    return output, lambda
end

"""
    inv_box_cox(x::AbstractArray; lambda::Real, biasadj::Union{Bool,Nothing}=false,
    fvar::Union{Nothing,Number,AbstractArray,Dict,NamedTuple}=nothing)

Reverses the Box-Cox transformation.

# Arguments
- `x::AbstractArray`: A numeric vector or time series.
- `lambda::Real`: Transformation parameter.
- `biasadj::Bool`: Use adjusted back-transformed mean for
Box-Cox transformations. If transformed data is used to
 produce forecasts and fitted values, a regular back transformation
  will result in median forecasts. If `biasadj` is `true`, an
  adjustment will be made to produce mean forecasts and fitted values.
- `fvar`: Optional parameter required if `biasadj = true`.
Can either be a numeric variance (Number), an array of forecast variances,
or a Dict/NamedTuple containing the interval level and corresponding upper and lower intervals.

# Returns
A numeric vector of the same length as `x`.


# References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed), OTexts. <https://otexts.com/fpp3/>
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations. JRSS B, 26, 211–246.
- Bickel, P. J. and Doksum, K. A. (1981). An Analysis of Transformations Revisited. JASA, 76, 296–311.
"""
function inv_box_cox(x::AbstractArray; lambda::Real, biasadj::Union{Bool,Nothing}=false,
    fvar::Union{Nothing,Number,AbstractArray,Dict,NamedTuple}=nothing)

    x_work = copy(x)
    if lambda < 0
        thresh = -1 / float(lambda)
        @inbounds for I in eachindex(x_work)
            if x_work[I] > thresh
                x_work[I] = NaN
            end
        end
    end

    out = similar(x_work, promote_type(eltype(x_work), Float64))
    if lambda == 0
        out .= exp.(x_work)
    else
        scaled = x_work .* lambda .+ 1
        out .= sign.(scaled) .* abs.(scaled).^(1 / lambda)
    end

    if isnothing(biasadj) || !(biasadj isa Bool)
        @warn "biasadj information not found, defaulting to false."
        biasadj = false
    end

    if biasadj
        forecast_variance = fvar
        isnothing(forecast_variance) && throw(ArgumentError("fvar must be provided when biasadj=true"))

        if (forecast_variance isa Dict) || (forecast_variance isa NamedTuple)
            level = maximum(forecast_variance[:level])
            upper = forecast_variance[:upper]
            lower = forecast_variance[:lower]

            if ndims(upper) == 2 && ndims(lower) == 2 && size(upper,2) > 1 && size(lower,2) > 1
                lvlvec = forecast_variance[:level]
                idx = findfirst(==(level), lvlvec)
                isnothing(idx) && throw(ArgumentError("Requested level $level not found in fvar[:level]"))
                upper = upper[:, idx]
                lower = lower[:, idx]
            end

            if level > 1
                level /= 100
            end
            level = mean((level, 1.0))

            q = dist_quantile(Normal(), level)
            forecast_variance = ((upper .- lower) ./ (q * 2)) .^ 2
        end

        if forecast_variance isa AbstractMatrix && size(forecast_variance, 2) > 1
            n = min(size(forecast_variance,1), size(forecast_variance,2))
            forecast_variance = [forecast_variance[i, i] for i in 1:n]
        end

        variance_adjusted = forecast_variance
        if forecast_variance isa AbstractVector
            L = length(forecast_variance)
            N = length(out)
            if L == 0
                throw(ArgumentError("fvar vector is empty"))
            end
            reps = cld(N, L)
            flat = repeat(forecast_variance, reps)[1:N]
            variance_adjusted = reshape(flat, size(out))
        elseif !(forecast_variance isa Number) && !(size(forecast_variance) == size(out))

            throw(DimensionMismatch("fvar shape $(size(forecast_variance)) is incompatible with out shape $(size(out)). Provide a vector for element-wise recycling or a same-shaped array."))
        end

        out .*= (1 .+ 0.5 .* variance_adjusted .* (1 .- lambda) ./ (out .^ (2 * lambda)))
    end

    return out
end
