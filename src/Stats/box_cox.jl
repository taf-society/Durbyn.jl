export box_cox_lambda, box_cox, inv_box_cox

function guer_cv(lam::Float64, x::Array{Float64,1}, m::Int; nonseasonal_length::Int=2)
    period = max(nonseasonal_length, m)
    nobsf = length(x)
    nyr = div(nobsf, period)
    nobst = nyr * period
    x_mat = reshape(x[(end-nobst+1):end], period, nyr)
    x_mean = vec(Statistics.mean(x_mat, dims=1))
    x_sd = vec(std(x_mat, dims=1))
    x_rat = x_sd ./ x_mean .^ (1 - lam)
    return std(x_rat) / mean(x_rat)
end

function guerrero(x::Array{Float64,1}, m::Int, lower::Float64=-1.0, upper::Float64=2.0, nonseasonal_length::Int=2)
    if any(x .<= 0)
        println("Warning: Guerrero's method for selecting a Box-Cox parameter (lambda) is given for strictly positive data.")
    end

    result = optimize(lam -> guer_cv(lam, x, m, nonseasonal_length=nonseasonal_length), lower, upper)
    return Optim.minimizer(result)
end

function qr_residuals(X, y)
    Q, R = qr(X)
    a = Q' * y
    b = R \ a[1:size(R, 2)]
    c = X * b
    return y .- c
end

function bcloglik(x::Vector{Float64}, m::Int; lower::Float64=-1.0, upper::Float64=2.0, is_ts=true)
    n = length(x)
    if any(x .<= 0)
        error("x must be positive")
    end

    x = filter(!ismissing, x)
    logx = log.(x)
    xdot = exp(mean(logx))

    if !is_ts
        fit = lm(@formula(x ~ 1), DataFrame(x=x))
    else
        if m > 1
            trend = 1:length(x)
            season = mod1.(1:length(x), m)
            df = DataFrame(x=x, trend=trend, season=CategoricalArray(season))
            fit = lm(@formula(x ~ trend + season), df)
        else
            trend = 1:length(x)
            fit = lm(@formula(x ~ trend), DataFrame(x=x, trend=trend))
        end
    end

    X = modelmatrix(fit)
    lambda = lower:0.05:upper
    xl = loglik = zeros(length(lambda))

    for (i, la) in enumerate(lambda)
        if abs(la) > 0.02
            xt = (x .^ la .- 1) / la
        else
            xt = logx .* (1 .+ (la .* logx) / 2 .* (1 .+ (la .* logx) / 3 .* (1 .+ (la .* logx) / 4)))
        end
        resid = qr_residuals(X, xt ./ xdot^(la - 1))
        loglik[i] = -n / 2 * log(sum(resid .^ 2))
    end

    return lambda[argmax(loglik)]
end

"""
    box_cox_lambda(x::AbstractVector, m::Int; method::String = "guerrero", lower::Real = -1, upper::Real = 2)

Automatic selection of Box-Cox transformation parameter.

# Description
If `method == "guerrero"`, Guerrero's (1993) method is used, 
where λ minimizes the coefficient of variation for subseries of `x`.

# Arguments
- `x::AbstractVector{<:Number}`: A numeric vector or time series.
- `m::Int`: The frequency of the data.
- `method::String`: Choose the method to be used in calculating λ. Options are `"guerrero"` or `"loglik"`.
- `lower::Float64`: Lower limit for possible λ values.
- `upper::Float64`: Upper limit for possible λ values.
- `nonseasonal_length::Int` Lenght of non-seasonal componants. Do not need to change.
- `is_ts::Bool` Is data time series?

# Details
If `method == "loglik"`, the value of λ is chosen to maximize the profile 
log likelihood of a linear model fitted to `x`. For non-seasonal data, 
a linear time trend is fitted while for seasonal data, a linear
     time trend with seasonal dummy variables is used.

# Returns
A number indicating the Box-Cox transformation parameter.

# References
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations. 
JRSS B, 26, 211–246.
- Guerrero, V.M. (1993). Time-series analysis supported by power 
transformations. Journal of Forecasting, 12, 37–48.
"""

function box_cox_lambda(x::AbstractVector{<:Number}, m::Int;
    method::String="guerrero", lower::Float64=-1.0, upper::Float64=2.0,
    nonseasonal_length::Int=2, is_ts::Bool=true)
    if any(collect(skipmissing(x)) .<= 0)
        lower = max(lower, 0.0)
    end

    if length(collect(skipmissing(x))) <= 2 * 12  # Assuming monthly data. Adjust as needed.
        return 1.0
    end

    if method == "loglik"
        return bcloglik(x, m, lower=lower, upper=upper, is_ts=is_ts)
    elseif method == "guerrero"
        return guerrero(x, m, lower, upper, nonseasonal_length)
    else
        error("Unknown method: $method. Choose either 'loglik' or 'guerrero'.")
    end
end

"""
    box_cox(x::AbstractVector{<:Number}, m::Int; lambda::Union{String, Number}="auto")

box_cox returns a transformation of the input variable using a Box-Cox transformation.

# Arguments
- `x::AbstractVector`: A numeric vector or time series.
- `lambda::Union{String, Number}`: Transformation parameter. 
If `"auto"`, then the transformation parameter λ is chosen using `box_cox_lambda` 
with a lower bound of -0.9.

# Details
The Box-Cox transformation as given by Bickel & Doksum 1981.
# Returns
A numeric vector of the same length as `x`.
"""
function box_cox(x::AbstractVector{<:Number}, m::Int; lambda::Union{String,Number}="auto")
    if lambda == "auto"
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
    inv_box_cox(x::AbstractVector, lambda::Real, biasadj::Bool = false, fvar = nothing)

Reverses the Box-Cox transformation.

# Arguments
- `x::AbstractVector`: A numeric vector or time series.
- `lambda::Real`: Transformation parameter.
- `biasadj::Bool`: Use adjusted back-transformed mean for 
Box-Cox transformations. If transformed data is used to
 produce forecasts and fitted values, a regular back transformation
  will result in median forecasts. If `biasadj` is `true`, an 
  adjustment will be made to produce mean forecasts and fitted values.
- `fvar`: Optional parameter required if `biasadj = true`. 
Can either be the forecast variance, or a list containing the interval level, a
nd the corresponding upper and lower intervals.

# Returns
A numeric vector of the same length as `x`.


# References
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations. JRSS B, 26, 211–246.
- Bickel, P. J. and Doksum K. A. (1981). An Analysis of Transformations Revisited. JASA, 76, 296-311.
"""

function inv_box_cox(x::AbstractVector; lambda::Float64, biasadj::Bool=false, fvar::Union{Nothing,Float64}=nothing)
    if lambda < 0
        x[x.>-1/lambda] .= NaN
    end

    if lambda == 0
        out = exp.(x)
    else
        xx = x * lambda .+ 1
        out = sign.(xx) .* abs.(xx) .^ (1 / lambda)
    end

    if !isa(biasadj, Bool)
        @warn "biasadj information not found, defaulting to false."
        biasadj = false
    end

    if biasadj
        if isnothing(fvar)
            error("fvar must be provided when biasadj=true")
        end

        if typeof(fvar) <: Dict
            level = maximum(fvar[:level])
            if size(fvar[:upper], 2) > 1 && size(fvar[:lower], 2) > 1
                i = findfirst(x -> x == level, fvar[:level])
                fvar[:upper] = fvar[:upper][:, i]
                fvar[:lower] = fvar[:lower][:, i]
            end
            if level > 1
                level /= 100
            end
            level = mean([level, 1])
            fvar = (fvar[:upper] - fvar[:lower]) / quantile(Normal(), level) / 2
            fvar = fvar .^ 2
        end

        if size(fvar, 2) > 1
            fvar = diagm(fvar)
        end

        out .= out .* (1 .+ 0.5 * fvar .* (1 .- lambda) ./ (out .^ (2 * lambda)))
    end

    return out
end