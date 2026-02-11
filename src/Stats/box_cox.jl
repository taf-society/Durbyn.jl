function guer_cv(lam::Float64, x::Array{Float64,1}, m::Int; nonseasonal_length::Int=2)
    period = max(nonseasonal_length, m)
    nobsf = length(x)
    nyr = div(nobsf, period)
    nobst = nyr * period
    x_mat = reshape(x[(end-nobst+1):end], period, nyr)
    x_mean = vec(mean(x_mat, dims=1))
    x_sd = vec(std(x_mat, dims=1))
    x_rat = x_sd ./ x_mean .^ (1 - lam)
    return std(x_rat) / mean(x_rat)
end

function guerrero(x::Array{Float64,1}, m::Int, lower::Float64=-1.0, upper::Float64=2.0, nonseasonal_length::Int=2)
    if any(x .<= 0)
        println("Warning: Guerrero's method for selecting a Box-Cox parameter (lambda) is given for strictly positive data.")
    end

    result = Optimize.fmin(lam -> guer_cv(lam, x, m, nonseasonal_length=nonseasonal_length), lower, upper)
    return result.x_opt
end

function qr_residuals(X, y)
    Q, R = qr(X)
    a = Q' * y
    b = R \ a[1:size(R, 2)]
    c = X * b
    return y .- c
end

function bcloglik(x::AbstractArray, m::Int; lower::Float64 = -1.0, upper::Float64 = 2.0, is_ts::Bool = true)
    x = filter(!ismissing, x)
    n = length(x)
    
    if any(x .<= 0)
        error("x must be positive")
    end
    
    logx = log.(x)
    xdot = exp(mean(logx))

    if !is_ts
        
        X = ones(n, 1)
    else
        t = collect(1:n)
        if m > 1
            s = mod1.(t, m)
            D = zeros(n, m-1)
            for j in 2:m
                D[:, j-1] .= (s .== j)
            end
            X = hcat(ones(n), t, D)
        else
            X = hcat(ones(n), t)
        end
    end

    λs = collect(lower:0.05:upper)
    loglik = similar(λs)

    for (i, λ) in enumerate(λs)
        xt = if abs(λ) > 0.02
            (x .^ λ .- 1) ./ λ
        else
            logx .* (1 .+ (λ .* logx)/2 .* (1 .+ (λ .* logx)/3 .* (1 .+ (λ .* logx)/4)))
        end

        z = xt ./ xdot^(λ - 1)
        β = X \ z
        r = z .- X*β

        loglik[i] = -n/2 * log(sum(abs2, r))
    end

    return λs[argmax(loglik)]
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

    if length(collect(skipmissing(x))) <= 2 * m
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
    x = copy(x)  # don't mutate caller's data
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
    box_cox!(output::AbstractVector, x::AbstractVector, m::Int; lambda::Real)

In-place version of box_cox that writes to a pre-allocated output buffer.
Identical mathematical behavior but eliminates intermediate allocations.

# Arguments
- `output::AbstractVector`: Pre-allocated output buffer
- `x::AbstractVector`: Input vector
- `m::Int`: Frequency parameter (kept for API consistency)
- `lambda::Real`: Transformation parameter (must be numeric, not "auto")

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
- Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations. JRSS B, 26, 211–246.
- Bickel, P. J. and Doksum K. A. (1981). An Analysis of Transformations Revisited. JASA, 76, 296-311.
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
        xx = x_work .* lambda .+ 1
        out .= sign.(xx) .* abs.(xx).^(1 / lambda)
    end

    if isnothing(biasadj) || !(biasadj isa Bool)
        @warn "biasadj information not found, defaulting to false."
        biasadj = false
    end

    if biasadj
        fvar_local = fvar
        isnothing(fvar_local) && error("fvar must be provided when biasadj=true")

        if (fvar_local isa Dict) || (fvar_local isa NamedTuple)
            level = maximum(fvar_local[:level])
            upper = fvar_local[:upper]
            lower = fvar_local[:lower]

            if ndims(upper) == 2 && ndims(lower) == 2 && size(upper,2) > 1 && size(lower,2) > 1
                lvlvec = fvar_local[:level]
                idx = findfirst(==(level), lvlvec)
                isnothing(idx) && error("Requested level $level not found in fvar[:level]")
                upper = upper[:, idx]
                lower = lower[:, idx]
            end

            if level > 1
                level /= 100
            end
            level = mean((level, 1.0))

            q = dist_quantile(Normal(), level)
            fvar_local = ((upper .- lower) ./ (q * 2)) .^ 2
        end

        if fvar_local isa AbstractMatrix && size(fvar_local, 2) > 1
            n = min(size(fvar_local,1), size(fvar_local,2))
            fvar_local = [fvar_local[i, i] for i in 1:n] 
        end

        fvar_bc = fvar_local
        if fvar_local isa AbstractVector
            L = length(fvar_local)
            N = length(out)
            if L == 0
                error("fvar vector is empty")
            end
            reps = cld(N, L)
            flat = repeat(fvar_local, reps)[1:N]
            fvar_bc = reshape(flat, size(out))
        elseif !(fvar_local isa Number) && !(size(fvar_local) == size(out))
            
            throw(DimensionMismatch("fvar shape $(size(fvar_local)) is incompatible with out shape $(size(out)). Provide a vector for R-style recycling or a same-shaped array."))
        end

        out .*= (1 .+ 0.5 .* fvar_bc .* (1 .- lambda) ./ (out .^ (2 * lambda)))
    end

    return out
end
