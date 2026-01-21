struct MeanFit
    mu::Number
    sd::Number
    lambda::Union{Nothing, Float64, String}
    fitted::AbstractArray
    residuals::AbstractArray
    m::Int
end

function meanf(y::AbstractArray, m::Int, lambda::Union{Nothing, Float64, String}=nothing)
    x = copy(y)
    if !isnothing(lambda)
        y, lambda = box_cox(y, m, lambda=lambda)
    end
    meanx = mean2(y, omit_na=true)
    fits = fill(meanx, length(x))
    res = x .- fits
    s = std(collect(skipmissing(x)))
    out = MeanFit(meanx, s, lambda, fits, res, m)
    return out
end

function forecast(object::MeanFit, h::Int=10, level::Vector{Float64}=[80.0, 95.0], fan::Bool=false, bootstrap::Bool=false, npaths::Int=5000)
    meanx = object.mu
    s = object.sd
    lambda = object.lambda
    m = object.m
    
    f = fill(meanx, h)
    
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
    lower = Matrix{Float64}(undef, h, nconf)
    upper = Matrix{Float64}(undef, h, nconf)
    
    if bootstrap
        e = collect(skipmissing(object.residuals)) .- mean(collect(skipmissing(object.residuals)))
        sim = [f .+ rand(Distributions.EmpiricalUnivariate(e), npaths) for _ in 1:h]
        for i in 1:nconf
            lower[:, i] = quantile.(sim, .5 .- level[i] / 200.0)
            upper[:, i] = quantile.(sim, .5 .+ level[i] / 200.0)
        end
    else
        n = length(collect(skipmissing(object.fitted)))
        for i in 1:nconf
            tfrac = quantile(TDist(n - 1), 0.5 - level[i] / 200.0)
            w = -tfrac * s * sqrt(1.0 + 1.0 / n)
            lower[:, i] .= f .- w
            upper[:, i] .= f .+ w
        end
    end
    
    # Inverse Box-Cox transformation if lambda is provided
    if !isnothing(lambda)
        f = inv_box_cox(f; lambda=lambda)
        lower .= inv_box_cox(lower; lambda=lambda)
        upper .= inv_box_cox(upper; lambda=lambda)
    end
    
    return Dict("mean" => f, "lower" => lower, "upper" => upper, "level" => level, "m" => m)
end