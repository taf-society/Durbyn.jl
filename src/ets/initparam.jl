function initparam(alpha::Union{Float64,Bool,Nothing}, beta::Union{Float64,Bool,Nothing},
    gamma::Union{Float64,Bool,Nothing}, phi::Union{Float64,Bool,Nothing},
    trendtype::String, seasontype::String,
    damped::Bool,
    lower::Vector{Float64}, upper::Vector{Float64},
    m::Int, bounds::String; nothing_as_nan::Bool=false)


    if bounds == "admissible"
        lower[1:3] .= 0
        upper[1:3] .= 1e-3
    elseif any(lower .> upper)
        throw(ArgumentError("Inconsistent parameter boundaries"))
    end

    # Select alpha
    if isnothing(alpha)
        alpha = lower[1] + 0.2 * (upper[1] - lower[1]) / m
        if alpha > 1 || alpha < 0
            alpha = lower[1] + 2e-3
        end
    else
        apha = 0.0
    end
    # Select beta
    if trendtype != "N" && (isnothing(beta))
        upper[2] = min(upper[2], alpha)
        beta = lower[2] + 0.1 * (upper[2] - lower[2])
        if beta < 0 || beta > alpha
            beta = alpha - 1e-3
        end
    end

    # Select gamma
    if seasontype != "N" && (isnothing(gamma))
        upper[3] = min(upper[3], 1 - alpha)
        gamma = lower[3] + 0.05 * (upper[3] - lower[3])
        if gamma < 0 || gamma > 1 - alpha
            gamma = 1 - alpha - 1e-3
        end
    end

    # Select phi
    if damped && isnothing(phi)
        phi = lower[4] + 0.99 * (upper[4] - lower[4])
        if phi < 0 || phi > 1
            phi = upper[4] - 1e-3
        end
    end

    if nothing_as_nan
        if isnothing(alpha)
            alpha = NaN
        end
        if isnothing(beta)
            beta = NaN
        end
        if isnothing(gamma)
            gamma = NaN
        end
        if isnothing(phi)
            phi = NaN
        end
    end

    return OrderedDict("alpha" => alpha, "beta" => beta, "gamma" => gamma, "phi" => phi)
end