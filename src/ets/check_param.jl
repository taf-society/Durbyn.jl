function check_param(alpha::Union{Float64,Nothing,Bool}, beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool}, phi::Union{Float64,Nothing,Bool},
    lower::Vector{Float64}, upper::Vector{Float64}, bounds::String, m::Int)

    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if bounds != "admissible"
        if !isnothing(alpha) && !isnan(alpha)
            if alpha < lower[1] || alpha > upper[1]
                return false
            end
        end
        if !isnothing(beta) && !isnan(beta)
            if beta < lower[2] || beta > alpha || beta > upper[2]
                return false
            end
        end
        if !isnothing(phi) && !isnan(phi)
            if phi < lower[4] || phi > upper[4]
                return false
            end
        end
        if !isnothing(gamma) && !isnan(gamma)
            if gamma < lower[3] || gamma > 1 - alpha || gamma > upper[3]
                return false
            end
        end
    end

    if bounds != "usual"
        if !admissible(alpha, beta, gamma, phi, m)
            return false
        end
    end
    return true
end