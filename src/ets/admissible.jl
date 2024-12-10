function admissible(alpha::Union{Float64,Nothing,Bool}, beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool}, phi::Union{Float64,Nothing,Bool}, m::Int)

    alpha = normalize_parameter(alpha)
    beta = normalize_parameter(beta)
    gamma = normalize_parameter(gamma)
    phi = normalize_parameter(phi)

    if isnothing(phi)
        phi = 1.0
    elseif phi isa Bool
        phi = phi ? 1.0 : 0.0
    end

    if phi < 0 || phi > 1 + 1e-8
        return false
    end

    if isnothing(gamma)
        if alpha isa Bool
            alpha = alpha ? 1.0 : 0.0
        end
        if alpha < 1 - 1 / phi || alpha > 1 + 1 / phi
            return false
        end

        if !isnothing(beta)
            if beta isa Bool
                beta = beta ? 1.0 : 0.0
            end
            if beta < alpha * (phi - 1) || beta > (1 + phi) * (2 - alpha)
                return false
            end
        end
    elseif m > 1  # Seasonal model
        if isnothing(beta)
            beta = 0.0
        elseif beta isa Bool
            beta = beta ? 1.0 : 0.0
        end

        if gamma isa Bool
            gamma = gamma ? 1.0 : 0.0
        end
        if gamma < max(1 - 1 / phi - alpha, 0.0) || gamma > 1 + 1 / phi - alpha
            return false
        end

        if alpha isa Bool
            alpha = alpha ? 1.0 : 0.0
        end
        if alpha < 1 - 1 / phi - gamma * (1 - m + phi + phi * m) / (2 * phi * m)
            return false
        end

        if beta < -(1 - phi) * (gamma / m + alpha)
            return false
        end

        a = phi * (1 - alpha - gamma)
        b = alpha + beta - alpha * phi + gamma - 1
        c = repeat([alpha + beta - alpha * phi], m - 2)
        d = alpha + beta - phi
        P = vcat([a, b], c, [d, 1])

        poly = Polynomial(P)
        poly_roots = roots(poly)

        if maximum(abs.(poly_roots)) > 1 + 1e-10
            return false
        end
    end

    # Passed all tests
    return true
end