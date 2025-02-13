function calculate_residuals(y::AbstractArray, m::Int, init_state::AbstractArray,
    errortype::String, trendtype::String, seasontype::String, damped::Bool,
    alpha::Union{Float64,Nothing,Bool}, beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool}, phi::Union{Float64,Nothing,Bool}, nmse::Int)

    n = length(y)
    p = length(init_state)
    x = fill(0.0, p * (n + 1))
    x[1:p] .= init_state
    e = fill(0.0, n)

    if !damped
        phi = 1.0
    end
    if trendtype == "N"
        beta = 0.0
    end
    if seasontype == "N"
        gamma = 0.0
    end

    amse = fill(0.0, nmse)

    switch_dict = Dict("N" => 0, "A" => 1, "M" => 2)

    errortype = switch_dict[errortype]
    trendtype = switch_dict[trendtype]
    seasontype = switch_dict[seasontype]

    if alpha isa Bool
        alpha = alpha ? 1.0 : 0.0
    end

    if beta isa Bool
        beta = beta ? 1.0 : 0.0
    end

    if gamma isa Bool
        gamma = gamma ? 1.0 : 0.0
    end

    if phi isa Bool
        phi = phi ? 1.0 : 0.0
    end

    likelihood = ets_base(y, n, x, m, errortype, trendtype, seasontype, alpha, beta, gamma, phi, e, amse, nmse)

    x = reshape_matrix_by_row(x, nrow = n + 1, ncol = p)

    if !isnan(likelihood)
        if abs(likelihood + 99999.0) < 1e-7
            likelihood = NaN
        end
    end

    return Dict(
        "likelihood" => likelihood,
        "amse" => amse,
        "errors" => e,
        "state" => x
    )
end

function reshape_matrix_by_row(x::AbstractArray; nrow::Int, ncol::Int)
    return reshape(x, ncol, nrow)'
end