function calculate_residuals(y::AbstractArray, m::Int, init_state::AbstractArray,
    errortype::String, trendtype::String, seasontype::String, damped::Bool,
    alpha::Union{Float64,Nothing,Bool}, beta::Union{Float64,Nothing,Bool},
    gamma::Union{Float64,Nothing,Bool}, phi::Union{Float64,Nothing,Bool}, nmse::Int)

    n = length(y)
    p = length(init_state)
    x = zeros(p * (n + 1))
    x[1:p] .= init_state
    e = zeros(n)
    amse = zeros(nmse)

    switch_dict = Dict("N" => 0, "A" => 1, "M" => 2)
    errortype, trendtype, seasontype = switch_dict[errortype], switch_dict[trendtype], switch_dict[seasontype]

    phi = ifelse(damped, phi, 1.0)
    beta = ifelse(trendtype == 0, 0.0, beta)
    gamma = ifelse(seasontype == 0, 0.0, gamma)

    alpha, beta, gamma, phi = Float64(alpha), Float64(beta), Float64(gamma), Float64(phi)

    likelihood = ets_base(y, n, x, m, errortype, trendtype, seasontype, alpha, beta, gamma, phi, e, amse, nmse)

    x = reshape(x, p, n + 1)'

    if abs(likelihood + 99999.0) < 1e-7
        likelihood = NaN
    end

    return Dict(
        "likelihood" => likelihood,
        "amse" => amse,
        "errors" => e,
        "state" => x
    )
end
