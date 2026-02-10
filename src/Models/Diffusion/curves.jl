"""
    Diffusion Curve Functions

Functions to generate diffusion curves for each model type.
Each returns a NamedTuple with `cumulative` and `adoption` vectors.
"""

"""
    bass_curve(n, m, p, q) -> NamedTuple

Generate Bass diffusion curve for n periods.

# Arguments
- `n::Int`: Number of periods
- `m::Real`: Market potential (saturation point)
- `p::Real`: Coefficient of innovation (external influence)
- `q::Real`: Coefficient of imitation (internal influence)

# Returns
NamedTuple with:
- `cumulative::Vector{Float64}`: Cumulative adoption at each period
- `adoption::Vector{Float64}`: Adoption per period
- `innovators::Vector{Float64}`: Innovation component per period
- `imitators::Vector{Float64}`: Imitation component per period

# Formula
Cumulative: ``A_t = m \\cdot \\frac{1 - e^{-(p+q)t}}{1 + \\frac{q}{p} e^{-(p+q)t}}``
"""
function bass_curve(n::Int, m::Real, p::Real, q::Real)
    T = Float64
    m, p, q = T(m), T(p), T(q)

    cumulative = zeros(T, n)
    adoption = zeros(T, n)
    innovators = zeros(T, n)
    imitators = zeros(T, n)

    for t in 1:n
        exp_term = exp(-(p + q) * t)
        cumulative[t] = m * (1 - exp_term) / (1 + (q / p) * exp_term)
    end

    adoption[1] = cumulative[1]
    for t in 2:n
        adoption[t] = cumulative[t] - cumulative[t-1]
    end

    for t in 1:n
        innovators[t] = p * (m - cumulative[t])
        imitators[t] = adoption[t] - innovators[t]
    end

    return (cumulative=cumulative, adoption=adoption,
            innovators=innovators, imitators=imitators)
end

"""
    gompertz_curve(n, m, a, b) -> NamedTuple

Generate Gompertz diffusion curve for n periods.

# Arguments
- `n::Int`: Number of periods
- `m::Real`: Market potential (saturation point)
- `a::Real`: X-axis displacement coefficient
- `b::Real`: Growth rate parameter

# Returns
NamedTuple with:
- `cumulative::Vector{Float64}`: Cumulative adoption at each period
- `adoption::Vector{Float64}`: Adoption per period

# Formula
Cumulative: ``A_t = m \\cdot e^{-a \\cdot e^{-b \\cdot t}}``
"""
function gompertz_curve(n::Int, m::Real, a::Real, b::Real)
    T = Float64
    m, a, b = T(m), T(a), T(b)

    cumulative = zeros(T, n)
    adoption = zeros(T, n)

    for t in 1:n
        cumulative[t] = m * exp(-a * exp(-b * t))
    end

    adoption[1] = cumulative[1]
    for t in 2:n
        adoption[t] = cumulative[t] - cumulative[t-1]
    end

    return (cumulative=cumulative, adoption=adoption)
end

"""
    gsgompertz_curve(n, m, a, b, c) -> NamedTuple

Generate Gamma/Shifted Gompertz diffusion curve for n periods.

# Arguments
- `n::Int`: Number of periods
- `m::Real`: Market potential (saturation point)
- `a::Real`: Shape parameter (related to heterogeneity)
- `b::Real`: Scale parameter
- `c::Real`: Shifting parameter (c=1 gives Bass-like behavior)

# Returns
NamedTuple with:
- `cumulative::Vector{Float64}`: Cumulative adoption at each period
- `adoption::Vector{Float64}`: Adoption per period

# Formula
Cumulative: ``A_t = m \\cdot (1 - e^{-b \\cdot t})(1 + a \\cdot e^{-b \\cdot t})^{-c}``

# References
Bemmaor (1994) showed that when c=1, this reduces to a Bass-like curve.
"""
function gsgompertz_curve(n::Int, m::Real, a::Real, b::Real, c::Real)
    T = Float64
    m, a, b, c = T(m), T(a), T(b), T(c)

    cumulative = zeros(T, n)
    adoption = zeros(T, n)

    for t in 1:n
        exp_term = exp(-b * t)
        cumulative[t] = m * (1 - exp_term) * (1 + a * exp_term)^(-c)
    end

    adoption[1] = cumulative[1]
    for t in 2:n
        adoption[t] = cumulative[t] - cumulative[t-1]
    end

    return (cumulative=cumulative, adoption=adoption)
end

"""
    weibull_curve(n, m, a, b) -> NamedTuple

Generate Weibull diffusion curve for n periods.

# Arguments
- `n::Int`: Number of periods
- `m::Real`: Market potential (saturation point)
- `a::Real`: Scale parameter (controls timing)
- `b::Real`: Shape parameter (controls curve steepness)

# Returns
NamedTuple with:
- `cumulative::Vector{Float64}`: Cumulative adoption at each period
- `adoption::Vector{Float64}`: Adoption per period

# Formula
Cumulative: ``A_t = m \\cdot (1 - e^{-(t/a)^b})``
"""
function weibull_curve(n::Int, m::Real, a::Real, b::Real)
    T = Float64
    m, a, b = T(m), T(a), T(b)

    cumulative = zeros(T, n)
    adoption = zeros(T, n)

    for t in 1:n
        cumulative[t] = m * (1 - exp(-(t / a)^b))
    end

    adoption[1] = cumulative[1]
    for t in 2:n
        adoption[t] = cumulative[t] - cumulative[t-1]
    end

    return (cumulative=cumulative, adoption=adoption)
end

"""
    get_curve(model_type, n, params) -> NamedTuple

Dispatch to the appropriate curve function based on model type.

# Arguments
- `model_type::DiffusionModelType`: The type of diffusion model
- `n::Int`: Number of periods
- `params::NamedTuple`: Model parameters

# Returns
NamedTuple with cumulative and adoption vectors (and model-specific extras).
"""
function get_curve(model_type::DiffusionModelType, n::Int, params::NamedTuple)
    if model_type == Bass
        return bass_curve(n, params.m, params.p, params.q)
    elseif model_type == Gompertz
        return gompertz_curve(n, params.m, params.a, params.b)
    elseif model_type == GSGompertz
        return gsgompertz_curve(n, params.m, params.a, params.b, params.c)
    elseif model_type == Weibull
        return weibull_curve(n, params.m, params.a, params.b)
    else
        error("Unknown model type: $model_type")
    end
end
