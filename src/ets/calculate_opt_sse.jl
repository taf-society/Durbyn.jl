function calculate_opt_sse(p, select::AbstractArray, x::AbstractArray, lenx::Int,
    alpha::Union{Nothing,Float64,Bool}, beta::Union{Nothing,Float64,Bool},
    gamma::Union{Nothing,Float64,Bool}, seasonal::String, m::Int,
    exponential::Union{Nothing,Bool}, phi::Union{Nothing,Float64},
    l_start::Union{Nothing,AbstractArray}, b_start::Union{Nothing,AbstractArray},
    s_start::Union{Nothing,AbstractArray})

    if select[1] > 0
        alpha = p[1]
    end
    if select[2] > 0
        beta = p[1+select[1]]
    end
    if select[3] > 0
        gamma = p[1+select[1]+select[2]]
    end

    dotrend = (!isa(beta, Bool) || beta)
    doseasonal = (!isa(gamma, Bool) || gamma)

    out = simple_holt_winters(x, lenx, alpha=alpha, beta=beta, gamma=gamma, phi=phi,
        seasonal=seasonal, m=m, dotrend=dotrend, doseasonal=doseasonal, l_start=l_start,
        exponential=exponential, b_start=b_start, s_start=s_start)

    return out.SSE
end