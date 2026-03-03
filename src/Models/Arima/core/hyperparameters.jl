function transform_unconstrained_to_ar_params!(
    p::Int,
    raw::AbstractVector,
    new::AbstractVector,
)
    if p > 100
        throw(ArgumentError("The function can only transform 100 parameters in arima0"))
    end

    @inbounds new[1:p] .= tanh.(raw[1:p])
    work = copy(new[1:p])

    @inbounds for j = 2:p
        a = new[j]
        for k = 1:(j-1)
            work[k] -= a * new[j-k]
        end
        new[1:(j-1)] .= work[1:(j-1)]
    end

end

function compute_arima_transform_gradient(x::AbstractArray, arma::AbstractArray)
    eps = 1e-3
    mp, mq, msp = arma[1:3]
    if mp > 100 || msp > 100
        throw(ArgumentError("AR order > 100 not supported (p=$mp, P=$msp)"))
    end
    n = length(x)
    y = Matrix{Float64}(I, n, n)

    w1 = Vector{Float64}(undef, 100)
    w2 = Vector{Float64}(undef, 100)
    w3 = Vector{Float64}(undef, 100)

    if mp > 0

        for i = 1:mp
            w1[i] = x[i]
        end

        transform_unconstrained_to_ar_params!(mp, w1, w2)

        for i = 1:mp
            w1[i] += eps
            transform_unconstrained_to_ar_params!(mp, w1, w3)
            for j = 1:mp
                y[i, j] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end

    if msp > 0
        v = mp + mq
        for i = 1:msp
            w1[i] = x[i+v]
        end
        transform_unconstrained_to_ar_params!(msp, w1, w2)
        for i = 1:msp
            w1[i] += eps
            transform_unconstrained_to_ar_params!(msp, w1, w3)
            for j = 1:msp
                y[i+v, j+v] = (w3[j] - w2[j]) / eps
            end
            w1[i] -= eps
        end
    end
    return y
end

function undo_arima_parameter_transform(x::AbstractArray, arma::AbstractArray)
    mp, mq, msp = arma[1:3]
    res = copy(x)
    if mp > 0
        transform_unconstrained_to_ar_params!(mp, x, res)
    end
    v = mp + mq
    if msp > 0
        transform_unconstrained_to_ar_params!(msp, @view(x[v+1:end]), @view(res[v+1:end]))
    end
    return res
end

function time_series_convolution(a::AbstractArray, b::AbstractArray)
    na = length(a)
    nb = length(b)
    nab = na + nb - 1
    ab = zeros(Float64, nab)

    for i = 1:na
        for j = 1:nb
            ab[i+j-1] += a[i] * b[j]
        end
    end
    return ab
end

function inverse_ar_parameter_transform(ϕ::AbstractVector)
    p = length(ϕ)
    new = Array{Float64}(undef, p)
    copy!(new, ϕ)
    work = similar(new)
    for j in p:-1:2
        a = new[j]
        denom = 1 - a^2
        denom ≠ 0 || throw(ArgumentError("Encountered unit root at j=$j (a=±1)."))
        for k in 1:j-1
            work[k] = (new[k] + a * new[j-k]) / denom
        end
        new[1:j-1] = work[1:j-1]
    end
    return map(x -> abs(x) <= 1 ? atanh(x) : NaN, new)
end

function inverse_arima_parameter_transform(θ::AbstractVector, arma::AbstractVector{Int})
    mp, mq, msp = arma
    n = length(θ)
    v = mp + mq
    v + msp ≤ n || throw(ArgumentError("Sum mp+mq+msp exceeds length(θ)"))
    raw = Array{Float64}(undef, n)
    copy!(raw, θ)
    transformed = raw

    if mp > 0
        transformed[1:mp] = inverse_ar_parameter_transform(raw[1:mp])
    end

    if msp > 0
        transformed[v+1:v+msp] = inverse_ar_parameter_transform(raw[v+1:v+msp])
    end

    return transformed
end

function transform_arima_parameters(
    params_in::AbstractArray,
    arma::Vector{Int},
    trans::Bool,
)
mp, mq, msp, msq, ns = arma
    p = mp + ns * msp
    q = mq + ns * msq

    phi = zeros(Float64, p)
    theta = zeros(Float64, q)
    params = copy(params_in)

    if trans
        if mp > 0
            transform_unconstrained_to_ar_params!(mp, params_in, params)
        end
        v = mp + mq
        if msp > 0
            transform_unconstrained_to_ar_params!(msp, @view(params_in[v+1:end]), @view(params[v+1:end]))
        end
    end

    if ns > 0
        @inbounds phi[1:mp] .= params[1:mp]
        @inbounds theta[1:mq] .= params[mp+1:mp+mq]

        @inbounds for j = 0:(msp-1)
            phi[(j+1)*ns] += params[mp+mq+j+1]
            for i = 0:(mp-1)
                phi[((j+1)*ns)+(i+1)] -= params[i+1] * params[mp+mq+j+1]
            end
        end

        @inbounds for j = 0:(msq-1)
            theta[(j+1)*ns] += params[mp+mq+msp+j+1]
            for i = 0:(mq-1)
                theta[((j+1)*ns)+(i+1)] += params[mp+i+1] * params[mp+mq+msp+j+1]
            end
        end
    else
        @inbounds phi[1:mp] .= params[1:mp]
        @inbounds theta[1:mq] .= params[mp+1:mp+mq]
    end

    return (phi, theta)
end

function ar_check(ar)
    v = vcat(1.0, -ar...)
    last_nz = findlast(x -> x != 0.0, v)
    p = isnothing(last_nz) ? 0 : last_nz - 1
    if p == 0
        return true
    end

    coeffs = vcat(1.0, -ar[1:p]...)
    rts = roots(Polynomial(coeffs))

    return all(abs.(rts) .> 1.0)
end

function ma_invert(ma)
    q = length(ma)
    cdesc = vcat(1.0, ma...)
    nz = findall(x -> x != 0.0, cdesc)
    q0 = isempty(nz) ? 0 : maximum(nz) - 1
    if q0 == 0
        return ma
    end
    cdesc_q = cdesc[1:q0+1]
    rts = roots(Polynomial(cdesc_q))
    ind = abs.(rts) .< 1.0
    if all(.!ind)
        return ma
    end
    if q0 == 1
        return vcat(1.0 / ma[1], zeros(q - q0))
    end
    rts[ind] .= 1.0 ./ rts[ind]
    x = [1.0]
    for r in rts
        x = vcat(x, 0.0) .- (vcat(0.0, x) ./ r)
    end
    return vcat(real.(x[2:end]), zeros(q - q0))
end
