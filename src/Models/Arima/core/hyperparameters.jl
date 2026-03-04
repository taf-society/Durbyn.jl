# --- Jones 1980: Constrained ↔ unconstrained AR parameter transform ---

function transform_unconstrained_to_ar_params!(
    p::Int,
    raw::AbstractVector,
    new::AbstractVector,
)
    work = Vector{Float64}(undef, max(1, p))
    return transform_unconstrained_to_ar_params!(p, raw, new, work)
end

function transform_unconstrained_to_ar_params!(
    p::Int,
    raw::AbstractVector,
    new::AbstractVector,
    work::AbstractVector,
)
    if p > 100
        throw(ArgumentError("AR parameter transformation supports at most 100 parameters (got p=$p)"))
    end
    length(work) >= p || throw(ArgumentError("work buffer length ($(length(work))) is smaller than p=$p"))

    @inbounds for i in 1:p
        new[i] = tanh(raw[i])
        work[i] = new[i]
    end

    @inbounds for j = 2:p
        a = new[j]
        for k = 1:(j-1)
            work[k] -= a * new[j-k]
        end
        for k = 1:(j-1)
            new[k] = work[k]
        end
    end

end

# --- Monahan 1984: Inverse AR parameter transform ---

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

function ar_check(ar)
    trailing_nonzero_index = findlast(x -> abs(x) > 0.0, ar)
    if isnothing(trailing_nonzero_index)
        return true
    end
    active_ar = @view ar[1:trailing_nonzero_index]
    return arima_is_stationary(active_ar)
end

# --- SARIMA parameter transforms using SARIMAOrder ---

@inline function _full_ar_order(order::SARIMAOrder)
    return order.s > 0 ? (order.p + order.s * order.P) : order.p
end

@inline function _full_ma_order(order::SARIMAOrder)
    return order.s > 0 ? (order.q + order.s * order.Q) : order.q
end

@inline function _expand_multiplicative_ar!(
    phi::Vector{Float64},
    params::AbstractVector,
    order::SARIMAOrder,
)
    mp, mq, msp, ns = order.p, order.q, order.P, order.s
    fill!(phi, 0.0)

    if mp > 0
        @inbounds for i = 1:mp
            phi[i] = Float64(params[i])
        end
    end

    if msp > 0 && ns > 0
        seasonal_start = mp + mq
        @inbounds for seasonal_idx = 1:msp
            lag = seasonal_idx * ns
            seasonal_coef = Float64(params[seasonal_start + seasonal_idx])
            phi[lag] += seasonal_coef

            for ar_idx = 1:mp
                phi[lag + ar_idx] -= Float64(params[ar_idx]) * seasonal_coef
            end
        end
    end
    return phi
end

@inline function _expand_multiplicative_ma!(
    theta::Vector{Float64},
    params::AbstractVector,
    order::SARIMAOrder,
)
    mp, mq, msp, msq, ns = order.p, order.q, order.P, order.Q, order.s
    fill!(theta, 0.0)

    if mq > 0
        ma_start = mp
        @inbounds for i = 1:mq
            theta[i] = Float64(params[ma_start + i])
        end
    end

    if msq > 0 && ns > 0
        seasonal_start = mp + mq + msp
        @inbounds for seasonal_idx = 1:msq
            lag = seasonal_idx * ns
            seasonal_coef = Float64(params[seasonal_start + seasonal_idx])
            theta[lag] += seasonal_coef

            for ma_idx = 1:mq
                theta[lag + ma_idx] += Float64(params[mp + ma_idx]) * seasonal_coef
            end
        end
    end
    return theta
end

function transform_arima_parameters!(
    phi::Vector{Float64},
    theta::Vector{Float64},
    params::AbstractVector,
    order::SARIMAOrder,
    trans::Bool,
    nonseasonal_ar_work::AbstractVector,
    seasonal_ar_work::AbstractVector,
)
    expected_phi = _full_ar_order(order)
    expected_theta = _full_ma_order(order)
    length(phi) == expected_phi ||
        throw(ArgumentError("phi buffer has length $(length(phi)); expected $expected_phi"))
    length(theta) == expected_theta ||
        throw(ArgumentError("theta buffer has length $(length(theta)); expected $expected_theta"))

    mp, mq, msp = order.p, order.q, order.P
    if trans
        if mp > 0
            transform_unconstrained_to_ar_params!(
                mp,
                @view(params[1:mp]),
                @view(params[1:mp]),
                nonseasonal_ar_work,
            )
        end

        if msp > 0
            seasonal_start = mp + mq + 1
            seasonal_stop = seasonal_start + msp - 1
            transform_unconstrained_to_ar_params!(
                msp,
                @view(params[seasonal_start:seasonal_stop]),
                @view(params[seasonal_start:seasonal_stop]),
                seasonal_ar_work,
            )
        end
    end

    _expand_multiplicative_ar!(phi, params, order)
    _expand_multiplicative_ma!(theta, params, order)
    return (phi, theta)
end

function transform_arima_parameters(
    params_in::AbstractArray,
    order::SARIMAOrder,
    trans::Bool,
)
    params = Float64.(params_in)
    phi = Vector{Float64}(undef, _full_ar_order(order))
    theta = Vector{Float64}(undef, _full_ma_order(order))
    nonseasonal_ar_work = Vector{Float64}(undef, max(1, order.p))
    seasonal_ar_work = Vector{Float64}(undef, max(1, order.P))
    return transform_arima_parameters!(
        phi,
        theta,
        params,
        order,
        trans,
        nonseasonal_ar_work,
        seasonal_ar_work,
    )
end

function compute_arima_transform_gradient(x::AbstractArray, order::SARIMAOrder)
    eps = 1e-3
    mp, mq, msp = order.p, order.q, order.P
    if mp > 100 || msp > 100
        throw(ArgumentError("AR order exceeds maximum of 100 (p=$mp, P=$msp)"))
    end
    n = length(x)
    y = Matrix{Float64}(I, n, n)

    buffer_input = Vector{Float64}(undef, 100)
    buffer_base = Vector{Float64}(undef, 100)
    buffer_perturbed = Vector{Float64}(undef, 100)

    if mp > 0
        for i = 1:mp
            buffer_input[i] = x[i]
        end
        transform_unconstrained_to_ar_params!(mp, buffer_input, buffer_base)
        for i = 1:mp
            buffer_input[i] += eps
            transform_unconstrained_to_ar_params!(mp, buffer_input, buffer_perturbed)
            for j = 1:mp
                y[i, j] = (buffer_perturbed[j] - buffer_base[j]) / eps
            end
            buffer_input[i] -= eps
        end
    end

    if msp > 0
        v = mp + mq
        for i = 1:msp
            buffer_input[i] = x[i+v]
        end
        transform_unconstrained_to_ar_params!(msp, buffer_input, buffer_base)
        for i = 1:msp
            buffer_input[i] += eps
            transform_unconstrained_to_ar_params!(msp, buffer_input, buffer_perturbed)
            for j = 1:msp
                y[i+v, j+v] = (buffer_perturbed[j] - buffer_base[j]) / eps
            end
            buffer_input[i] -= eps
        end
    end
    return y
end

function undo_arima_parameter_transform(x::AbstractArray, order::SARIMAOrder)
    mp, mq, msp = order.p, order.q, order.P
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

function inverse_arima_parameter_transform(θ::AbstractVector, order::SARIMAOrder)
    mp, mq, msp = order.p, order.q, order.P
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
