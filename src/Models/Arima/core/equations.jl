"""
Equation-derived ARIMA core primitives used by Durbyn's SARIMA implementation.

These helpers implement:
- polynomial convolution,
- differencing polynomial construction,
- multiplicative AR/MA polynomial expansion,
- companion-matrix root computation,
- AR stationarity checks,
- MA root reflection for invertibility,
- discrete Lyapunov solution for stationary state covariance.
"""

function arima_poly_convolution(a::AbstractVector, b::AbstractVector)
    na = length(a)
    nb = length(b)
    output = zeros(Float64, na + nb - 1)
    @inbounds for i in 1:na
        ai = Float64(a[i])
        for j in 1:nb
            output[i + j - 1] += ai * Float64(b[j])
        end
    end
    return output
end

function arima_differencing_delta(d::Int, D::Int, s::Int)
    difference_poly = Float64[1.0]

    for _ in 1:d
        difference_poly = arima_poly_convolution(difference_poly, Float64[1.0, -1.0])
    end

    if D > 0
        s > 0 || throw(ArgumentError("seasonal period s must be positive when D > 0"))
        seasonal_difference_poly = zeros(Float64, s + 1)
        seasonal_difference_poly[1] = 1.0
        seasonal_difference_poly[end] = -1.0
        for _ in 1:D
            difference_poly = arima_poly_convolution(difference_poly, seasonal_difference_poly)
        end
    end

    return -difference_poly[2:end]
end

function arima_expand_ar(nonseasonal_ar::AbstractVector, seasonal_ar::AbstractVector, seasonal_period::Int)
    nonseasonal_poly = vcat(1.0, -Float64.(nonseasonal_ar))
    if isempty(seasonal_ar) || seasonal_period <= 0
        return -nonseasonal_poly[2:end]
    end

    seasonal_poly = zeros(Float64, length(seasonal_ar) * seasonal_period + 1)
    seasonal_poly[1] = 1.0
    @inbounds for seasonal_index in eachindex(seasonal_ar)
        seasonal_poly[seasonal_index * seasonal_period + 1] = -Float64(seasonal_ar[seasonal_index])
    end

    full_ar_poly = arima_poly_convolution(nonseasonal_poly, seasonal_poly)
    return -full_ar_poly[2:end]
end

function arima_expand_ma(nonseasonal_ma::AbstractVector, seasonal_ma::AbstractVector, seasonal_period::Int)
    nonseasonal_poly = vcat(1.0, Float64.(nonseasonal_ma))
    if isempty(seasonal_ma) || seasonal_period <= 0
        return nonseasonal_poly[2:end]
    end

    seasonal_poly = zeros(Float64, length(seasonal_ma) * seasonal_period + 1)
    seasonal_poly[1] = 1.0
    @inbounds for seasonal_index in eachindex(seasonal_ma)
        seasonal_poly[seasonal_index * seasonal_period + 1] = Float64(seasonal_ma[seasonal_index])
    end

    full_ma_poly = arima_poly_convolution(nonseasonal_poly, seasonal_poly)
    return full_ma_poly[2:end]
end

function arima_polynomial_roots(coefficients::AbstractVector)
    coefficient_vector = ComplexF64.(coefficients)
    while length(coefficient_vector) > 1 && abs(coefficient_vector[end]) == 0.0
        pop!(coefficient_vector)
    end
    polynomial_degree = length(coefficient_vector) - 1
    polynomial_degree <= 0 && return ComplexF64[]

    companion = zeros(ComplexF64, polynomial_degree, polynomial_degree)
    @inbounds for row_index in 1:(polynomial_degree - 1)
        companion[row_index + 1, row_index] = 1.0
    end
    leading = coefficient_vector[end]
    @inbounds for row_index in 1:polynomial_degree
        companion[row_index, polynomial_degree] = -coefficient_vector[row_index] / leading
    end
    return eigvals(companion)
end

function arima_is_stationary(ar_coefficients::AbstractVector)
    isempty(ar_coefficients) && return true
    characteristic_coefficients = vcat(1.0, -Float64.(ar_coefficients))
    roots = arima_polynomial_roots(characteristic_coefficients)
    return all(abs.(roots) .> 1.0)
end

function arima_reflect_ma_roots(ma_coefficients::AbstractVector{<:Real})
    reflected = Float64.(ma_coefficients)
    trailing_nonzero_index = findlast(x -> abs(x) > 0.0, reflected)
    isnothing(trailing_nonzero_index) && return reflected

    active_order = Int(trailing_nonzero_index)
    active_coefficients = reflected[1:active_order]
    roots = arima_polynomial_roots(vcat(1.0, active_coefficients))
    changed = false
    for root_index in eachindex(roots)
        if abs(roots[root_index]) < 1.0
            roots[root_index] = inv(roots[root_index])
            changed = true
        end
    end
    changed || return reflected

    reconstructed = ComplexF64[1.0 + 0.0im]
    for root_value in roots
        reconstructed = vcat(reconstructed, 0.0 + 0.0im) .-
                        (vcat(0.0 + 0.0im, reconstructed) ./ root_value)
    end
    active_new_coefficients = real.(reconstructed[2:end])
    reflected[1:active_order] .= active_new_coefficients
    return reflected
end

function arima_solve_discrete_lyapunov(transition::AbstractMatrix, innovation_covariance::AbstractMatrix)
    state_dimension = size(transition, 1)
    transition_power = Matrix{Float64}(transition)
    covariance_accumulator = Matrix{Float64}(innovation_covariance)
    covariance_next = similar(covariance_accumulator)
    transition_power_next = similar(transition_power)
    workspace = similar(transition_power)

    max_iterations = 80
    absolute_tolerance = 1e-12
    relative_tolerance = 1e-10
    converged = false

    for _ in 1:max_iterations
        mul!(workspace, transition_power, covariance_accumulator)
        mul!(covariance_next, workspace, transpose(transition_power))

        diff_max = 0.0
        scale_max = 0.0
        @inbounds for index in eachindex(covariance_next)
            updated = covariance_next[index] + covariance_accumulator[index]
            delta = abs(updated - covariance_accumulator[index])
            covariance_next[index] = updated
            diff_max = max(diff_max, delta)
            scale_max = max(scale_max, abs(updated))
        end

        mul!(transition_power_next, transition_power, transition_power)
        transition_scale = opnorm(transition_power_next, Inf)

        if !isfinite(diff_max) || !isfinite(scale_max) || !isfinite(transition_scale)
            converged = false
            break
        end

        if diff_max <= absolute_tolerance + relative_tolerance * max(1.0, scale_max)
            converged = true
            covariance_accumulator, covariance_next = covariance_next, covariance_accumulator
            break
        end

        covariance_accumulator, covariance_next = covariance_next, covariance_accumulator
        transition_power, transition_power_next = transition_power_next, transition_power
    end

    if converged
        return Matrix(Symmetric(covariance_accumulator))
    end

    # Fallback to exact Kronecker solve for difficult near-unit-root cases.
    identity_matrix = Matrix{Float64}(I, state_dimension, state_dimension)
    kronecker_operator = kron(identity_matrix, identity_matrix) - kron(transition, transition)
    solution = reshape(kronecker_operator \ vec(innovation_covariance), state_dimension, state_dimension)
    return Matrix(Symmetric(solution))
end
