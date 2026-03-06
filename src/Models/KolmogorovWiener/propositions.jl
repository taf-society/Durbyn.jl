"""
    proposition1_stationary(gamma, ideal_B, n1, n2)

Proposition 1 (Schleicher 2004): Optimal filter for a **stationary** process (d=0).

For observation at position t (with n1 obs before, n2 after), the optimal filter
weights B_hat satisfy:

    Gamma_hat * B_hat = Gamma * B

where Gamma_hat is the N x N sample autocovariance Toeplitz matrix and
Gamma * B is the cross-covariance between observations and the ideal filtered value.

Returns the filter weight vector of length N = n1 + 1 + n2.
"""
function proposition1_stationary(gamma::AbstractVector, ideal_B::AbstractVector, n1::Int, n2::Int)
    N = n1 + 1 + n2
    Q = div(length(ideal_B) - 1, 2)

    Gamma_hat = build_toeplitz_gamma_hat(gamma, N)
    Gamma_cross = build_gamma_cross(gamma, N, Q, n1)

    # Right-hand side: Gamma * B (N-vector)
    rhs = Gamma_cross * ideal_B

    # Solve Gamma_hat * B_hat = rhs
    B_hat = Gamma_hat \ rhs
    return B_hat
end

"""
    proposition2_random_walk(ideal_B, n1, n2)

Proposition 2 (Schleicher 2004): Optimal filter for a **random walk** (d=1)
with **white noise** stationary component (no ARMA structure).

Interior observations use ideal weights. Endpoint observations get modified weights
that redistribute the truncated tail sums to the nearest available observation.

Returns the filter weight vector of length N.
"""
function proposition2_random_walk(ideal_B::AbstractVector, n1::Int, n2::Int)
    N = n1 + 1 + n2
    Q = div(length(ideal_B) - 1, 2)
    center = Q + 1  # index of B_0 in ideal_B

    B_hat = zeros(N)

    # Copy ideal coefficients for lags that fall within the sample
    # lag j relative to position t maps to observation index n1+1+j
    # valid when 1 <= n1+1+j <= N, i.e., -n1 <= j <= n2
    # also j must be in [-Q, Q] for ideal_B to have a coefficient
    j_lo = max(-Q, -n1)
    j_hi = min(Q, n2)
    for j in j_lo:j_hi
        B_hat[n1 + 1 + j] = ideal_B[center + j]
    end

    # Left tail: ideal coefficients for j < -n1 (positions before first observation)
    # These get lumped onto the first observation
    if n1 < Q
        left_tail = 0.0
        for j in -Q:(-n1 - 1)
            left_tail += ideal_B[center + j]
        end
        B_hat[1] += left_tail
    end

    # Right tail: ideal coefficients for j > n2 (positions after last observation)
    # These get lumped onto the last observation
    if n2 < Q
        right_tail = 0.0
        for j in (n2 + 1):Q
            right_tail += ideal_B[center + j]
        end
        B_hat[N] += right_tail
    end

    return B_hat
end

"""
    proposition3_rw_symmetric(ideal_B, n1, n2, beta)

Proposition 3 (Schleicher 2004): Optimal filter for a **random walk** (d=1)
with **white noise** stationary component, for highpass/cycle extraction.

When beta = 0 (highpass), uses the same tail-redistribution as Prop 2.
When beta != 0, solves the augmented system [D; iota'] * B_hat = [rhs; beta].

Returns the filter weight vector of length N.
"""
function proposition3_rw_symmetric(ideal_B::AbstractVector, n1::Int, n2::Int, beta::Real)
    N = n1 + 1 + n2
    Q = div(length(ideal_B) - 1, 2)
    center = Q + 1

    if abs(beta) < 1e-15
        # Highpass (beta=0): same redistribution as Prop 2, sum is preserved at 0
        return proposition2_random_walk(ideal_B, n1, n2)
    end

    # General case (Eq. 21): solve [D; ι'] * B̂ = [M*B + (β/2)*τ; β]
    # D = cumulation matrix (N-1)×N, M = block matrix (N-1)×N, τ = ones(N-1)
    D = build_D_cumul_matrix(N)
    M = build_M_matrix(n1, n2)
    tau = ones(N - 1)

    # Map ideal filter coefficients into N-dimensional observation space
    B_ext = zeros(N)
    j_lo = max(-Q, -n1)
    j_hi = min(Q, n2)
    for j in j_lo:j_hi
        B_ext[n1 + 1 + j] = ideal_B[center + j]
    end

    rhs_upper = M * B_ext .+ (beta / 2) .* tau

    A = vcat(D, ones(1, N))
    rhs = vcat(rhs_upper, [beta])

    B_hat = A \ rhs
    return B_hat
end

"""
    proposition4_arima(gamma, ideal_B, n1, n2, beta)

Proposition 4 (Schleicher 2004): Optimal filter for an **ARIMA** process (d>=1)
with non-trivial ARMA stationary component.

Solves the augmented system (Eq. 25):
    [Γ̂D; ι'] * B̂ = [ΓC; β]

where Γ̂ is (N-1)×(N-1) autocovariance Toeplitz, D is (N-1)×N cumulation matrix,
Γ is (N-1)×(N-1+2Q) integrated cross-covariance, and C is the cumulated ideal
coefficient vector.

When `symmetric=true`, uses the symmetric Proposition 4 form:
    [Γ̂D; ι'] * B̂ = [Γ(MB + (β/2)τ); β]
with extended `M` and `τ` as in Schleicher (2004).

Returns the filter weight vector of length N.
"""
function proposition4_arima(gamma::AbstractVector, ideal_B::AbstractVector, n1::Int, n2::Int, beta::Real; symmetric::Bool=false)
    N = n1 + 1 + n2
    Q = div(length(ideal_B) - 1, 2)

    Gamma_hat = build_toeplitz_gamma_hat(gamma, N - 1)          # (N-1)×(N-1)
    D = build_D_cumul_matrix(N)                                  # (N-1)×N
    Gamma_cross = build_gamma_cross_integrated(gamma, N, Q)      # (N-1)×(N-1+2Q)

    # Γ̂D: (N-1)×N
    GammaD = Gamma_hat * D

    rhs_vector = if symmetric
        M_ext = build_M_matrix(n1 + Q, n2 + Q)                   # (N-1+2Q)×(N+2Q)
        B_ext = _build_extended_ideal_vector(ideal_B, n1, n2)    # (N+2Q)-vector
        tau_ext = _build_tau_symmetric_prop4(N, Q)               # (N-1+2Q)-vector
        M_ext * B_ext .+ (beta / 2) .* tau_ext
    else
        build_C_vector(ideal_B, N, Q, n1)                        # (N-1+2Q)-vector
    end

    # Γ * rhs_vector: (N-1)-vector
    rhs_upper = Gamma_cross * rhs_vector

    # [Γ̂D; ι'] B̂ = [ΓC; β]
    A = vcat(Matrix(GammaD), ones(1, N))
    rhs = vcat(rhs_upper, [beta])

    B_hat = A \ rhs
    return B_hat
end

@inline function _is_symmetric_filter(ideal_B::AbstractVector{<:Real}; atol::Float64=1e-10)
    n = length(ideal_B)
    isodd(n) || return false
    center = (n + 1) >>> 1
    @inbounds for j in 1:(center - 1)
        left = Float64(ideal_B[center - j])
        right = Float64(ideal_B[center + j])
        scale = max(abs(left), abs(right), 1.0)
        abs(left - right) <= atol * scale || return false
    end
    return true
end

function _build_extended_ideal_vector(ideal_B::AbstractVector{<:Real}, n1::Int, n2::Int)
    Q = div(length(ideal_B) - 1, 2)
    N = n1 + 1 + n2
    ncols = N + 2Q
    center = Q + 1
    B_ext = zeros(Float64, ncols)

    @inbounds for c in 1:ncols
        lag = c - (n1 + Q + 1)
        if -Q <= lag <= Q
            B_ext[c] = Float64(ideal_B[center + lag])
        end
    end
    return B_ext
end

function _build_tau_symmetric_prop4(N::Int, Q::Int)
    n = N - 1 + 2Q
    tau = fill(-1.0, n)
    split = Q + N - 1
    @inbounds for i in 1:split
        tau[i] = 1.0
    end
    return tau
end

"""
    compute_optimal_filter(gamma, ideal_B, n1, n2, d)

Dispatch to the appropriate proposition based on integration order `d`
and whether the stationary component is white noise.

- d=0: Proposition 1 (stationary)
    - d>=1 + white noise:
        - symmetric ideal filter   -> Proposition 3
        - non-symmetric ideal filter -> Proposition 2
    - d>=1 + ARMA structure:
        - Proposition 4 (symmetric variant when the ideal filter is symmetric)
"""
function compute_optimal_filter(gamma::AbstractVector, ideal_B::AbstractVector, n1::Int, n2::Int, d::Int)
    beta = sum(ideal_B)
    is_symmetric = _is_symmetric_filter(ideal_B)

    if d == 0
        return proposition1_stationary(gamma, ideal_B, n1, n2)
    end

    # Check if stationary component is white noise
    is_white_noise = true
    if length(gamma) > 1
        threshold = 1e-10 * abs(gamma[1])
        for k in Iterators.drop(eachindex(gamma), 1)
            if abs(gamma[k]) > threshold
                is_white_noise = false
                break
            end
        end
    end

    if is_white_noise
        if is_symmetric
            return proposition3_rw_symmetric(ideal_B, n1, n2, beta)
        end
        return proposition2_random_walk(ideal_B, n1, n2)
    else
        return proposition4_arima(gamma, ideal_B, n1, n2, beta; symmetric=is_symmetric)
    end
end
