"""
    _solve_stationary_covariance(phi, theta) -> Matrix{Float64}

Compute the stationary (unconditional) covariance matrix P₀ of the ARMA
companion-form state vector by solving the discrete Lyapunov equation

    P = T P T' + R R'

where T is the companion transition matrix (Harvey 1989, §3.3.3) and
R = [1, θ₁, …, θ_{r-1}]'.

Uses the Kronecker identity vec(P) = (I - T⊗T)⁻¹ vec(RR')
(Hamilton 1994, §10.2, eq 10.2.18).
"""
function _solve_stationary_covariance(phi::AbstractVector, theta::AbstractVector)
    p, q = length(phi), length(theta)
    r = max(p, q + 1)

    # Scalar AR(1) / white-noise: direct closed-form
    if r == 1
        val = p > 0 ? 1.0 / (1.0 - phi[1]^2) : 1.0
        return fill(val, 1, 1)
    end

    # Companion transition matrix T  (r × r)
    T = zeros(Float64, r, r)
    for i in 1:min(p, r)
        T[1, i] = phi[i]
    end
    for i in 2:r
        T[i, i-1] = 1.0
    end

    # Noise loading vector R = [1, θ₁, …, θ_{q}] padded to length r
    R_vec = zeros(Float64, r)
    R_vec[1] = 1.0
    for i in 1:min(q, r - 1)
        R_vec[i+1] = theta[i]
    end
    Q = R_vec * R_vec'

    # Solve discrete Lyapunov: P = T P T' + Q
    r2 = r * r
    P_vec = (Matrix{Float64}(I, r2, r2) - kron(T, T)) \ vec(Q)
    P = reshape(P_vec, r, r)

    return Matrix(Symmetric(P))
end
