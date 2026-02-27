"""
    wold_coefficients(phi, theta, maxlag)

Compute Wold (MA(infinity)) representation coefficients psi_0, psi_1, ..., psi_maxlag
from ARMA parameters via the recursion:

    psi_0 = 1
    psi_k = theta_k + sum_{j=1}^{min(k,p)} phi_j * psi_{k-j}

where theta_k = 0 for k > q.

Returns a vector of length `maxlag + 1` (indices 0:maxlag stored at 1:maxlag+1).
"""
function wold_coefficients(phi::AbstractVector, theta::AbstractVector, maxlag::Int)
    p = length(phi)
    q = length(theta)
    psi = zeros(maxlag + 1)
    psi[1] = 1.0  # psi_0
    for k in 1:maxlag
        val = k <= q ? theta[k] : 0.0
        for j in 1:min(k, p)
            val += phi[j] * psi[k - j + 1]
        end
        psi[k + 1] = val
    end
    return psi
end

"""
    arma_autocovariance(phi, theta, sigma2, maxlag)

Compute theoretical autocovariance gamma_0, gamma_1, ..., gamma_maxlag for a
stationary ARMA(p,q) process with innovation variance sigma2.

Uses the Wold representation: gamma_k = sigma2 * sum_{j=0}^{L} psi_j * psi_{j+k}
where L is chosen large enough for convergence (maxlag + 500).
"""
function arma_autocovariance(phi::AbstractVector, theta::AbstractVector, sigma2::Real, maxlag::Int)
    L = maxlag + 500
    psi = wold_coefficients(phi, theta, L)
    gamma = zeros(maxlag + 1)
    for k in 0:maxlag
        s = 0.0
        for j in 1:(L - k + 1)
            s += psi[j] * psi[j + k]
        end
        gamma[k + 1] = sigma2 * s
    end
    return gamma
end

"""
    extract_arma_for_kw(fit::ArimaFit)

Extract expanded AR (phi) and MA (theta) coefficients, innovation variance (sigma2),
and total integration order (d) from a fitted ARIMA model for use in the KW filter.

The AR and MA polynomials include seasonal expansion:
- phi includes both nonseasonal AR and seasonal AR coefficients from `fit.model.phi`
- theta includes both nonseasonal MA and seasonal MA coefficients from `fit.model.theta`
- d = nonseasonal d + seasonal D (from `fit.arma`)

Returns `(phi, theta, sigma2, d)`.
"""
function extract_arma_for_kw(fit::ArimaFit)
    # fit.arma = [p, q, P, Q, s, d, D]
    d_nonseasonal = fit.arma[6]
    D_seasonal = fit.arma[7]
    # Approximation: the paper assumes (1-L)^d but we collapse seasonal
    # differencing into the same order (d = d_ns + D).  The autocovariance γ
    # is still exact (computed from the expanded ARMA representation); only
    # the filter weight computation (Props 2–4) uses this simplified order.
    d = d_nonseasonal + D_seasonal

    # The state-space model stores the expanded (full) AR and MA polynomials
    phi = Float64.(fit.model.phi)
    theta = Float64.(fit.model.theta)
    sigma2 = Float64(fit.sigma2)

    return (phi, theta, sigma2, d)
end
