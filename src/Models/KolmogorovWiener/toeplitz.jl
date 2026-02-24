"""
    build_toeplitz_gamma_hat(gamma, N)

Construct the N x N symmetric Toeplitz autocovariance matrix Gamma_hat
where entry (i,j) = gamma_{|i-j|}.

This is the autocovariance matrix of the observed sample {y_1, ..., y_N}.
"""
function build_toeplitz_gamma_hat(gamma::AbstractVector, N::Int)
    maxlag = length(gamma) - 1
    G = zeros(N, N)
    for i in 1:N
        for j in i:N
            lag = j - i
            G[i, j] = lag <= maxlag ? gamma[lag + 1] : 0.0
            G[j, i] = G[i, j]
        end
    end
    return Symmetric(G)
end

"""
    build_gamma_cross(gamma, N, Q, n1)

Construct the N x (2Q+1) cross-covariance matrix Gamma for the KW filter.

For observation at position t = n1+1, the entry (m, c) gives:
    Cov(y_m, y_{t+j}) = gamma(|m - t - j|)

where c = Q+1+j indexes into the ideal filter coefficient vector (j = -Q:Q).
"""
function build_gamma_cross(gamma::AbstractVector, N::Int, Q::Int, n1::Int)
    maxlag = length(gamma) - 1
    ncols = 2 * Q + 1
    G = zeros(N, ncols)
    for m in 1:N
        for c in 1:ncols
            # c = Q+1+j, so j = c - Q - 1
            # lag = |m - (n1+1) - j| = |m - n1 - 1 - c + Q + 1| = |m + Q - n1 - c|
            lag = abs(m + Q - n1 - c)
            G[m, c] = lag <= maxlag ? gamma[lag + 1] : 0.0
        end
    end
    return G
end

"""
    build_D_matrix(N)

Construct the (N-1) x N first-difference matrix D (Proposition 2/4).

D is a bidiagonal matrix such that D*y computes first differences:
    (D*y)_i = y_{i+1} - y_i
"""
function build_D_matrix(N::Int)
    D = zeros(N - 1, N)
    for i in 1:(N - 1)
        D[i, i] = -1.0
        D[i, i + 1] = 1.0
    end
    return D
end
