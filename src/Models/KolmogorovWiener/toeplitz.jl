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

"""
    build_D_cumul_matrix(N)

Construct the (N-1) x N cumulation matrix D (Eq. 21, 25 in Schleicher 2002).

Lower-triangular ones in the first N-1 columns, zero in the last column:
    D[i, j] = 1  if j <= i
    D[i, j] = 0  otherwise

So (D*y)_i = y_1 + y_2 + ... + y_i (partial sums).
"""
function build_D_cumul_matrix(N::Int)
    D = zeros(N - 1, N)
    for i in 1:(N - 1)
        for j in 1:i
            D[i, j] = 1.0
        end
    end
    return D
end

"""
    build_M_matrix(n1, n2)

Construct the (N-1) x N block matrix M for Proposition 3 (Eq. 21, Schleicher 2002).

M has block structure:
    [0_{n1×1}  M1_{n1×n1}  0_{n1×n2}]
    [0_{n2×n1} M2_{n2×n2}  0_{n2×1} ]

where:
- M1 is upper triangular: -1 everywhere except last column which has -1/2
- M2 is lower triangular: 1 everywhere except first column which has 1/2
"""
function build_M_matrix(n1::Int, n2::Int)
    N = n1 + 1 + n2
    M = zeros(N - 1, N)

    # Top block: rows 1:n1, cols 2:(n1+1) → M1 (upper tri, -1 fill, -1/2 last col)
    for i in 1:n1
        for j in i:n1
            col = j + 1  # offset by 1 (column 1 is all zeros)
            M[i, col] = j == n1 ? -0.5 : -1.0
        end
    end

    # Bottom block: rows (n1+1):(N-1), cols (n1+1):(n1+n2) → M2 (lower tri, 1 fill, 1/2 first col)
    for i in 1:n2
        row = n1 + i
        for j in 1:i
            col = n1 + j
            M[row, col] = j == 1 ? 0.5 : 1.0
        end
    end

    return M
end

"""
    build_gamma_cross_integrated(gamma, N, Q)

Construct the (N-1) x (N-1+2Q) cross-covariance matrix for Proposition 4 (Eq. 25).

Entry (m, c) = γ_{|m + Q - c|}, connecting cumulated observations (rows)
with cumulated ideal filter positions (columns).
"""
function build_gamma_cross_integrated(gamma::AbstractVector, N::Int, Q::Int)
    maxlag = length(gamma) - 1
    nrows = N - 1
    ncols = N - 1 + 2Q
    G = zeros(nrows, ncols)
    for m in 1:nrows
        for c in 1:ncols
            # Column c maps to level-domain offset j = c - n1 - Q - 1, so level
            # position c - Q.  Lag between cumulated observation m and position
            # c - Q is |m - (c - Q)| = |m + Q - c|.
            lag = abs(m + Q - c)
            G[m, c] = lag <= maxlag ? gamma[lag + 1] : 0.0
        end
    end
    return G
end

"""
    build_C_vector(ideal_B, N, Q, n1)

Build the cumulated ideal coefficient vector C for Proposition 4 (Eq. 25).

C has length N-1+2Q. Column c in the integrated cross-covariance matrix
corresponds to ideal filter lag j = c - n1 - Q - 1, and:
- C_j = 0           for j < -Q
- C_j = Σ_{k=-Q}^j B_k  for -Q ≤ j ≤ Q
- C_j = β           for j > Q
"""
function build_C_vector(ideal_B::AbstractVector, N::Int, Q::Int, n1::Int)
    beta = sum(ideal_B)
    partial_sums = cumsum(ideal_B)  # partial_sums[k] = Σ_{i=1}^k B_i = Σ_{j=-Q}^{-Q+k-1} B_j
    ncols = N - 1 + 2Q
    C = zeros(ncols)
    for c in 1:ncols
        j = c - n1 - Q - 1
        if j < -Q
            C[c] = 0.0
        elseif j <= Q
            C[c] = partial_sums[j + Q + 1]
        else
            C[c] = beta
        end
    end
    return C
end
