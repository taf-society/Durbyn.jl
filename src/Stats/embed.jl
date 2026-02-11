"""
    embed(x::AbstractVector, dimension::Integer=1) -> Matrix
    embed(x::AbstractMatrix, dimension::Integer=1) -> Matrix

Construct a time-delay embedding of a vector or each column of a matrix.

# Arguments
- `x`: A vector or a 2D matrix.
- `dimension`: Positive integer `d`. The embedding size (number of lagged columns).
  Must satisfy `1 ≤ d ≤ length(x)` for vectors and `1 ≤ d ≤ size(x, 1)` for matrices.

# Details
- **Vector input:** For a vector `x` of length `n`, returns a matrix with
  `m = n - d + 1` rows and `d` columns. Columns are ordered with the **largest lag first**,
  i.e. `[:, 1] = x[d:n]`, `[:, 2] = x[d-1:n-1]`, …, `[:, d] = x[1:m]`.
  This matches R's column order (`embed(x, d)`).

- **Matrix input:** For an `nxp` matrix, the operation is applied independently to
  each column, producing an `(n - d + 1) x (d * p)` matrix. Columns are **interleaved by variable then lag**:
  `col1_lag_d, col2_lag_d, …, colp_lag_d, col1_lag_(d-1), …, colp_lag_(d-1), …, col1_lag_1, …, colp_lag_1`.

# Value
A dense `Matrix` whose element type matches `x`'s element type.  
For vectors: size `(n - d + 1, d)`.  
For matrices: size `(n - d + 1, d * p)`.

# Errors
Throws `ArgumentError("wrong embedding dimension")` if `dimension < 1` or exceeds
the number of rows/elements of `x`.

# Notes
- The result is a standard `Matrix`; it does not carry time-series metadata.
- Missing values are not generated or removed; any `missing` in the input
  will appear in the appropriate output positions.
- This function is useful for building lagged feature matrices for regression,
  AR/ARIMA-style models, state-space reconstructions, and delay-coordinate embeddings.

# Examples
```julia
julia> x = [1, 2, 3, 4, 5];

julia> embed(x, 2)
4x2 Matrix{Int64}:
 2  1
 3  2
 4  3
 5  4

julia> X = [1 10; 2 20; 3 30; 4 40];

julia> embed(X, 2)  # size: (4-2+1) x (2*2) == 3 x 4
3x4 Matrix{Int64}:
 2  20  1  10
 3  30  2  20
 4  40  3  30
# columns: col1_lag2, col2_lag2, col1_lag1, col2_lag1
```
"""
function embed(x::AbstractVector{T}, dimension::Integer=1) where {T}
    n = length(x)
    d = dimension
    if d < 1 || d > n
        throw(ArgumentError("wrong embedding dimension"))
    end
    m = n - d + 1
    y = Matrix{T}(undef, m, d)
    for k in 1:d
        start = d - k + 1
        y[:, k] = view(x, start : start + m - 1)
    end
    return y
end

function embed(x::AbstractMatrix{T}, dimension::Integer=1) where {T}
    n, p = size(x)
    d = dimension
    if d < 1 || d > n
        throw(ArgumentError("wrong embedding dimension"))
    end
    rows = n - d + 1
    y = Matrix{T}(undef, rows, d * p)
    for j in 1:p
        col_embed = embed(view(x, :, j), d)
        for k in 1:d
            y[:, j + (k - 1) * p] = view(col_embed, :, k)
        end
    end
    return y
end