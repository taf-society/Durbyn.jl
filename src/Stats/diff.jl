function lag_series(x::AbstractVector, k::Int)
    n = length(x)
    result = fill(NaN, length(x))
    
    if k > 0
        for i in (k+1):n
            result[i] = x[i - k]
        end
    elseif k < 0
        for i in 1:(n + k)
            result[i] = x[i - k]
        end
    else
        result .= x
    end

    return result
end

"""
    diff(x::AbstractVector; lag::Int=1, differences::Int=1)

Compute the lagged differences of a vector `x`.

This function returns a new vector where each element is the result of subtracting
the value `lag` steps earlier, repeated `differences` times.

Missing values are inserted at the beginning to account for undefined differences.

# Arguments
- `x::AbstractVector`: Input data vector.
- `lag::Int=1`: The number of steps to lag when computing differences.
- `differences::Int=1`: The number of times to apply differencing recursively.

# Returns
A vector of the same length as `x`, with `missing` values in the positions where
insufficient data is available to compute the difference.

# Example
```julia
x = [1, 2, 4, 7, 11, 16]
diff(x; lag=1, differences=1)
# Returns: [missing, 1, 2, 3, 4, 5]
```
"""
function diff(x::AbstractVector; lag::Int=1, differences::Int=1)
    if lag < 1 || differences < 1
        throw(ArgumentError("Bad value for 'lag' or 'differences'"))
    end
    if lag * differences >= length(x)
        return x[1:0]
    end

    r = copy(x)
    for _ in 1:differences
        r = r .- lag_series(r, lag)
    end

    return r
end


"""
    diff(x::AbstractMatrix; lag::Int=1, differences::Int=1)

Compute lagged differences column-wise for a matrix `x`.

Each column is differenced independently as if it were a separate time series.
The result is a matrix of the same size, with `missing` values in the top rows where
the difference cannot be computed due to lag.

# Arguments
- `x::AbstractMatrix`: Input data matrix where rows represent time and columns represent variables.
- `lag::Int=1`: The lag interval used for differencing.
- `differences::Int=1`: Number of times to recursively apply the difference operation.

# Returns
A matrix of the same dimensions as `x`, with `missing` values in rows where differences
are not defined.

# Example
```julia
x = [
    1.0  10.0;
    2.0  20.0;
    4.0  30.0;
    7.0  40.0;
    11.0 50.0;
    16.0 60.0
]
diff(x; lag=1, differences=1)
# Returns:
# [missing missing;
#  1.0     10.0;
#  2.0     10.0;
#  3.0     10.0;
#  4.0     10.0;
#  5.0     10.0]
```
"""
function diff(x::AbstractMatrix; lag::Int=1, differences::Int=1)
    nrow, ncol = size(x)
    if lag < 1 || differences < 1
        throw(ArgumentError("Bad value for 'lag' or 'differences'"))
    end
    if lag * differences >= nrow
        return x[1:0, :]
    end

    r = Matrix{Float64}(x)
    tmp = similar(r)
    for _ in 1:differences
        for j in 1:ncol
            @views begin
                col = view(r, :, j)
                tmp[:, j] = col .- lag_series(col, lag)
            end
        end
        r, tmp = tmp, r
    end

    return r
end

"""
    diff(x::NamedMatrix; lag::Int=1, differences::Int=1) -> NamedMatrix

Apply lagged differencing to each column of a `NamedMatrix`, preserving column names
and row names (when present).
"""
function diff(x::NamedMatrix; lag::Int=1, differences::Int=1)
    data = diff(x.data; lag = lag, differences = differences)
    rownames = isnothing(x.rownames) ? nothing : copy(x.rownames)
    return NamedMatrix{eltype(data)}(data, rownames, copy(x.colnames))
end

