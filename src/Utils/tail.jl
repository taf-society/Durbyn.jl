"""
    _checkdims(A, dims) -> Int

Validate `dims` for array `A` and return it as an `Int`.
"""
@inline function _checkdims(A::AbstractArray, dims::Integer)
    nd = ndims(A)
    1 <= dims <= nd || throw(ArgumentError("dims must be between 1 and $nd (got $dims)"))
    return Int(dims)
end

"""
    _slicer(A, rng, dims) -> NTuple{N, Any}

Build an index tuple selecting `rng` along `dims` and `:` (all) along other dimensions.
"""
@inline function _slicer(A::AbstractArray, rng, dims::Integer)
    ntuple(i -> (i == dims ? rng : Colon()), ndims(A))
end

"""
    _keepcount(len::Int, n::Integer) -> Int

How many items to keep given length `len` and argument `n`:
- `n ≥ 0`  → keep `min(n, len)`
- `n < 0`  → keep `max(len + n, 0)`  (drop `-n` from the opposite end)
"""
@inline function _keepcount(len::Int, n::Integer)
    return n >= 0 ? min(n, len) : max(len + n, 0)
end

"""
    tail(A::AbstractArray, n::Integer=6; dims::Integer=1) -> SubArray

Return a **view** of the last `n` elements of `A` along dimension `dims`.
- If `n ≥ 0`, keep the last `n` along `dims` (clamped to size).
- If `n < 0`, drop the first `-n` along `dims` (keep `size(A,dims)+n`).

Use `copy(tail(...))` if you need a materialized array.

# Examples
```jldoctest
julia> v = collect(1:10);

julia> collect(tail(v))
6-element Vector{Int64}:
  5
  6
  7
  8
  9
 10

julia> M = reshape(1:12, 3, 4);
julia> collect(tail(M, 2))        # last 2 rows
2x4 Matrix{Int64}:
  2   5   8  11
  3   6   9  12

julia> collect(tail(M, 3; dims=2)) # last 3 columns
3x3 Matrix{Int64}:
  2   5   8
  3   6   9
  4   7  10
```
"""
function tail(A::AbstractArray, n::Integer=6; dims::Integer=1)
    d   = _checkdims(A, dims)
    len = size(A, d)
    keep = _keepcount(len, n)

    iN = lastindex(A, d)
    rng = keep == 0 ? (iN+1:iN) : (iN - keep + 1 : iN)

    return @view A[_slicer(A, rng, d)...]
end

"""
    tail(N::NamedMatrix, n::Integer=6; dims::Integer=1) -> NamedMatrix

Return the **last** `n` entries of `N` along dimension `dims`
(`1` = rows, `2` = columns). Negative `n` drops from the start
(e.g. `n = -k` drops the first `k` along `dims`).

Preserves `rownames`/`colnames` and returns a new `NamedMatrix`.
Assumes `NamedMatrix{T}` has a constructor `(data, rownames, colnames)`.
"""
function tail(N::NamedMatrix, n::Integer=6; dims::Integer=1)
    (dims == 1 || dims == 2) || throw(ArgumentError("dims must be 1 (rows) or 2 (cols)"))
    T = eltype(N.data)

    if dims == 1
        len  = size(N.data, 1)
        keep = _keepcount(len, n)
        rrng = keep == 0 ? (len+1:len) : (len - keep + 1 : len)
        sub  = @view N.data[rrng, :]
        rns  = N.rownames === nothing ? nothing : N.rownames[rrng]
        cns  = N.colnames
        return NamedMatrix{T}(Matrix{T}(sub), rns, cns)
    else
        len  = size(N.data, 2)
        keep = _keepcount(len, n)
        crng = keep == 0 ? (len+1:len) : (len - keep + 1 : len)
        sub  = @view N.data[:, crng]
        rns  = N.rownames
        cns  = N.colnames[crng]
        return NamedMatrix{T}(Matrix{T}(sub), rns, cns)
    end
end