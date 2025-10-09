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
    head(A::AbstractArray, n::Integer=6; dims::Integer=1) -> SubArray

Return a **view** of the first `n` elements of `A` along dimension `dims`.
- If `n ≥ 0`, keep the first `n` along `dims` (clamped to size).
- If `n < 0`, drop the last `-n` along `dims` (keep `size(A,dims)+n`).

Use `copy(head(...))` if you need a materialized array.

# Examples
```jldoctest
julia> v = collect(1:10);

julia> collect(head(v))
6-element Vector{Int64}:
 1
 2
 3
 4
 5
 6

julia> M = reshape(1:12, 3, 4);
julia> collect(head(M, 2))        # first 2 rows
2x4 Matrix{Int64}:
 1  4  7  10
 2  5  8  11

julia> collect(head(M, 2; dims=2)) # first 2 columns
3x2 Matrix{Int64}:
 1  4
 2  5
 3  6
```
"""
function head(A::AbstractArray, n::Integer=6; dims::Integer=1)
    d   = _checkdims(A, dims)
    len = size(A, d)
    keep = _keepcount(len, n)

    i1 = firstindex(A, d)
    stop = i1 + keep - 1
    rng = keep == 0 ? (i1:(i1-1)) : (i1:stop)

    return @view A[_slicer(A, rng, d)...]
end

"""
    head(N::NamedMatrix, n::Integer=6; dims::Integer=1) -> NamedMatrix

Return the **first** `n` entries of `N` along dimension `dims`
(`1` = rows, `2` = columns). Negative `n` drops from the opposite end
(e.g. `n = -k` drops the last `k` along `dims`).

Preserves `rownames`/`colnames` and returns a new `NamedMatrix`.
Assumes `NamedMatrix{T}` has a constructor `(data, rownames, colnames)`.

# Examples
```jldoctest
julia> data = reshape(1:12, 3, 4);
julia> nm = NamedMatrix(data, ["A","B","C","D"]);

julia> head(nm, 2)  # first 2 rows
# => NamedMatrix with 2x4 data, colnames preserved

julia> head(nm, 2; dims=2)  # first 2 columns
# => NamedMatrix with 3x2 data, colnames=["A","B"]
```
"""
function head(N::NamedMatrix, n::Integer=6; dims::Integer=1)
    (dims == 1 || dims == 2) || throw(ArgumentError("dims must be 1 (rows) or 2 (cols)"))
    T = eltype(N.data)

    if dims == 1
        len  = size(N.data, 1)
        keep = _keepcount(len, n)
        rrng = keep == 0 ? (1:0) : (1:keep)
        sub  = @view N.data[rrng, :]
        rns  = isnothing(N.rownames) ? nothing : N.rownames[rrng]
        cns  = N.colnames
        return NamedMatrix{T}(Matrix{T}(sub), rns, cns)
    else
        len  = size(N.data, 2)
        keep = _keepcount(len, n)
        crng = keep == 0 ? (1:0) : (1:keep)
        sub  = @view N.data[:, crng]
        rns  = N.rownames
        cns  = N.colnames[crng]
        return NamedMatrix{T}(Matrix{T}(sub), rns, cns)
    end
end