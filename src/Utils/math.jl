"""
    row_sums(M::AbstractMatrix; na_rm::Bool=false, nan_rm::Bool=false) -> Vector
    row_sums(X::NamedMatrix;    na_rm::Bool=false, nan_rm::Bool=false) -> Vector

Row-wise sums.

- If `na_rm=false` and a row contains `missing`, that row’s result is `missing`.
- If `nan_rm=false` and a row contains `NaN` (in float data), that row’s result is `NaN`.
- Set `na_rm=true` and/or `nan_rm=true` to skip those entries when summing.

The element type of the result is `Union{Missing,T}` when `na_rm=false`,
and `T` when `na_rm=true`, where `T` is a promoted real type (typically `Float64`).

# Examples
```julia
julia> M = Union{Missing,Float64}[1 2 missing; 4 missing 6];

julia> row_sums(M)
2-element Vector{Union{Missing, Float64}}:
  missing
  missing

julia> row_sums(M; na_rm=true)
2-element Vector{Float64}:
 3.0
 10.0

julia> N = [1.0 2.0 NaN; 4.0 5.0 6.0];

julia> row_sums(N)
2-element Vector{Float64}:
 NaN
 15.0

julia> row_sums(N; nan_rm=true)
2-element Vector{Float64}:
 3.0
 15.0

julia> nm = NamedMatrix{Float64}(N, ["r1","r2"], ["c1","c2","c3"]);

julia> row_sums(nm)
2-element Vector{Float64}:
 NaN
 15.0
````

"""
function row_sums(M::AbstractMatrix; na_rm::Bool=false)

    nan_rm = na_rm
    Telt = nonmissingtype(eltype(M))
    Tsum = promote_type(Float64, Telt)

    if na_rm
        out = Vector{Tsum}(undef, size(M,1))
        @inbounds for i in 1:size(M,1)
            s = zero(Tsum); has_nan = false
            for j in 1:size(M,2)
                v = M[i,j]
                if v === missing
                    continue
                elseif v isa AbstractFloat && isnan(v)
                    if nan_rm
                        continue
                    else
                        has_nan = true
                        break
                    end
                else
                    s += v
                end
            end
            out[i] = has_nan ? Tsum(NaN) : s
        end
        return out
    else
        out = Vector{Union{Missing,Tsum}}(undef, size(M,1))
        @inbounds for i in 1:size(M,1)
            s = zero(Tsum); has_missing = false; has_nan = false
            for j in 1:size(M,2)
                v = M[i,j]
                if v === missing
                    has_missing = true
                    break
                elseif v isa AbstractFloat && isnan(v)
                    has_nan = true
                    break
                else
                    s += v
                end
            end
            out[i] = has_nan ? Tsum(NaN) : (has_missing ? missing : s)
        end
        return out
    end
end

function row_sums(X::NamedMatrix; na_rm::Bool=false)
    row_sums(X.data; na_rm = na_rm)
end


"""
    col_sums(M::AbstractMatrix; na_rm::Bool=false, nan_rm::Bool=false) -> Vector
    col_sums(X::NamedMatrix;    na_rm::Bool=false, nan_rm::Bool=false) -> Vector

Column-wise sums.

- If `na_rm=false` and a column contains `missing`, that column’s result is `missing`.
- If `nan_rm=false` and a column contains `NaN` (in float data), that column’s result is `NaN`.
- Set `na_rm=true` and/or `nan_rm=true` to skip those entries when summing.

The element type of the result is `Union{Missing,T}` when `na_rm=false`,
and `T` when `na_rm=true`, where `T` is a promoted real type (typically `Float64`).

# Examples
```julia
julia> M = Union{Missing,Float64}[1 2 missing; 4 missing 6];

julia> col_sums(M)
3-element Vector{Union{Missing, Float64}}:
 5.0
  missing
  missing

julia> col_sums(M; na_rm=true)
3-element Vector{Float64}:
 5.0
 2.0
 6.0

julia> N = [1.0 2.0 NaN; 4.0 5.0 6.0];

julia> col_sums(N)
3-element Vector{Float64}:
 5.0
 7.0
 NaN

julia> col_sums(N; nan_rm=true)
3-element Vector{Float64}:
 5.0
 7.0
 6.0

julia> nm = NamedMatrix{Float64}(N, ["r1","r2"], ["c1","c2","c3"]);

julia> col_sums(nm; nan_rm=true)
3-element Vector{Float64}:
 5.0
 7.0
 6.0
````

"""
function col_sums(M::AbstractMatrix; na_rm::Bool=false)

    nan_rm = na_rm
    Telt = nonmissingtype(eltype(M))
    Tsum = promote_type(Float64, Telt)

    if na_rm
        out = Vector{Tsum}(undef, size(M,2))
        @inbounds for j in 1:size(M,2)
            s = zero(Tsum); has_nan = false
            for i in 1:size(M,1)
                v = M[i,j]
                if v === missing
                    continue
                elseif v isa AbstractFloat && isnan(v)
                    if nan_rm
                        continue
                    else
                        has_nan = true
                        break
                    end
                else
                    s += v
                end
            end
            out[j] = has_nan ? Tsum(NaN) : s
        end
        return out
    else
        out = Vector{Union{Missing,Tsum}}(undef, size(M,2))
        @inbounds for j in 1:size(M,2)
            s = zero(Tsum); has_missing = false; has_nan = false
            for i in 1:size(M,1)
                v = M[i,j]
                if v === missing
                    has_missing = true
                    break
                elseif v isa AbstractFloat && isnan(v)
                    has_nan = true
                    break
                else
                    s += v
                end
            end
            out[j] = has_nan ? Tsum(NaN) : (has_missing ? missing : s)
        end
        return out
    end
end

function col_sums(X::NamedMatrix; na_rm::Bool=false)
    col_sums(X.data; na_rm = na_rm)
end

"""
    row_sums(M::AbstractMatrix{<:Real}) -> Vector{T}
    col_sums(M::AbstractMatrix{<:Real}) -> Vector{T}

Fast paths for real-valued matrices with no `missing`.  
Equivalent to `sum(M, dims=2)` / `sum(M, dims=1)` but return plain vectors.
`NaN` values are **not** filtered (they propagate as in normal summation).

`T` is a promoted real type (typically `Float64`).

# Example
```julia
julia> A = [1 2 3; 4 5 6];

julia> row_sums(A)
2-element Vector{Float64}:
  6.0
 15.0

julia> col_sums(A)
3-element Vector{Float64}:
 5.0
 7.0
 9.0
````

"""
function row_sums(M::AbstractMatrix{<:Real})
    n, m = size(M)
    Tsum = promote_type(eltype(M), Float64)
    out = Vector{Tsum}(undef, n)
    @inbounds for i in 1:n
        s = zero(Tsum)
        @simd for j in 1:m
            s += M[i,j]
        end
        out[i] = s
    end
    out
end

function col_sums(M::AbstractMatrix{<:Real})
    n, m = size(M)
    Tsum = promote_type(eltype(M), Float64)
    out = Vector{Tsum}(undef, m)
    @inbounds for j in 1:m
        s = zero(Tsum)
        @simd for i in 1:n
            s += M[i,j]
        end
        out[j] = s
    end
    out
end