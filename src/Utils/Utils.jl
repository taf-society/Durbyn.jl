module Utils
using LinearAlgebra
using Base: @static

import Statistics: mean
export NamedMatrix, get_elements, get_vector, align_columns, add_drift_term, cbind, setrow!
export Formula, parse_formula, compile, model_matrix, model_frame, as_vector
export air_passengers, ausbeer, lynx, sunspots, pedestrian_counts, simulate_seasonal_data
export complete_cases
include("named_matrix.jl")
include("model_frame.jl")
include("math.jl")
include("datasets.jl")


"""
    as_vector(x::Matrix)

Convert a matrix slice to a `Vector` if it has only 1 row or 1 column.
Throws an error otherwise.
"""
function as_vector(x::AbstractMatrix)
    r, c = size(x)
    if r == 1 || c == 1
        return vec(x)   # turn 1×N or M×1 into a Vector
    else
        error("Matrix must have exactly 1 row or 1 column; got $(r)×$(c).")
    end
end


"""
    struct ModelFitError <: Exception

A custom exception to indicate that no model was able to be fitted.
Use this error to notify users when the model fitting process fails
due to the inability to find a suitable model.

# Fields
- `msg::String`: A message providing details about why the model fitting failed.

# Example

```julia
if best_ic == Inf
    throw(ModelFitError("No model was able to be fitted with the provided data and parameters."))
end
```
"""
struct ModelFitError <: Exception
    msg::String
end

"""
    Base.showerror(io::IO, e::ModelFitError)

Prints the custom error message for `ModelFitError`.

# Example

```julia
try
    throw(ModelFitError("Test error"))
catch e
    showerror(stdout, e)
end
```
"""
function Base.showerror(io::IO, e::ModelFitError)
    print(io, "ModelFitError: ", e.msg, "\n")
end

function as_integer(x::AbstractVector{T}) where {T<:AbstractFloat}
    floor.(Int64, x)
end

function as_integer(x::AbstractVector{Int})
    x
end

function as_integer(x::T) where {T<:AbstractFloat}
    floor(Int64, x)
end

function as_integer(x::Int)
    x
end


"""
    is_constant(data::AbstractVector) -> Bool

Return `true` if all **non-missing** elements of `data` are equal, treating multiple
`NaN` values as equal. Returns `true` for an empty vector or an all-`missing` vector.

### Rules
- `missing` values are ignored when checking constancy.
- `NaN` values are considered equal to each other (via `isequal`).
- If any two **non-missing** values differ (by `isequal`), return `false`.

# Examples
```jldoctest
julia> is_constant([missing, missing])
true

julia> is_constant([5.0, 5.0, missing, 5.0])
true

julia> is_constant([NaN, NaN, missing])
true

julia> is_constant([NaN, 1.0, NaN])
false

julia> is_constant([5.0, 6.0, 5.0])
false

julia> is_constant(Float64[])
true
```
"""
function is_constant(data::AbstractVector)
    isempty(data) && return true
    it = skipmissing(data)
    first_pair = iterate(it)
    isnothing(first_pair) && return true  # all values were missing
    v, st = first_pair
    while true
        nxt = iterate(it, st)
        isnothing(nxt) && return true
        x, st = nxt
        if !isequal(x, v)
            return false
        end
    end
end

"""
    is_constant(X::AbstractMatrix) -> Vector{Bool}

Apply [`is_constant(::AbstractVector)`] to each **column** of `X` and return a
`Vector{Bool}` of the same length as `size(X, 2)`.

This answers: “Is each column constant (ignoring missings, treating NaNs as equal)?”

If you want a single answer for *all* columns, use:
`all(is_constant, eachcol(X))`.

# Examples
```jldoctest
julia> X = [1.0  NaN   2.0;
            1.0  NaN   2.0;
            1.0  NaN   3.0];

julia> is_constant(X)
3-element BitVector:
  1  # first column all 1.0
  1  # second column all NaN (treated equal)
  0  # third column has 2.0 and 3.0

julia> all(is_constant, eachcol(X))
false
```
"""
function is_constant(X::AbstractMatrix)
    map(is_constant, eachcol(X))
end

"""
    is_constant_all(data::AbstractVector) -> Bool
    is_constant_all(X::AbstractMatrix) -> Bool

Return `true` if the container is entirely constant.

- For a vector: returns the same as [`is_constant(::AbstractVector)`],
  i.e. `true` if all non-missing elements are equal (treating multiple `NaN`s as equal).
- For a matrix: returns `true` only if **every column** is constant.

# Examples
```jldoctest
julia> is_constant_all([missing, 5.0, 5.0, missing])
true

julia> X = [1.0  2.0;
            1.0  2.0;
            1.0  2.0];

julia> is_constant_all(X)   # both columns constant
true

julia> Y = [1.0  2.0;
            1.0  3.0;
            1.0  2.0];

julia> is_constant_all(Y)   # second column not constant
false
"""
is_constant_all(X::AbstractMatrix) = all(is_constant(X))
is_constant_all(data::AbstractVector) = is_constant(data)


"""
    na_omit_pair(x::AbstractVector, X::AbstractMatrix)

Remove all rows from the input vector `x` and matrix `X` where either `x` or any column of the corresponding row in `X` contains missing data or `NaN`.

# Arguments
- `x::AbstractVector`: A vector of numeric values, which may contain `missing` or `NaN`.
- `X::AbstractMatrix`: A matrix of numeric values, with the same number of rows as `x`. May contain `missing` or `NaN`.

# Returns
- `x_clean::Vector`: A vector containing the elements of `x` where both `x` and the corresponding row in `X` contain no `missing` or `NaN` values.
- `X_clean::Matrix`: A matrix containing only the rows of `X` where both `x` and the corresponding row in `X` contain no `missing` or `NaN` values.

# Details
A row is removed if **either** the value in `x` or **any** value in the corresponding row of `X` is `missing` or `NaN`. Only the rows that are fully complete (i.e., not `missing` and not `NaN` in both `x` and `X`) are kept.

# Example

```julia
x = [1.0, NaN, 3.0, missing, 5.0]
X = [1.0 2.0;
     3.0 NaN;
     5.0 6.0;
     7.0 8.0;
     missing 10.0]

x_clean, X_clean = na_omit_pair(x, X)

println(x_clean)  # Output: [1.0, 3.0]
println(X_clean)  # Output: [1.0 2.0; 5.0 6.0]
```
"""
function na_omit_pair(x::AbstractVector, X::AbstractMatrix)
    @assert length(x) == size(X, 1) "x and X must have the same number of rows"
    function row_ok(i)
        xi = x[i]
        Xi = X[i, :]
        not_missing = !ismissing(xi) && !isnan(xi) &&
                      all(!ismissing(Xi[j]) && !isnan(Xi[j]) for j in 1:length(Xi))
        return not_missing
    end
    idxs = [i for i in 1:length(x) if row_ok(i)]
    return x[idxs], X[idxs, :]
end

function na_omit(x::AbstractVector)
    [v for v in x if !ismissing(v) && !(v isa AbstractFloat && isnan(v))]
end

isna(v) = ismissing(v) || (v isa AbstractFloat && isnan(v))


function duplicated(arr::Vector{T})::Vector{Bool} where {T}
    seen = Dict{T,Bool}()
    result = falses(length(arr))
    for i in 1:length(arr)
        if haskey(seen, arr[i])
            result[i] = true
        else
            seen[arr[i]] = true
        end
    end
    return result
end

function match_arg(arg, choices)
    return !isnothing(findfirst(x -> x == arg, choices)) ? arg : error("Invalid argument")
end

function complete_cases(x::AbstractArray)
    return .!ismissing.(x)
end

function mean2(x::AbstractVector{<:Union{Missing,Number}}; omit_na::Bool=false)
    if omit_na
        x = na_omit(x)
    end
    return mean(x)
end

function check_component(container, key)
    if isa(container, Dict) && haskey(container, key)
        return container[key]
    elseif isa(container, AbstractVector) && key in container
        return key
    else
        return nothing
    end
end

function evaluation_metrics(actual, pred)
    error = actual .- pred
    mse = mean(error.^2)
    mae = mean(abs.(error))
    n = length(actual)
    temp = cumsum(actual) ./ (1:n)
    n = ceil(Int, 0.3 * n)
    temp[1:n] .= temp[n]
    error2 = pred .- temp
    mar = sum(abs.(error2))
    msr = sum(error2.^2)
    return Dict("mse" => mse, "mae" => mae, "mar" => mar, "msr" => msr)
end

end