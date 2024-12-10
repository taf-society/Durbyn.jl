export air_passengers

function as_integer(x::AbstractVector{T}) where {T<:AbstractFloat}
    floor.(Int32, x)
end

function as_integer(x::AbstractVector{Int})
    x
end

function as_integer(x::T) where {T<:AbstractFloat}
    floor(Int32, x)
end

function as_integer(x::Int)
    x
end

function is_constant(data::AbstractArray)
    return all(x -> x == data[1], data)
end

function na_omit(x::Vector)
    filter(y -> !ismissing(y) && !isnan(y), skipmissing(x))
end

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
    return findfirst(x -> x == arg, choices) !== nothing ? arg : error("Invalid argument")
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

function na_contiguous(x::AbstractArray)
    good = .!ismissing.(x)
    if sum(good) == 0
        error("all times contain an NA")
    end
    tt = cumsum(Int[!g for g in good])
    ln = [sum(tt .== i) for i = 0:maximum(tt)]
    seg = findfirst(x -> x == maximum(ln), ln) - 1
    keep = tt .== seg
    st = findfirst(keep)

    if !good[st]
        st += 1
    end

    en = findlast(keep)
    omit = Int[]
    n = length(x)

    if st > 1
        append!(omit, 1:(st-1))
    end

    if en < n
        append!(omit, (en+1):n)
    end

    if length(omit) > 0
        x = x[st:en]
    end

    return x
end

function na_fail(x::Union{AbstractArray,DataFrame})
    if all(complete_cases(x))
        return x
    else
        throw(ArgumentError("missing values in x"))
    end
end

"""
    air_passengers() -> Vector{Float64}

Returns a vector of monthly airline passenger numbers (in thousands) from 1949 to 1960.

# Returns
- A vector of `Float64` values representing the number of passengers.

# Example
```julia
passengers = air_passengers()
println(passengers)
```
"""
function air_passengers()
    return [
    112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0,
    148.0, 136.0, 119.0, 104.0, 118.0, 115.0, 126.0,
    141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0,
    133.0, 114.0, 140.0, 145.0, 150.0, 178.0, 163.0,
    172.0, 178.0, 199.0, 199.0, 184.0, 162.0, 146.0,
    166.0, 171.0, 180.0, 193.0, 181.0, 183.0, 218.0,
    230.0, 242.0, 209.0, 191.0, 172.0, 194.0, 196.0,
    196.0, 236.0, 235.0, 229.0, 243.0, 264.0, 272.0,
    237.0, 211.0, 180.0, 201.0, 204.0, 188.0, 235.0,
    227.0, 234.0, 264.0, 302.0, 293.0, 259.0, 229.0,
    203.0, 229.0, 242.0, 233.0, 267.0, 269.0, 270.0,
    315.0, 364.0, 347.0, 312.0, 274.0, 237.0, 278.0,
    284.0, 277.0, 317.0, 313.0, 318.0, 374.0, 413.0,
    405.0, 355.0, 306.0, 271.0, 306.0, 315.0, 301.0,
    356.0, 348.0, 355.0, 422.0, 465.0, 467.0, 404.0,
    347.0, 305.0, 336.0, 340.0, 318.0, 362.0, 348.0,
    363.0, 435.0, 491.0, 505.0, 404.0, 359.0, 310.0,
    337.0, 360.0, 342.0, 406.0, 396.0, 420.0, 472.0,
    548.0, 559.0, 463.0, 407.0, 362.0, 405.0, 417.0,
    391.0, 419.0, 461.0, 472.0, 535.0, 622.0, 606.0,
    508.0, 461.0, 390.0, 432.0]
end 