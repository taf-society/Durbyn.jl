sinpi(x) = sin(pi * x)
cospi(x) = cos(pi * x)

function base_fourier(x, K, times, period)
    @assert length(period) == length(K) "Number of periods does not match number of orders"
        
    @assert !any(2 .* K .> period) "K must not be greater than period/2"
    
    p = Float64[]
    labels = String[]
    
    for j in eachindex(period)
        if K[j] > 0
            append!(p, (1:K[j]) ./ period[j])
            append!(labels, [string("S", i, "-", round(period[j])) for i in 1:K[j]])
            append!(labels, [string("C", i, "-", round(period[j])) for i in 1:K[j]])
        end
    end
    
    k = duplicated(p)
    p = p[.!k]
    labels = labels[.!repeat(k, inner=2)]

    # Ensure 'k' is correctly formed to avoid indexing errors
    k = abs.(2 .* p .- round.(2 .* p)) .> eps(Float64)
    
    # Initialize matrix X
    X = zeros(Float64, length(times), 2 * length(p))
    
    for j in eachindex(p)
        if k[j]
            X[:, 2 * j - 1] .= sinpi.(2 .* p[j] .* times)
        end
        X[:, 2 * j] .= cospi.(2 .* p[j] .* times)
    end

    colnames = labels
    valid_columns = .!isnan.(sum(X, dims=1))
    
    # Convert valid_columns to a vector for indexing
    X = X[:, vec(valid_columns)]  
    
    return X
end
"""
    fourier(; x::Vector{T}, m::Int, K::Int, h::Union{Int, Nothing}=nothing) -> Any where T

Fourier terms for modelling seasonality

# Arguments
- `x::Vector{T}`: A vector representing the time series data.
- `m::Int`: The frequency of the time series data.
- `K::Int`: The number of Fourier terms to generate.
- `h::Int`: The forecast horizon.

# Returns
- Numerical matrix.

# Example
```julia
y = air_passengers()
resulth = fourier(y, m=12, K=6, h=12)
println(resulth)
result = fourier(y, m=12, K=6)
println(result)
```
"""
function fourier(x::Vector{T}; m::Int, K::Int, h::Union{Int, Nothing}=nothing) where T
    if isnothing(h)
        out = base_fourier(x, [K], 1:length(x), [m])
        @assert size(out) == (size(x, 1), 2 * K) "Dimentions are wrong!"
    else
        out = base_fourier(x, [K], length(x) .+ (1:h), [m])
        @assert size(out) == (h, 2 * K) "Dimentions are wrong!"
    end
    
    return out
end