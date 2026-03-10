function _build_fourier_matrix(x, K, times, period)
    length(period) == length(K) || throw(ArgumentError("Number of periods does not match number of orders"))

    !any(2 .* K .> period) || throw(ArgumentError("K must not be greater than period/2"))

    frequencies = Float64[]
    labels = String[]

    for j in eachindex(period)
        if K[j] > 0
            append!(frequencies, (1:K[j]) ./ period[j])
            # Interleaved labels: S1, C1, S2, C2, ...
            # Period suffix added only for multi-seasonal disambiguation
            suffix = length(period) > 1 ? string("-", round(Int, period[j])) : ""
            for i in 1:K[j]
                push!(labels, string("S", i, suffix))
                push!(labels, string("C", i, suffix))
            end
        end
    end

    # Remove duplicate frequencies (multi-seasonal)
    is_duplicate = duplicated(frequencies)
    frequencies = frequencies[.!is_duplicate]
    labels = labels[.!repeat(is_duplicate, inner=2)]

    # Identify frequencies where sinpi=0 (K = period/2)
    has_sin_term = abs.(2 .* frequencies .- round.(2 .* frequencies)) .> eps(Float64)

    matrix = fill(NaN, length(times), 2 * length(frequencies))

    for j in eachindex(frequencies)
        if has_sin_term[j]
            matrix[:, 2 * j - 1] .= sinpi.(2 .* frequencies[j] .* times)
        end
        matrix[:, 2 * j] .= cospi.(2 .* frequencies[j] .* times)
    end

    # Remove NaN columns (skipped sin terms) and corresponding labels
    valid_columns = vec(.!isnan.(sum(matrix, dims=1)))
    matrix = matrix[:, valid_columns]
    labels = labels[valid_columns]

    return matrix, labels
end
"""
    fourier(x::Vector{T}; m::Int, K::Int, h::Union{Int, Nothing}=nothing) -> NamedTuple

Fourier terms for modelling seasonality. Returns a `NamedTuple` of vectors
keyed by `:S1`, `:C1`, `:S2`, `:C2`, etc.

When `K == m/2`, the degenerate sin term (`sin(πt) ≡ 0` for integer `t`) is
automatically dropped (the term is identically zero for integer time indices).

# Arguments
- `x::Vector{T}`: Time series data.
- `m::Int`: Seasonal period (frequency).
- `K::Int`: Number of Fourier pairs to generate (`K ≤ m/2`).
- `h::Int`: Forecast horizon. When provided, returns future Fourier terms at
  times `n+1, …, n+h` instead of the in-sample terms.

# Returns
A `NamedTuple` with `≤ 2K` entries (strictly less when a sin column is dropped).

# Example
```julia
y = air_passengers()
f_train = fourier(y; m=12, K=6)        # NamedTuple with 11 entries
f_future = fourier(y; m=12, K=6, h=24) # NamedTuple with 11 entries

data = merge((value = y,), f_train)
newdata = f_future
```

# References
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed), OTexts.
"""
function fourier(x::Vector{T}; m::Int, K::Int, h::Union{Int, Nothing}=nothing) where T
    if isnothing(h)
        X, colnames = _build_fourier_matrix(x, [K], 1:length(x), [m])
        size(X, 1) == length(x) || throw(ArgumentError("Row dimension is wrong"))
    else
        X, colnames = _build_fourier_matrix(x, [K], length(x) .+ (1:h), [m])
        size(X, 1) == h || throw(ArgumentError("Row dimension is wrong"))
    end

    col_names = Tuple(Symbol.(colnames))
    col_vectors = Tuple(X[:, j] for j in 1:size(X, 2))
    return NamedTuple{col_names}(col_vectors)
end
