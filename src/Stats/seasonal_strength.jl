@inline function _variance_skip_nan(values::AbstractVector{<:Real})::Float64
    count_valid = 0
    running_mean = 0.0
    centered_sum_squares = 0.0
    @inbounds for value in values
        value_float = Float64(value)
        if !isnan(value_float)
            count_valid += 1
            centered_delta = value_float - running_mean
            running_mean += centered_delta / count_valid
            centered_sum_squares += centered_delta * (value_float - running_mean)
        end
    end
    return count_valid > 1 ? centered_sum_squares / (count_valid - 1) : 0.0
end

@inline function _variance_sum_skip_nan(
    first_component::AbstractVector{<:Real},
    second_component::AbstractVector{<:Real},
)::Float64
    length(first_component) == length(second_component) ||
        throw(ArgumentError("components must have the same length"))

    count_valid = 0
    running_mean = 0.0
    centered_sum_squares = 0.0
    @inbounds for observation_index in eachindex(first_component, second_component)
        first_value = Float64(first_component[observation_index])
        second_value = Float64(second_component[observation_index])
        if !isnan(first_value) && !isnan(second_value)
            combined_value = first_value + second_value
            count_valid += 1
            centered_delta = combined_value - running_mean
            running_mean += centered_delta / count_valid
            centered_sum_squares += centered_delta * (combined_value - running_mean)
        end
    end
    return count_valid > 1 ? centered_sum_squares / (count_valid - 1) : 0.0
end

raw"""
    seasonal_strength(res::MSTLResult) -> Vector{Float64}

Compute the seasonal strength of each seasonal component in an MSTL decomposition.

Let `R` be the MSTL remainder and `Sᵢ` the `i`-th seasonal component. The
implemented statistic is:

```math
F_{\text{seasonal},i} = 1 - \frac{\operatorname{Var}(R)}{\operatorname{Var}(R + S_i)}
```

For numerical robustness, the returned value is clipped to `[0, 1]`.

# References
- Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering
  for time series data. Data Mining and Knowledge Discovery, 13(3), 335-364.
"""
function seasonal_strength(res::MSTLResult)
    if isempty(res.seasonals)
        return Float64[]
    end

    remainder_variance = _variance_skip_nan(res.remainder)
    seasonal_strength_values = Vector{Float64}(undef, length(res.seasonals))

    @inbounds for seasonal_index in eachindex(res.seasonals)
        seasonal_component = res.seasonals[seasonal_index]
        combined_variance = _variance_sum_skip_nan(res.remainder, seasonal_component)
        seasonal_strength_raw = combined_variance <= 0.0 ? 0.0 : (1.0 - remainder_variance / combined_variance)
        seasonal_strength_values[seasonal_index] = clamp(seasonal_strength_raw, 0.0, 1.0)
    end
    return seasonal_strength_values
end

raw"""
    seasonal_strength(res::MSTLResult) -> Vector{Float64}
    seasonal_strength(x::AbstractVector, m; kwargs...) -> Vector{Float64}

Compute the **seasonal strength** of each seasonal component in an MSTL decomposition.

For each seasonal component `sᵢ`, the seasonal strength is defined as

```math
F_{\text{seasonal},i} = 1 - \frac{\operatorname{Var}(R)}{\operatorname{Var}(R + S_i)}
```

where `R` is the remainder from the MSTL decomposition.
A value near 1 indicates a strong seasonal signal, while values near 0 indicate little
seasonal contribution relative to the remainder. Returned values are clipped to `[0, 1]`.

# Arguments
- `res::MSTLResult`: an existing decomposition returned by [`mstl`](@ref).
- `x::AbstractVector`: a univariate time series. In this form, `mstl(x, m; kwargs...)`
  is called internally before computing seasonal strength.
- `m`: seasonal periods to use if decomposing `x` directly (forwarded to `mstl`).
- `kwargs...`: other keyword arguments passed through to `mstl`.

# Returns
A vector of seasonal strength values, one per seasonal period in `res.m`.
If no seasonal components exist, an empty `Vector{Float64}` is returned.

# Notes
- This implements the seasonal strength heuristic from Wang, Smith & Hyndman (2006).
- Missing values (`NaN`s) are ignored when computing variances.
- The order of returned strengths matches the order of `res.m`.

# References
- Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering
  for time series data. Data Mining and Knowledge Discovery, 13(3), 335-364.

# Examples
```julia
y = rand(200) .+ 2sin.(2π*(1:200)/7) .+ 0.5sin.(2π*(1:200)/30)
res = mstl(y, [7,30])
strengths = seasonal_strength(res)
# e.g., [0.85, 0.42]

# Convenience form: run mstl inside
seasonal_strength(y, [7,30]; iterate=2)
```
"""
function seasonal_strength(
    x::AbstractVector{T},
    m::Union{Integer,AbstractVector{<:Integer}};
    kwargs...,
) where {T<:Real}
    res = mstl(x, m; kwargs...)
    return seasonal_strength(res)
end
