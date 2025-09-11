function seasonal_strength(res::MSTLResult)
    # variance ignoring NaNs (if any)
    var_nan(v) = begin
        mask = .!isnan.(v)
        any(mask) ? var(view(v, mask)) : 0.0
    end

    if isempty(res.seasonals)
        return Float64[]
    end

    vare = var_nan(res.remainder)
    strengths = similar(res.m, Float64)

    for (i, s) in enumerate(res.seasonals)
        denom = var_nan(res.remainder .+ s)
        strength = (denom == 0) ? 0.0 : (1 - vare / denom)
        strengths[i] = clamp(strength, 0.0, 1.0)
    end
    return strengths
end

"""
    seasonal_strength(res::MSTLResult) -> Vector{Float64}
    seasonal_strength(x::AbstractVector; m=..., kwargs...) -> Vector{Float64}

Compute the **seasonal strength** of each seasonal component in an MSTL decomposition.

For each seasonal component `sᵢ`, the seasonal strength is defined as

```

strengthᵢ = clamp(1 - var(R) / var(R + sᵢ), 0, 1)

````

where `R` is the remainder from the MSTL decomposition.  
A value near 1 indicates a strong seasonal signal, while values near 0 indicate little
seasonal contribution relative to the remainder.

# Arguments
- `res::MSTLResult`: an existing decomposition returned by [`mstl`](@ref).
- `x::AbstractVector`: a univariate time series. In this form, `mstl(x; periods, kwargs...)`
  is called internally before computing seasonal strength.
- `m`: seasonal periods to use if decomposing `x` directly (forwarded to `mstl`).
- `kwargs...`: other keyword arguments passed through to `mstl`.

# Returns
A vector of seasonal strength values, one per seasonal period in `res.m`.  
If no seasonal components exist, an empty `Vector{Float64}` is returned.

# Notes
- This is equivalent to Hyndman's “seasonal strength” heuristic used in R's `seas.heuristic`.
- Missing values (`NaN`s) are ignored when computing variances.
- The order of returned strengths matches the order of `res.m`.

# Examples
```julia
y = rand(200) .+ 2sin.(2π*(1:200)/7) .+ 0.5sin.(2π*(1:200)/30)
res = mstl(y; m=[7,30])
strengths = seasonal_strength(res)
# e.g., [0.85, 0.42]

# Convenience form: run mstl inside
seasonal_strength(y; m=[7,30], iterate=2)
"""
function seasonal_strength(
    x::AbstractVector{T},
    m::Union{Integer,AbstractVector{<:Integer}};
    kwargs...,
) where {T<:Real}
    res = mstl(x, m, kwargs...)
    return seasonal_strength(res)
end