"""
    scaler(x::AbstractVector, scale::AbstractVector)

Scale the parameter vector `x` elementwise by the vector `scale`, returning `x ./ scale`.

This is useful for rescaling optimization parameters.

# Arguments
- `x::AbstractVector`: Parameter vector to scale.
- `scale::AbstractVector`: Vector of scaling factors (must match the length of `x`).

# Returns
- Scaled parameter vector.
"""
function scaler(x::AbstractVector, scale::AbstractVector)
    @assert length(x) == length(scale)
    return x ./ scale
end

"""
    descaler(x_scaled::AbstractVector, scale::AbstractVector)

Recover the original parameter vector by multiplying the scaled vector `x_scaled` elementwise by `scale`, returning `x_scaled .* scale`.

This undoes the scaling performed by `scaler`.

# Arguments
- `x_scaled::AbstractVector`: Scaled parameter vector.
- `scale::AbstractVector`: Vector of scaling factors (must match the length of `x_scaled`).

# Returns
- Original, unscaled parameter vector.
"""
function descaler(x_scaled::AbstractVector, scale::AbstractVector)
    @assert length(x_scaled) == length(scale)
    return x_scaled .* scale
end