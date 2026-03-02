"""
    NumericalGradientCache

Pre-allocated workspace for efficient numerical gradient computation.
By reusing buffers between iterations, it avoids dynamic memory allocations and
reduces runtime overhead in repeated evaluations.

# Fields
- `x_trial::Vector{Float64}` — Working buffer for perturbed parameter vectors (length `n`)
- `gradient::Vector{Float64}` — Output buffer for computed gradient (length `n`)

# Example
```julia
cache = NumericalGradientCache(5)
```

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Section 8.1.
  Springer.
"""
mutable struct NumericalGradientCache
    x_trial::Vector{Float64}
    gradient::Vector{Float64}

    function NumericalGradientCache(n::Int)
        new(Vector{Float64}(undef, n), Vector{Float64}(undef, n))
    end
end

"""
    numgrad!(grad, x_trial, f, x, step_sizes) -> Vector{Float64}

Compute the numerical gradient of a scalar-valued function `f` using
**central finite differences**, with pre-allocated buffers for maximum efficiency.

The central difference formula for component `i` is:

    ∂f/∂xᵢ ≈ (f(x + hᵢeᵢ) - f(x - hᵢeᵢ)) / (2hᵢ)

where `hᵢ` is the step size for dimension `i`.

# Arguments
- `grad::AbstractVector{Float64}` — Pre-allocated output buffer for the gradient
- `x_trial::AbstractVector{Float64}` — Pre-allocated trial vector buffer
- `f::Function` — Objective function, called as `f(x)` → scalar
- `x::AbstractVector{Float64}` — Point at which to evaluate the gradient
- `step_sizes::AbstractVector{Float64}` — Step sizes for numerical differentiation

# Returns
- `Vector{Float64}` — Numerical gradient approximation (same object as `grad`)

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Section 8.1.
  Springer.
"""
@inline function numgrad!(
    grad::AbstractVector{Float64},
    x_trial::AbstractVector{Float64},
    f::Function,
    x::AbstractVector{Float64},
    step_sizes::AbstractVector{Float64},
)
    n = length(x)
    @inbounds for i in 1:n
        x_trial[i] = x[i]
    end

    @inbounds for i in 1:n
        step_i = step_sizes[i]

        x_trial[i] = x[i] + step_i
        f_plus = f(x_trial)

        x_trial[i] = x[i] - step_i
        f_minus = f(x_trial)

        grad[i] = (f_plus - f_minus) * inv(2.0 * step_i)

        if !isfinite(grad[i])
            error("non-finite finite-difference value [$i]")
        end

        x_trial[i] = x[i]
    end

    return grad
end
"""
    numgrad_bounded!(grad, x_trial, f, x, step_sizes, lower, upper) -> Vector{Float64}

Compute the numerical gradient with **bound constraints**, using pre-allocated
buffers and central finite differences. Ensures perturbations stay within
specified lower and upper bounds.

When a perturbation would exceed a bound, the step size is reduced to keep the
trial point feasible, using the asymmetric difference:

    ∂f/∂xᵢ ≈ (f(x + h⁺ᵢeᵢ) - f(x - h⁻ᵢeᵢ)) / (h⁺ᵢ + h⁻ᵢ)

# Arguments
- `grad::AbstractVector{Float64}` — Pre-allocated gradient output buffer
- `x_trial::AbstractVector{Float64}` — Pre-allocated trial vector buffer
- `f::Function` — Objective function, called as `f(x)` → scalar
- `x::AbstractVector{Float64}` — Current evaluation point
- `step_sizes::AbstractVector{Float64}` — Step sizes for numerical differentiation
- `lower::Union{Nothing,AbstractVector{Float64}}` — Lower bounds (or `nothing`)
- `upper::Union{Nothing,AbstractVector{Float64}}` — Upper bounds (or `nothing`)

# Returns
- `Vector{Float64}` — Numerical gradient (same object as `grad`)

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Section 8.1.
  Springer.
"""
@inline function numgrad_bounded!(
    grad::AbstractVector{Float64},
    x_trial::AbstractVector{Float64},
    f::Function,
    x::AbstractVector{Float64},
    step_sizes::AbstractVector{Float64},
    lower::Union{Nothing,AbstractVector{Float64}},
    upper::Union{Nothing,AbstractVector{Float64}},
)
    n = length(x)
    @inbounds for i in 1:n
        x_trial[i] = x[i]
    end

    @inbounds for i in 1:n
        step_fwd = step_sizes[i]
        step_bwd = step_sizes[i]

        tmp = x[i] + step_fwd
        if !isnothing(upper) && tmp > upper[i]
            tmp = upper[i]
            step_fwd = tmp - x[i]
        end
        x_trial[i] = tmp
        f_plus = f(x_trial)

        tmp = x[i] - step_bwd
        if !isnothing(lower) && tmp < lower[i]
            tmp = lower[i]
            step_bwd = x[i] - tmp
        end
        x_trial[i] = tmp
        f_minus = f(x_trial)

        grad[i] = (f_plus - f_minus) * inv(step_fwd + step_bwd)

        if !isfinite(grad[i])
            error("non-finite finite-difference value [$i]")
        end

        x_trial[i] = x[i]
    end

    return grad
end

"""
    numgrad(f, x, step_sizes; usebounds=false, lower=nothing, upper=nothing) -> Vector{Float64}

Convenient high-level wrapper for computing numerical gradients.
Creates a temporary cache internally (single-use).

For repeated gradient evaluations, create a `NumericalGradientCache` and use
[`numgrad_with_cache!`](@ref) for best performance.

# Example
```julia
f(x) = sum(x.^2)
x = [1.0, 2.0, 3.0]
step_sizes = fill(1e-3, 3)

# One-off computation
g = numgrad(f, x, step_sizes)

# Cached computation for repeated optimization steps
cache = NumericalGradientCache(3)
for iter in 1:1000
    g = numgrad_with_cache!(cache, f, x, step_sizes)
end
```
"""
function numgrad(
    f::Function,
    x::Vector{Float64},
    step_sizes::Vector{Float64};
    usebounds::Bool = false,
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
)
    n = length(x)
    cache = NumericalGradientCache(n)

    if !usebounds
        return numgrad!(cache.gradient, cache.x_trial, f, x, step_sizes)
    else
        return numgrad_bounded!(cache.gradient, cache.x_trial, f, x, step_sizes, lower, upper)
    end
end

"""
    numgrad_with_cache!(cache, f, x, step_sizes; usebounds=false, lower=nothing, upper=nothing) -> Vector{Float64}

Compute the numerical gradient using a **pre-allocated cache**, providing the
fastest possible implementation for repeated evaluations inside optimizers
such as BFGS, L-BFGS.

# Arguments
- `cache::NumericalGradientCache` — Pre-allocated workspace
- `f::Function` — Objective function, called as `f(x)` → scalar
- `x::AbstractVector{Float64}` — Current point
- `step_sizes::AbstractVector{Float64}` — Step sizes for finite differences
- `usebounds::Bool=false` — Whether to enforce bound constraints
- `lower`, `upper` — Optional bound vectors

# Returns
- `Vector{Float64}` — Numerical gradient (references `cache.gradient`)

"""
@inline function numgrad_with_cache!(
    cache::NumericalGradientCache,
    f::Function,
    x::AbstractVector{Float64},
    step_sizes::AbstractVector{Float64};
    usebounds::Bool = false,
    lower::Union{Nothing,AbstractVector{Float64}} = nothing,
    upper::Union{Nothing,AbstractVector{Float64}} = nothing,
)
    if !usebounds
        return numgrad!(cache.gradient, cache.x_trial, f, x, step_sizes)
    else
        return numgrad_bounded!(cache.gradient, cache.x_trial, f, x, step_sizes, lower, upper)
    end
end

"""
    numgrad(f, x; step_sizes=fill(1e-3, length(x)), kwargs...) -> Vector{Float64}

Convenience wrapper with default step sizes.
Equivalent to calling [`numgrad`](@ref) with `step_sizes = fill(1e-3, length(x))`.

# Example
```julia
f(x) = sum(x.^2)
x = [1.0, 2.0, 3.0]
g = numgrad(f, x)
```
"""
function numgrad(
    f::Function,
    x::Vector{Float64};
    step_sizes::Vector{Float64} = fill(1e-3, length(x)),
    usebounds::Bool = false,
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
)
    return numgrad(f, x, step_sizes;
        usebounds = usebounds, lower = lower, upper = upper)
end
