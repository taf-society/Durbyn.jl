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
"""
mutable struct NumericalGradientCache
    x_trial::Vector{Float64}
    gradient::Vector{Float64}

    function NumericalGradientCache(n::Int)
        new(Vector{Float64}(undef, n), Vector{Float64}(undef, n))
    end
end

"""
    numgrad!(grad, x_trial, f, n, x, extra, step_sizes) -> Vector{Float64}

Compute the numerical gradient of a scalar-valued function `f` using
**central finite differences**, with pre-allocated buffers for maximum efficiency.

# Arguments
- `grad::AbstractVector{Float64}` — Pre-allocated output buffer for the gradient (length `n`)
- `x_trial::AbstractVector{Float64}` — Pre-allocated trial vector buffer (length `n`)
- `f::Function` — Objective function, called as `f(n, x, extra)`
- `n::Int` — Problem dimension
- `x::AbstractVector{Float64}` — Point at which to evaluate the gradient
- `extra` — Extra argument passed to `f` (may be `nothing`)
- `step_sizes::AbstractVector{Float64}` — Step sizes for numerical differentiation

# Returns
- `Vector{Float64}` — Numerical gradient approximation (same object as `grad`)

"""
@inline function numgrad!(
    grad::AbstractVector{Float64},
    x_trial::AbstractVector{Float64},
    f::Function,
    n::Int,
    x::AbstractVector{Float64},
    extra,
    step_sizes::AbstractVector{Float64},
)
    @inbounds for i in 1:n
        x_trial[i] = x[i]
    end

    @inbounds for i in 1:n
        eps = step_sizes[i]

        x_trial[i] = x[i] + eps
        val1 = f(n, x_trial, extra)

        x_trial[i] = x[i] - eps
        val2 = f(n, x_trial, extra)

        grad[i] = (val1 - val2) * inv(2.0 * eps)

        if !isfinite(grad[i])
            error("non-finite finite-difference value [$i]")
        end

        x_trial[i] = x[i]
    end

    return grad
end
"""
    numgrad_bounded!(grad, x_trial, f, n, x, extra, step_sizes, lower, upper) -> Vector{Float64}

Compute the numerical gradient with **bound constraints**, using pre-allocated
buffers and central finite differences. Ensures perturbations stay within
specified lower and upper bounds.

# Arguments
- `grad::AbstractVector{Float64}` — Pre-allocated gradient output buffer (length `n`)
- `x_trial::AbstractVector{Float64}` — Pre-allocated trial vector buffer (length `n`)
- `f::Function` — Objective function
- `n::Int` — Problem dimension
- `x::AbstractVector{Float64}` — Current evaluation point
- `extra` — Extra argument (may be `nothing`)
- `step_sizes::AbstractVector{Float64}` — Step sizes for numerical differentiation
- `lower::Union{Nothing,AbstractVector{Float64}}` — Lower bounds (or `nothing`)
- `upper::Union{Nothing,AbstractVector{Float64}}` — Upper bounds (or `nothing`)

# Returns
- `Vector{Float64}` — Numerical gradient (same object as `grad`)

# Notes
Adjusts step sizes near boundaries to remain within feasible limits.
"""
@inline function numgrad_bounded!(
    grad::AbstractVector{Float64},
    x_trial::AbstractVector{Float64},
    f::Function,
    n::Int,
    x::AbstractVector{Float64},
    extra,
    step_sizes::AbstractVector{Float64},
    lower::Union{Nothing,AbstractVector{Float64}},
    upper::Union{Nothing,AbstractVector{Float64}},
)
    @inbounds for i in 1:n
        x_trial[i] = x[i]
    end

    @inbounds for i in 1:n
        epsused = eps = step_sizes[i]

        tmp = x[i] + eps
        if !isnothing(upper) && tmp > upper[i]
            tmp = upper[i]
            epsused = tmp - x[i]
        end
        x_trial[i] = tmp
        val1 = f(n, x_trial, extra)

        tmp = x[i] - eps
        if !isnothing(lower) && tmp < lower[i]
            tmp = lower[i]
            eps = x[i] - tmp
        end
        x_trial[i] = tmp
        val2 = f(n, x_trial, extra)

        grad[i] = (val1 - val2) * inv(epsused + eps)

        if !isfinite(grad[i])
            error("non-finite finite-difference value [$i]")
        end

        x_trial[i] = x[i]
    end

    return grad
end

"""
    numgrad(f, n, x, extra, step_sizes; usebounds=false, lower=nothing, upper=nothing) -> Vector{Float64}

Convenient high-level wrapper for computing numerical gradients.
Creates a temporary cache internally (single-use).

For repeated gradient evaluations, create a `NumericalGradientCache` and use
[`numgrad_with_cache!`](@ref) for best performance.

# Example
```julia
f(n, x, ex) = sum(x.^2)
x = [1.0, 2.0, 3.0]
step_sizes = fill(1e-3, 3)

# One-off computation
g = numgrad(f, 3, x, nothing, step_sizes)

# Cached computation for repeated optimization steps
cache = NumericalGradientCache(3)
for iter in 1:1000
    g = numgrad_with_cache!(cache, f, 3, x, nothing, step_sizes)
end
```
"""
function numgrad(
    f::Function,
    n::Int,
    x::Vector{Float64},
    extra,
    step_sizes::Vector{Float64};
    usebounds::Bool = false,
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
)
    cache = NumericalGradientCache(n)

    if !usebounds
        return numgrad!(cache.gradient, cache.x_trial, f, n, x, extra, step_sizes)
    else
        return numgrad_bounded!(cache.gradient, cache.x_trial, f, n, x, extra, step_sizes, lower, upper)
    end
end

"""
    numgrad_with_cache!(cache, f, n, x, extra, step_sizes; usebounds=false, lower=nothing, upper=nothing) -> Vector{Float64}

Compute the numerical gradient using a **pre-allocated cache**, providing the
fastest possible implementation for repeated evaluations inside optimizers
such as BFGS, L-BFGS.

# Arguments
- `cache::NumericalGradientCache` — Pre-allocated workspace
- `f::Function` — Objective function
- `n::Int` — Dimension of the problem
- `x::AbstractVector{Float64}` — Current point
- `extra` — Extra argument (may be `nothing`)
- `step_sizes::AbstractVector{Float64}` — Step sizes for finite differences
- `usebounds::Bool=false` — Whether to enforce bound constraints
- `lower`, `upper` — Optional bound vectors

# Returns
- `Vector{Float64}` — Numerical gradient (references `cache.gradient`)

"""
@inline function numgrad_with_cache!(
    cache::NumericalGradientCache,
    f::Function,
    n::Int,
    x::AbstractVector{Float64},
    extra,
    step_sizes::AbstractVector{Float64};
    usebounds::Bool = false,
    lower::Union{Nothing,AbstractVector{Float64}} = nothing,
    upper::Union{Nothing,AbstractVector{Float64}} = nothing,
)
    if !usebounds
        return numgrad!(cache.gradient, cache.x_trial, f, n, x, extra, step_sizes)
    else
        return numgrad_bounded!(cache.gradient, cache.x_trial, f, n, x, extra, step_sizes, lower, upper)
    end
end

"""
    numgrad(f, n, x, extra; step_sizes=fill(1e-3, n), kwargs...) -> Vector{Float64}

Convenience wrapper with default step sizes.
Equivalent to calling [`numgrad`](@ref) with `step_sizes = fill(1e-3, n)`.

# Example
```julia
f(n, x, ex) = sum(x.^2)
x = [1.0, 2.0, 3.0]
g = numgrad(f, 3, x, nothing)
```
"""
function numgrad(
    f::Function,
    n::Int,
    x::Vector{Float64},
    extra;
    step_sizes::Vector{Float64} = fill(1e-3, n),
    usebounds::Bool = false,
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
)
    return numgrad(f, n, x, extra, step_sizes;
        usebounds = usebounds, lower = lower, upper = upper)
end
