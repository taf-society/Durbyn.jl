"""
    NumericalGradientCache

Pre-allocated workspace for efficient numerical gradient computation.
By reusing buffers between iterations, it avoids dynamic memory allocations and
reduces runtime overhead in repeated evaluations.

# Fields
- `xtrial::Vector{Float64}` — Working buffer for perturbed parameter vectors (length `n`)
- `df::Vector{Float64}` — Output buffer for computed gradient (length `n`)

# Example
```julia
cache = NumericalGradientCache(5)
```
"""
mutable struct NumericalGradientCache
    xtrial::Vector{Float64}
    df::Vector{Float64}

    function NumericalGradientCache(n::Int)
        new(Vector{Float64}(undef, n), Vector{Float64}(undef, n))
    end
end

"""
    numgrad!(df, xtrial, f, n, x, ex, ndeps) -> Vector{Float64}

Compute the numerical gradient of a scalar-valued function `f` using
**central finite differences**, with pre-allocated buffers for maximum efficiency.

# Arguments
- `df::AbstractVector{Float64}` — Pre-allocated output buffer for the gradient (length `n`)
- `xtrial::AbstractVector{Float64}` — Pre-allocated trial vector buffer (length `n`)
- `f::Function` — Objective function, called as `f(n, x, ex)`
- `n::Int` — Problem dimension
- `x::AbstractVector{Float64}` — Point at which to evaluate the gradient
- `ex` — Extra argument passed to `f` (may be `nothing`)
- `ndeps::AbstractVector{Float64}` — Step sizes for numerical differentiation

# Returns
- `Vector{Float64}` — Numerical gradient approximation (same object as `df`)

"""
@inline function numgrad!(
    df::AbstractVector{Float64},
    xtrial::AbstractVector{Float64},
    f::Function,
    n::Int,
    x::AbstractVector{Float64},
    ex,
    ndeps::AbstractVector{Float64},
)
    # Use pre-allocated buffers (NO allocations)
    # Copy x to xtrial initially
    @inbounds for i in 1:n
        xtrial[i] = x[i]
    end

    @inbounds for i in 1:n
        eps = ndeps[i]

        # Forward perturbation
        xtrial[i] = x[i] + eps
        val1 = f(n, xtrial, ex)

        # Backward perturbation
        xtrial[i] = x[i] - eps
        val2 = f(n, xtrial, ex)

        # Central difference (SIMD-friendly)
        df[i] = (val1 - val2) * inv(2.0 * eps)

        if !isfinite(df[i])
            error("non-finite finite-difference value [$i]")
        end

        # Reset (efficient)
        xtrial[i] = x[i]
    end

    return df
end
"""
    numgrad_bounded!(df, xtrial, f, n, x, ex, ndeps, lower, upper) -> Vector{Float64}

Compute the numerical gradient with **bound constraints**, using pre-allocated
buffers and central finite differences. Ensures perturbations stay within
specified lower and upper bounds.

# Arguments
- `df::AbstractVector{Float64}` — Pre-allocated gradient output buffer (length `n`)
- `xtrial::AbstractVector{Float64}` — Pre-allocated trial vector buffer (length `n`)
- `f::Function` — Objective function
- `n::Int` — Problem dimension
- `x::AbstractVector{Float64}` — Current evaluation point
- `ex` — Extra argument (may be `nothing`)
- `ndeps::AbstractVector{Float64}` — Step sizes for numerical differentiation
- `lower::Union{Nothing,AbstractVector{Float64}}` — Lower bounds (or `nothing`)
- `upper::Union{Nothing,AbstractVector{Float64}}` — Upper bounds (or `nothing`)

# Returns
- `Vector{Float64}` — Numerical gradient (same object as `df`)

# Notes
Adjusts step sizes near boundaries to remain within feasible limits.
"""
@inline function numgrad_bounded!(
    df::AbstractVector{Float64},
    xtrial::AbstractVector{Float64},
    f::Function,
    n::Int,
    x::AbstractVector{Float64},
    ex,
    ndeps::AbstractVector{Float64},
    lower::Union{Nothing,AbstractVector{Float64}},
    upper::Union{Nothing,AbstractVector{Float64}},
)
    # Copy x to xtrial initially
    @inbounds for i in 1:n
        xtrial[i] = x[i]
    end

    @inbounds for i in 1:n
        epsused = eps = ndeps[i]

        # Forward perturbation with bounds
        tmp = x[i] + eps
        if !isnothing(upper) && tmp > upper[i]
            tmp = upper[i]
            epsused = tmp - x[i]
        end
        xtrial[i] = tmp
        val1 = f(n, xtrial, ex)

        # Backward perturbation with bounds
        tmp = x[i] - eps
        if !isnothing(lower) && tmp < lower[i]
            tmp = lower[i]
            eps = x[i] - tmp
        end
        xtrial[i] = tmp
        val2 = f(n, xtrial, ex)

        # Central difference (SIMD-friendly)
        df[i] = (val1 - val2) * inv(epsused + eps)

        if !isfinite(df[i])
            error("non-finite finite-difference value [$i]")
        end

        # Reset
        xtrial[i] = x[i]
    end

    return df
end

"""
    numgrad(f, n, x, ex, ndeps; usebounds=false, lower=nothing, upper=nothing) -> Vector{Float64}

Convenient high-level wrapper for computing numerical gradients.  
Creates a temporary cache internally (single-use).  

For repeated gradient evaluations, create a `NumericalGradientCache` and use
[`numgrad_with_cache!`](@ref) for best performance.

# Example
```julia
f(n, x, ex) = sum(x.^2)
x = [1.0, 2.0, 3.0]
ndeps = fill(1e-3, 3)

# One-off computation
g = numgrad(f, 3, x, nothing, ndeps)

# Cached computation for repeated optimization steps
cache = NumericalGradientCache(3)
for iter in 1:1000
    g = numgrad_with_cache!(cache, f, 3, x, nothing, ndeps)
end
```
"""
function numgrad(
    f::Function,
    n::Int,
    x::Vector{Float64},
    ex,
    ndeps::Vector{Float64};
    usebounds::Bool = false,
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
)
    cache = NumericalGradientCache(n)

    if !usebounds
        return numgrad!(cache.df, cache.xtrial, f, n, x, ex, ndeps)
    else
        return numgrad_bounded!(cache.df, cache.xtrial, f, n, x, ex, ndeps, lower, upper)
    end
end

"""
    numgrad_with_cache!(cache, f, n, x, ex, ndeps; usebounds=false, lower=nothing, upper=nothing) -> Vector{Float64}

Compute the numerical gradient using a **pre-allocated cache**, providing the
fastest possible implementation for repeated evaluations inside optimizers
such as BFGS, L-BFGS.

# Arguments
- `cache::NumericalGradientCache` — Pre-allocated workspace
- `f::Function` — Objective function
- `n::Int` — Dimension of the problem
- `x::AbstractVector{Float64}` — Current point
- `ex` — Extra argument (may be `nothing`)
- `ndeps::AbstractVector{Float64}` — Step sizes for finite differences
- `usebounds::Bool=false` — Whether to enforce bound constraints
- `lower`, `upper` — Optional bound vectors

# Returns
- `Vector{Float64}` — Numerical gradient (references `cache.df`)

"""
@inline function numgrad_with_cache!(
    cache::NumericalGradientCache,
    f::Function,
    n::Int,
    x::AbstractVector{Float64},
    ex,
    ndeps::AbstractVector{Float64};
    usebounds::Bool = false,
    lower::Union{Nothing,AbstractVector{Float64}} = nothing,
    upper::Union{Nothing,AbstractVector{Float64}} = nothing,
)
    if !usebounds
        return numgrad!(cache.df, cache.xtrial, f, n, x, ex, ndeps)
    else
        return numgrad_bounded!(cache.df, cache.xtrial, f, n, x, ex, ndeps, lower, upper)
    end
end

"""
    numgrad(f, n, x, ex; ndeps=fill(1e-3, n), kwargs...) -> Vector{Float64}

Convenience wrapper with default step sizes.
Equivalent to calling [`numgrad`](@ref) with `ndeps = fill(1e-3, n)`.

# Example
```julia
f(n, x, ex) = sum(x.^2)
x = [1.0, 2.0, 3.0]
g = numgrad(f, 3, x, nothing)
````
"""
function numgrad(
    f::Function,
    n::Int,
    x::Vector{Float64},
    ex;
    ndeps::Vector{Float64} = fill(1e-3, n),
    usebounds::Bool = false,
    lower::Union{Nothing,Vector{Float64}} = nothing,
    upper::Union{Nothing,Vector{Float64}} = nothing,
)
    return numgrad(f, n, x, ex, ndeps;
        usebounds = usebounds, lower = lower, upper = upper)
end
