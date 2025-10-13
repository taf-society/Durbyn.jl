"""
    numgrad(f, n, x, ex, ndeps; usebounds=false, lower=nothing, upper=nothing)

Compute numerical gradient using central differences.

# Arguments
- `f::Function`: Objective function, called as `f(n, x, ex)`
- `n::Int`: Dimension of the problem
- `x::Vector{Float64}`: Point at which to evaluate gradient
- `ex`: Extra argument passed to f (can be `nothing`)
- `ndeps::Vector{Float64}`: Step sizes for numerical differentiation (length n)

# Keyword Arguments
- `usebounds::Bool=false`: Whether to respect bounds
- `lower::Union{Nothing,Vector{Float64}}=nothing`: Lower bounds
- `upper::Union{Nothing,Vector{Float64}}=nothing`: Upper bounds

# Returns
- `Vector{Float64}`: Numerical gradient approximation

# Notes
This function uses central differences: df[i] = (f(x+h) - f(x-h)) / (2h)
When bounds are active, it adjusts step sizes to stay within bounds.

"""
function numgrad(f::Function, n::Int, x::Vector{Float64}, ex,
                 ndeps::Vector{Float64};
                 usebounds::Bool=false,
                 lower::Union{Nothing,Vector{Float64}}=nothing,
                 upper::Union{Nothing,Vector{Float64}}=nothing)

    df = Vector{Float64}(undef, n)
    xtrial = copy(x)

    if !usebounds
        for i in 1:n
            eps = ndeps[i]

            xtrial[i] = x[i] + eps
            val1 = f(n, xtrial, ex)

            xtrial[i] = x[i] - eps
            val2 = f(n, xtrial, ex)

            df[i] = (val1 - val2) / (2.0 * eps)

            if !isfinite(df[i])
                error("non-finite finite-difference value [$i]")
            end

            xtrial[i] = x[i]
        end
    else
        for i in 1:n
            epsused = eps = ndeps[i]

            tmp = x[i] + eps
            if !isnothing(upper) && tmp > upper[i]
                tmp = upper[i]
                epsused = tmp - x[i]
            end
            xtrial[i] = tmp
            val1 = f(n, xtrial, ex)

            tmp = x[i] - eps
            if !isnothing(lower) && tmp < lower[i]
                tmp = lower[i]
                eps = x[i] - tmp
            end
            xtrial[i] = tmp
            val2 = f(n, xtrial, ex)

            df[i] = (val1 - val2) / (epsused + eps)

            if !isfinite(df[i])
                error("non-finite finite-difference value [$i]")
            end

            xtrial[i] = x[i]
        end
    end

    return df
end


"""
    numgrad(f, n, x, ex; ndeps=fill(1e-3, n), kwargs...)

Convenience version with default step sizes.

# Example
```julia
f(n, x, ex) = sum(x.^2)
x = [1.0, 2.0, 3.0]
g = numgrad(f, 3, x, nothing)  # Returns [2.0, 4.0, 6.0] approximately
```
"""
function numgrad(f::Function, n::Int, x::Vector{Float64}, ex;
                 ndeps::Vector{Float64}=fill(1e-3, n),
                 usebounds::Bool=false,
                 lower::Union{Nothing,Vector{Float64}}=nothing,
                 upper::Union{Nothing,Vector{Float64}}=nothing)

    return numgrad(f, n, x, ex, ndeps;
                   usebounds=usebounds, lower=lower, upper=upper)
end
