"""
    numerical_hessian(fn, x, grad=nothing; fnscale=1.0, parscale=nothing, step_sizes=nothing, kwargs...)

Compute the Hessian matrix of `fn` at `x` using finite differences of the gradient.

If an analytical gradient `grad` is provided, second derivatives are approximated by
differencing the gradient. Otherwise, each gradient evaluation is itself approximated
via central finite differences, yielding a fully numerical second-derivative estimate.
The result is symmetrized by averaging `H[i,j]` and `H[j,i]`.

# Arguments

- `fn::Function`: Scalar-valued objective function.
- `x::Vector`: Parameter vector at which to evaluate the Hessian.
- `grad::Union{Function,Nothing}`: Gradient function (optional).
- `fnscale::Number`: Function scaling factor (default: 1.0).
- `parscale::Vector`: Parameter scaling factors (default: ones).
- `step_sizes::Vector`: Finite difference step sizes (default: 0.001).

# Returns

Symmetric Hessian matrix of size `(n, n)`.

# References

- Nocedal, J. & Wright, S. J. (2006). *Numerical Optimization*, 2nd ed., Section 8.1.
  Springer.

# Examples

```julia
rosenbrock(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
H = numerical_hessian(rosenbrock, [1.0, 1.0])
```
"""
function numerical_hessian(fn, x, grad = nothing; fnscale = 1.0, parscale = nothing, step_sizes = nothing, kwargs...)
    n = length(x)
    ps = something(parscale, ones(n))
    h = something(step_sizes, fill(0.001, n))

    fn_k = let fn = fn, kw = values(kwargs)
        x -> fn(x; kw...)
    end

    grad_k = if !isnothing(grad)
        let grad = grad, kw = values(kwargs)
            x -> grad(x; kw...)
        end
    else
        nothing
    end

    # Evaluate gradient at a point in original parameter space.
    # When no analytical gradient is available, uses central finite differences
    # with perturbation h[j] * parscale[j] in each coordinate.
    function gradient_at(xp)
        if !isnothing(grad_k)
            return grad_k(xp)
        end
        g = Vector{Float64}(undef, n)
        xt = copy(xp)
        for j in 1:n
            delta_j = h[j] * ps[j]
            xt[j] = xp[j] + delta_j
            fp = fn_k(xt)
            xt[j] = xp[j] - delta_j
            fm = fn_k(xt)
            g[j] = (fp - fm) / (2 * delta_j)
            xt[j] = xp[j]
            isfinite(g[j]) || error("non-finite finite-difference value at index $j")
        end
        return g
    end

    H = zeros(n, n)
    xp = copy(x)

    # Second derivatives via gradient differencing: H[i,j] ≈ (g_j(x+δ_i) - g_j(x-δ_i)) / (2δ_i)
    for i in 1:n
        delta_i = h[i]

        xp[i] = x[i] + delta_i
        g_plus = gradient_at(xp)

        xp[i] = x[i] - delta_i
        g_minus = gradient_at(xp)

        xp[i] = x[i]  # restore

        for j in 1:n
            H[i, j] = (g_plus[j] - g_minus[j]) / (2 * delta_i)
        end
    end

    # Symmetrize: average H[i,j] and H[j,i] to reduce numerical asymmetry
    for i in 2:n, j in 1:i-1
        avg = (H[i, j] + H[j, i]) / 2
        H[i, j] = avg
        H[j, i] = avg
    end

    return H
end
