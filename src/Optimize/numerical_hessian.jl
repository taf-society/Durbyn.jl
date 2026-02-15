function compute_gradient(p, fn, grad, fnscale, parscale, step_sizes; kwargs...)
    n = length(p)
    df = zeros(n)

    if !isnothing(grad)
        x = p .* parscale
        df = grad(x; kwargs...) .* parscale ./ fnscale
        return df
    end

    x = p .* parscale
    for i in 1:n
        eps = step_sizes[i]
        xp = copy(x); xp[i] += eps * parscale[i]
        val1 = fn(xp; kwargs...) / fnscale

        xm = copy(x); xm[i] -= eps * parscale[i]
        val2 = fn(xm; kwargs...) / fnscale

        df[i] = (val1 - val2) / (2 * eps)

        if !isfinite(df[i])
            error("non-finite finite-difference value at index $(i)")
        end
    end

    return df
end


"""
    numerical_hessian(fn, x, grad=nothing; fnscale=1.0, parscale=nothing, step_sizes=nothing, kwargs...)

Compute the Hessian matrix of `fn` at `x` using finite differences.

If a gradient function `grad` is provided, the Hessian is computed by differencing
the gradient. Otherwise, the gradient is first estimated numerically via central
differences, then differenced again to obtain second derivatives. The result is
symmetrized by averaging off-diagonal elements.

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

- Nocedal, J. & Wright, S. J. (1999). *Numerical Optimization*, Chapter 8.
  Springer.

# Examples

```julia
rosenbrock(x) = 100*(x[2]-x[1]^2)^2 + (1-x[1])^2
H = numerical_hessian(rosenbrock, [1.0, 1.0])
```
"""
function numerical_hessian(fn, x, grad = nothing; fnscale = 1.0, parscale = nothing, step_sizes = nothing, kwargs...)

    npar = length(x)
    fn1(x) = fn(x; kwargs...)

    parscale = isnothing(parscale) ? ones(npar) : parscale
    step_sizes = isnothing(step_sizes) ? fill(0.001, npar) : step_sizes

    hessian = zeros(npar, npar)
    dpar = x ./ parscale

    for i in 1:npar
        eps = step_sizes[i] / parscale[i]

        dpar[i] += eps
        df1 = compute_gradient(dpar, fn1, grad, fnscale, parscale, step_sizes; kwargs...)

        dpar[i] -= 2 * eps
        df2 = compute_gradient(dpar, fn1, grad, fnscale, parscale, step_sizes; kwargs...)

        for j in 1:npar
            hessian[i, j] = fnscale * (df1[j] - df2[j]) /
                            (2 * eps * parscale[i] * parscale[j])
        end

        dpar[i] += eps
    end

    for i in 1:npar
        for j in 1:(i - 1)
            tmp = 0.5 * (hessian[i, j] + hessian[j, i])
            hessian[i, j] = hessian[j, i] = tmp
        end
    end
    return hessian
end


